"""Product-side edge-llm LLM backend adapter for the voxedge tool loop.

Implements voxedge's :class:`voxedge.backends.base.LLMBackend` contract
(``stream_events(messages, tools=...) -> AsyncIterator[LLMEvent]``) by POSTing
to an OpenAI-compatible edge-llm chat service
(``edge-llm-chat-service`` / tensorrt-edge-llm ``/v1/chat/completions``).

This is the server-side LLM hop used ONLY on the new
``OVS_V2V_SERVER_LOOP`` path (spec §2 step 2, ``docs/specs/
tool-calling-engine-migration.md``). The existing ``/v2v`` default path
(client-driven LLM, ``OVS_V2V_SERVER_LOOP`` off) never constructs this — so it
is a strict additive, zero-behavior-change component.

Why httpx, not the ``openai`` SDK: the agent's reference backend
(``agent/openvoicestream_agent/llm/{edge_llm,openai_compat}.py``) uses
``AsyncOpenAI``, but the product process deliberately does not carry the
``openai`` dependency. We replicate the same OpenAI-compatible streaming SSE
request/response contract over ``httpx`` (which the agent's ``edge_llm.py``
warmup already uses), so the wire behavior — request shape (temperature /
max_tokens / tools / ``enable_thinking`` + edge-llm ``prefix_cache`` /
``save_system_prompt_kv_cache`` extra-body flags) and the text /
``tool_call_delta`` / ``finish`` event parsing — matches the agent backend.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional

import httpx

from voxedge.backends.base import LLMBackend, LLMEvent

logger = logging.getLogger(__name__)


# Substrings that mark an upstream prefix_cache-only failure (mirrors the agent
# edge_llm.py heuristic). Currently informational; the server loop does not yet
# carry per-session prefix-cache latch state, so a failure simply surfaces.
_PREFIX_CACHE_MARKERS = (
    "prefix_cache",
    "prefix cache",
    "kv cache",
    "kv_cache",
    "kv mismatch",
    "prefix_messages",
)


def edge_llm_base_url() -> str:
    """Resolve the edge-llm chat endpoint base URL from env/profile.

    ``EDGE_LLM_BASE_URL`` is the explicit override; falls back to
    ``EDGE_LLM_CHAT_URL`` then a sane localhost default. The value is the
    OpenAI-compatible root (with or without a trailing ``/v1``); the adapter
    normalizes to ``.../v1/chat/completions``.
    """
    return (
        os.environ.get("EDGE_LLM_BASE_URL")
        or os.environ.get("EDGE_LLM_CHAT_URL")
        or "http://127.0.0.1:8000/v1"
    )


class EdgeLLMBackend(LLMBackend):
    """OpenAI-compatible streaming edge-llm backend over httpx.

    Satisfies voxedge ``LLMBackend.stream_events`` — the only method the
    engine tool pump (``Session._llm_turn_with_tools``) calls. ``stream`` is
    the back-compat text-only filter the ABC also declares.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        model: str = "qwen3",
        api_key: str = "edge-llm",
        default_params: Optional[dict[str, Any]] = None,
        enable_thinking: bool = False,
        request_timeout_s: float = 60.0,
    ) -> None:
        self.base_url = (base_url or edge_llm_base_url()).rstrip("/")
        self.model = model
        self.api_key = api_key
        # Request params forwarded to the chat call (temperature / max_tokens /
        # top_p ...). The voxedge engine also forwards its own ``llm_params``
        # per stream_events call; both merge here (per-call wins).
        self.default_params = dict(default_params or {})
        self.enable_thinking = bool(enable_thinking)
        self.request_timeout_s = float(request_timeout_s)
        self._chat_url = self._resolve_chat_url(self.base_url)
        self._client: Optional[httpx.AsyncClient] = None

    @staticmethod
    def _resolve_chat_url(base: str) -> str:
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout_s)
        return self._client

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the OpenAI-compatible chat request body.

        Mirrors ``OpenAICompatBackend._do_stream`` + ``EdgeLLMBackend.
        _build_extra_body`` request construction: top-level OpenAI params
        (model / messages / stream / temperature / max_tokens / tools) plus
        edge-llm's ``extra_body`` flags (``save_system_prompt_kv_cache`` /
        ``return_cache_metrics`` / ``enable_thinking``) flattened — the
        tensorrt-edge-llm server reads these top-level on the chat route.
        """
        p = dict(params)
        # extra_body from the agent path is flattened into the top-level JSON;
        # the edge-llm server accepts these as top-level keys.
        extra_body = p.pop("extra_body", None) or {}
        body: dict[str, Any] = {
            "model": p.pop("model", self.model),
            "messages": messages,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
        # edge-llm cache flags (cold-path: cache the system prompt KV, report
        # metrics). Matches agent edge_llm.py:_build_extra_body cold path.
        body.setdefault("save_system_prompt_kv_cache", True)
        body.setdefault("return_cache_metrics", True)
        body.setdefault("enable_thinking", self.enable_thinking)
        # Caller-provided params (temperature / max_tokens / top_p ...).
        for k, v in p.items():
            body[k] = v
        for k, v in extra_body.items():
            body[k] = v
        return body

    async def stream_events(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        session: Any = None,  # accepted for agent-parity; unused server-side
        **kw: Any,
    ) -> AsyncIterator[LLMEvent]:
        """Stream OpenAI-compatible SSE chunks as voxedge ``LLMEvent``s.

        Yields ``kind="text"`` for content deltas, ``kind="tool_call_delta"``
        for streamed ``delta.tool_calls`` (per OpenAI tool-call index), and
        ``kind="finish"`` for each ``finish_reason`` — exactly the event shape
        ``OpenAICompatBackend._do_stream`` produces (the contract the engine
        tool pump consumes)."""
        params = {**self.default_params, **kw}
        body = self._build_body(messages, tools, params)
        client = self._ensure_client()
        async with client.stream("POST", self._chat_url, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0]
                delta = choice0.get("delta") or {}
                finish_reason = choice0.get("finish_reason")
                if finish_reason == "error":
                    raise RuntimeError(
                        "edge-llm emitted finish_reason=error mid-stream"
                    )
                content = delta.get("content")
                if content:
                    yield LLMEvent(kind="text", text=content)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index")
                    if idx is None:
                        idx = 0
                    fn = tc.get("function") or {}
                    yield LLMEvent(
                        kind="tool_call_delta",
                        tool_call_index=idx,
                        tool_call_id=tc.get("id"),
                        name=fn.get("name"),
                        arguments=fn.get("arguments"),
                    )
                if finish_reason:
                    yield LLMEvent(kind="finish", finish_reason=finish_reason)

    async def stream(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> AsyncIterator[str]:
        """Back-compat text-only iterator (ABC requirement)."""
        async for ev in self.stream_events(messages, **kw):
            if ev.kind == "text" and ev.text:
                yield ev.text

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover - best effort
                pass
            self._client = None


__all__ = ["EdgeLLMBackend", "edge_llm_base_url"]
