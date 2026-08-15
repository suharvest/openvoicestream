"""OpenAI-compatible streaming chat backend."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from .base import LLMBackend, LLMEvent

logger = logging.getLogger(__name__)


class LLMStreamError(Exception):
    """Raised when an upstream OpenAI-compatible server signals failure
    mid-stream (typically via an SSE chunk with ``finish_reason="error"``)
    while still returning HTTP 200. Not a subclass of ``APIError`` so the
    retry logic deliberately leaves it alone — by the time we see it the
    caller has already received partial tokens and a transparent retry
    would emit duplicates."""


def _is_retryable(exc: BaseException) -> bool:
    """True when ``exc`` is a transient upstream failure worth retrying."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and 500 <= status < 600:
            return True
    return False


def _has_event_payload(ev: LLMEvent) -> bool:
    """A stream event 'counts' as first-token for retry-cutoff purposes
    when it carries either text or tool-call payload (mirrors the
    warehouse_system ``_has_payload_delta`` trick — a role-only delta
    must not lock out the retry path)."""
    if ev.kind == "text" and ev.text:
        return True
    if ev.kind == "tool_call_delta" and (
        ev.tool_call_id or ev.name or ev.arguments
    ):
        return True
    return False


class OpenAICompatBackend(LLMBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        default_params: dict[str, Any] | None = None,
        retry_on_transient: int = 1,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.default_params = dict(default_params or {})
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        # Number of *additional* attempts after the first failure. The
        # default of 1 means: 1 initial try + 1 retry = up to 2 attempts.
        self._retry_on_transient = max(0, int(retry_on_transient))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))

    async def _do_stream(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[LLMEvent]:
        """One full attempt at the upstream call. Yields :class:`LLMEvent`.

        Detects upstream SSE error frames (``finish_reason="error"``)
        emitted mid-stream and converts them into ``LLMStreamError`` so
        the caller sees a real exception instead of a silent EOS.
        """
        params: dict[str, Any] = {**self.default_params, **kw}
        # ``session`` is agent-internal state used by EdgeLLM for prefix-cache
        # control. Tool-enabled turns pass it through the common runner, but
        # it is not part of the OpenAI Chat Completions API and must never
        # reach the official SDK (or third-party compatible providers).
        params.pop("session", None)
        extra_body = params.pop("extra_body", None)
        tools = params.pop("tools", None)
        request_kwargs: dict[str, Any] = {
            "model": params.pop("model", self.model),
            "messages": messages,
            "stream": True,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        if tools:
            request_kwargs["tools"] = tools
        # Forward any remaining caller params (temperature, max_tokens...).
        request_kwargs.update(params)

        # Reset per-call cache metrics; EdgeLLMBackend overrides stream and
        # checks this attribute after the iterator drains.
        self.last_cache_metrics = None
        response = await self.client.chat.completions.create(**request_kwargs)
        async for chunk in response:
            # Surface prefix-cache metrics if the server attached them.
            # edge-llm-chat-service places them on the final chunk envelope
            # (top-level or under `metadata` / `cache_metrics`).
            try:
                me = getattr(chunk, "model_extra", None) or {}
                if not isinstance(me, dict):
                    me = {}
                cm = (
                    getattr(chunk, "cache_metrics", None)
                    or me.get("cache_metrics")
                    or (me.get("metadata") or {}).get("cache_metrics")
                )
                if cm is None and isinstance(chunk, dict):
                    cm = (
                        chunk.get("cache_metrics")
                        or (chunk.get("metadata") or {}).get("cache_metrics")
                    )
                if cm is not None:
                    if hasattr(cm, "model_dump"):
                        cm = cm.model_dump()
                    elif hasattr(cm, "dict"):
                        cm = cm.dict()
                    self.last_cache_metrics = cm
            except Exception:  # pragma: no cover - defensive
                pass
            try:
                choice0 = chunk.choices[0] if chunk.choices else None
            except (IndexError, AttributeError):
                choice0 = None
            if choice0 is None:
                continue
            delta = getattr(choice0, "delta", None)
            finish_reason = getattr(choice0, "finish_reason", None)
            # Detect upstream "I gave up" SSE frame. edge-llm's
            # api_server.py emits `finish_reason="error"` with an empty
            # delta when streaming generation blows up after HTTP 200.
            if finish_reason == "error":
                raise LLMStreamError(
                    "upstream emitted finish_reason=error mid-stream"
                )
            if delta is not None:
                content = getattr(delta, "content", None)
                if content:
                    yield LLMEvent(kind="text", text=content)
                tool_calls = getattr(delta, "tool_calls", None) or []
                for tc in tool_calls:
                    idx = getattr(tc, "index", None)
                    if idx is None:
                        idx = 0
                    fn = getattr(tc, "function", None)
                    yield LLMEvent(
                        kind="tool_call_delta",
                        tool_call_index=idx,
                        tool_call_id=getattr(tc, "id", None),
                        name=getattr(fn, "name", None) if fn else None,
                        arguments=getattr(fn, "arguments", None) if fn else None,
                    )
            if finish_reason:
                yield LLMEvent(kind="finish", finish_reason=finish_reason)

    async def stream_events(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[LLMEvent]:
        """Public event-stream entry — wraps ``_do_stream`` with
        transient-failure retry. We only retry while we have *not yet
        yielded any payload* (text or tool-call info): once the caller
        sees content we must not duplicate it on a re-attempt.

        ``LLMStreamError`` is never retried — by definition it surfaces
        after the upstream has already produced (or tried to produce)
        output, so a retry would risk double-speak.

        Internal protocol — ``_retry_disabled`` (kwarg, not user-facing):
            When ``True``, the A3 transient-failure retry loop is bypassed
            (single attempt). EdgeLLMBackend's A4 fallback path passes this
            so we don't end up with A3-retry × A4-retry = 4 upstream calls
            when prefix_cache failures arrive as 5xx.
        """
        retry_disabled = bool(kw.pop("_retry_disabled", False))
        attempts = 1 if retry_disabled else (1 + self._retry_on_transient)
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            gen = self._do_stream(messages, **kw)
            payload_seen = False
            retry_now = False
            try:
                while True:
                    try:
                        ev = await gen.__anext__()
                    except StopAsyncIteration:
                        return
                    except BaseException as e:  # noqa: BLE001
                        if (
                            not payload_seen
                            and _is_retryable(e)
                            and attempt < attempts - 1
                        ):
                            last_exc = e
                            logger.warning(
                                "LLM transient failure on attempt %d/%d: %r"
                                " — retrying in %.2fs",
                                attempt + 1,
                                attempts,
                                e,
                                self._retry_backoff_s,
                            )
                            retry_now = True
                            break
                        raise
                    if _has_event_payload(ev):
                        payload_seen = True
                    yield ev
            finally:
                await gen.aclose()
            if retry_now:
                await asyncio.sleep(self._retry_backoff_s)
                continue
            return
        # Loop fell through without a return — means we exhausted retries
        # entirely on the connect/first-token path. Re-raise the last
        # observed exception.
        if last_exc is not None:  # pragma: no cover - defensive
            raise last_exc

    async def stream(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        """Backward-compat text-only iterator. Existing callers that
        only care about assistant text get a plain ``str`` stream and
        don't need to know tool_calls exist."""
        async for ev in self.stream_events(messages, **kw):
            if ev.kind == "text" and ev.text:
                yield ev.text

    async def warmup(  # type: ignore[override]
        self,
        *,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        timeout_s: float | None = 20.0,
    ) -> dict[str, Any]:
        """Prime the persistent cloud connection before the first user turn.

        OpenAI-compatible cloud providers do not expose the local edge
        backend's prefix-cache endpoint, but one minimal streamed completion
        still removes DNS/TLS/HTTP connection setup and gives the provider a
        chance to load the selected model before a customer speaks.  The
        generated token is fully drained and discarded; it never reaches the
        conversation history or TTS pipeline.

        Warmup is deliberately fail-open.  A transient cloud outage must not
        prevent the audio agent from starting; the first real turn retains the
        normal retry path.
        """
        started = time.perf_counter()
        result: dict[str, Any] = {
            "cache_warmed": False,
            "graph_warmed": False,
            "graph_warmup_ms": 0,
        }
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "."})

        extra_body = dict(self.default_params.get("extra_body") or {})
        extra_body["enable_thinking"] = bool(enable_thinking)
        try:
            async with asyncio.timeout(timeout_s or 20.0):
                async for _ in self._do_stream(
                    messages,
                    tools=tools,
                    max_tokens=1,
                    temperature=0.0,
                    extra_body=extra_body,
                ):
                    pass
            result["graph_warmed"] = True
        except Exception as exc:
            logger.warning(
                "OpenAI-compatible LLM warmup failed: %s "
                "(first turn may pay cloud cold-start latency)",
                exc,
            )
        result["graph_warmup_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    async def aclose(self) -> None:
        try:
            await self.client.close()
        except Exception:  # pragma: no cover - best effort
            pass
