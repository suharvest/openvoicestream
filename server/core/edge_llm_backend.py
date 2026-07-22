"""Product-side edge-llm LLM backend adapter for the voxedge tool loop.

A thin subclass of voxedge's generic
:class:`voxedge.backends.llm.OpenAICompatBackend` (the SSE → ``LLMEvent`` parse
loop now lives there). This layer adds only the edge-llm-specific concerns that
do NOT belong in the reusable engine:

  * endpoint resolution from product env (``EDGE_LLM_BASE_URL`` /
    ``EDGE_LLM_CHAT_URL``) — :func:`edge_llm_base_url`;
  * edge-llm request flags (``save_system_prompt_kv_cache`` /
    ``return_cache_metrics`` / ``enable_thinking``) injected into the chat body.

Used ONLY on the ``OVS_V2V_SERVER_LOOP`` path (spec §2 step 2,
``docs/specs/tool-calling-engine-migration.md``). The default ``/v2v`` path
(client-driven LLM, flag off) never constructs this, so it stays a strict
additive, zero-behaviour-change component.

The richer agent backend (``agent/ovs_agent/llm/{openai_compat,edge_llm}.py``,
prefix-cache session latch + warmup, on the ``openai`` SDK) is a separate
process and is intentionally not folded in here — only the wire contract is
shared, now via the voxedge base.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from voxedge.backends.llm import OpenAICompatBackend

__all__ = ["EdgeLLMBackend", "edge_llm_base_url"]


def edge_llm_base_url() -> str:
    """Resolve the edge-llm chat endpoint base URL from env/profile.

    ``EDGE_LLM_BASE_URL`` is the explicit override; falls back to
    ``EDGE_LLM_CHAT_URL`` then a sane localhost default. The value is the
    OpenAI-compatible root (with or without a trailing ``/v1``); the base class
    normalizes to ``.../v1/chat/completions``.
    """
    return (
        os.environ.get("EDGE_LLM_BASE_URL")
        or os.environ.get("EDGE_LLM_CHAT_URL")
        or "http://127.0.0.1:8000/v1"
    )


class EdgeLLMBackend(OpenAICompatBackend):
    """OpenAI-compatible edge-llm backend: voxedge base + edge-llm cache flags."""

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
        super().__init__(
            base_url or edge_llm_base_url(),
            model=model,
            api_key=api_key,
            default_params=default_params,
            request_timeout_s=request_timeout_s,
        )
        self.enable_thinking = bool(enable_thinking)

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Generic OpenAI body + edge-llm cache flags.

        ``save_system_prompt_kv_cache`` / ``return_cache_metrics`` /
        ``enable_thinking`` are the tensorrt-edge-llm cold-path defaults the
        chat route reads top-level; ``setdefault`` lets a caller override per
        request. Matches the agent edge_llm.py ``_build_extra_body`` cold path.
        """
        body = super()._build_body(messages, tools, params)
        body.setdefault("save_system_prompt_kv_cache", True)
        body.setdefault("return_cache_metrics", True)
        body.setdefault("enable_thinking", self.enable_thinking)
        return body
