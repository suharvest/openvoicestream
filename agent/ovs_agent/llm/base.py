"""Abstract LLM backend interface.

Re-exports the canonical :class:`~voxedge.backends.base.LLMBackend` /
:class:`~voxedge.backends.base.LLMEvent` from voxedge — the single source of
truth for the streaming LLM contract (``stream_events`` → text / tool_call_delta
/ finish, plus ``stream`` / ``warmup`` / ``aclose`` lifecycle hooks).

This used to define a parallel copy of that ABC; voxedge's ``backends/base.py``
reproduced it (to stay free of the agent package's openai/httpx deps). Now that
the agent depends on voxedge, the duplicate is collapsed here so both layers
share one definition. Agent backends (``openai_compat`` / ``edge_llm`` /
``noop``) keep importing ``from .base import LLMBackend, LLMEvent`` unchanged.
"""
from __future__ import annotations

from voxedge.backends.base import LLMBackend, LLMEvent

__all__ = ["LLMBackend", "LLMEvent"]
