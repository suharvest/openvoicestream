"""Multi-turn LLM ↔ tool runner (thin shim over ``voxedge.turn_driver``).

Drives the dialog: stream LLM events; if the model emits
``finish_reason="tool_calls"`` accumulate the deltas, execute each
tool, append ``role:tool`` results, and re-issue the LLM call. Repeat
until a non-tool finish or the iteration cap.

The *loop algorithm* now lives in
``voxedge.engine.turn_driver.run_turn`` (provider-agnostic, no I/O — see
``docs/plans/turn-driver-unification.md`` P1). ``stream_with_tools``
remains the agent's public entrypoint: it wires the driver's seams to
the agent's private concepts (``Session`` history mirroring, allowlist →
schema, prefix-cache injection on iter >0, iteration-limit EventBus
event, the ``session`` llm kwarg, and the final-text return value). The
driver has a single behaviour (name-keyed preamble dedup + all_join
template fast-path) since P2a — the client no longer pins a strategy.

The shim mutates ``session.history`` AND the caller-supplied
``messages`` list in lock-step. On cancel or iteration-cap it rolls
both back to the pre-call anchor so the next user turn sees clean
state (no orphan ``assistant(tool_calls)`` without matching ``tool``
follow-up).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from voxedge.engine.turn_driver import run_turn

from ..session import Session
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolCallCtx:
    """Per-turn context passed to tools. Tools that need access to app
    state declare ``ctx: ToolCallCtx`` (or just ``ctx``) in their
    signature; the registry injects this on dispatch."""

    session: Session
    mode_manager: Any = None
    event_bus: Any = None
    config: Any = None


@dataclass
class _ToolCallAcc:
    """Accumulator for one tool_call's streamed deltas (per OpenAI
    index slot)."""

    id: str = ""
    name: str = ""
    arguments: str = ""


# Callback signatures.
AssistantTokenCB = Callable[[str], Awaitable[None]]
ToolStartedCB = Callable[[dict[str, Any]], Awaitable[None]]
ToolCompletedCB = Callable[[dict[str, Any], dict[str, Any], float], Awaitable[None]]
# Fired right after on_tool_started, ONLY if the dispatched tool was
# registered with a non-empty ``preamble_text``. The string is the
# verbatim preamble (e.g. "好的。") — the app wires this to its TTS
# channel. Callers that don't care can leave it None.
ToolPreambleCB = Callable[[str], Awaitable[None]]
# Fired in "template" response_mode INSTEAD of running LLM round 2.
# Receives the registered Tool.completion_text and (like preamble) is
# wired by app_mode to slv.send_text. Failures are swallowed.
ToolCompletionTextCB = Callable[[str], Awaitable[None]]


def _open_stream(llm: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any]):
    """Return an async iterator of LLMEvent regardless of which streaming
    channel the backend exposes.

    Tests + a handful of legacy callers implement only ``stream`` (text
    deltas as ``str``). Wrap those on the fly so the runner only needs
    to know about ``LLMEvent`` shape.
    """
    if hasattr(llm, "stream_events"):
        return llm.stream_events(messages, **kwargs)
    # Lazy import keeps the module loadable when LLMEvent isn't needed.
    from ..llm.base import LLMEvent

    async def _wrap():
        async for tok in llm.stream(messages, **kwargs):
            if tok:
                yield LLMEvent(kind="text", text=tok)
        yield LLMEvent(kind="finish", finish_reason="stop")

    return _wrap()


class _AgentRegistryAdapter:
    """Adapt the agent ``ToolRegistry`` to the seam the driver expects.

    The driver looks tools up via ``registry.get(name)`` (voxedge
    ``ToolRegistry`` exposes that; the agent registry stores them in
    ``_tools``). It also calls ``list_openai_tools()`` (no-arg) and
    ``dispatch(name, args, ctx)`` — both pass through unchanged. The
    no-arg ``list_openai_tools`` is never reached because the shim always
    supplies a pre-resolved ``tools_schema``.
    """

    def __init__(self, registry: ToolRegistry):
        self._r = registry

    def get(self, name: str):
        return self._r._tools.get(name)

    def list_openai_tools(self):
        return self._r.list_openai_tools()

    async def dispatch(self, name, args, ctx):
        return await self._r.dispatch(name, args, ctx)


class _AgentMessageSink:
    """``MessageSink`` mirroring driver writes into BOTH the caller's
    ``messages`` list and ``session.history`` (agent's private dual-write,
    runner.py original lock-step semantics)."""

    def __init__(self, session: Session, messages: list[dict[str, Any]]):
        self._session = session
        self._messages = messages

    def working_messages(self) -> list[dict[str, Any]]:
        return self._messages

    def add_assistant_tool_calls(self, content, tool_calls) -> None:
        self._messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })
        self._session.add_assistant_tool_calls(content, tool_calls)

    def add_assistant_text(self, content: str) -> None:
        self._session.add_assistant(content)
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._session.add_tool_result(tool_call_id, content)


class _TokenTextSink:
    """``TextSink`` routing driver text/preamble to the agent callbacks.

    * ``text`` (streamed assistant tokens) → ``on_assistant_token``.
    * ``preamble`` → ``on_tool_preamble`` (fail-open, mirrors the
      original swallow-on-raise behaviour).
    * ``flush`` is a no-op here — the agent flushes TTS in ``app_mode``'s
      ``finally`` block, never from inside the loop.
    """

    def __init__(
        self,
        on_assistant_token: "AssistantTokenCB",
        on_tool_preamble: "ToolPreambleCB | None",
    ):
        self._on_token = on_assistant_token
        self._on_preamble = on_tool_preamble

    async def text(self, s: str) -> None:
        await self._on_token(s)

    async def preamble(self, s: str) -> None:
        if self._on_preamble is None:
            return
        logger.info("tool preamble (early): text=%r", s)
        try:
            await self._on_preamble(s)
        except Exception:  # noqa: BLE001
            logger.debug("on_tool_preamble raised", exc_info=True)

    async def flush(self) -> None:
        return None


async def stream_with_tools(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    session: Session,
    registry: ToolRegistry,
    allowed_tools: set[str] | None,
    ctx: ToolCallCtx,
    max_iterations: int = 5,
    on_assistant_token: AssistantTokenCB,
    on_tool_started: ToolStartedCB | None = None,
    on_tool_preamble: ToolPreambleCB | None = None,
    on_tool_completion_text: ToolCompletionTextCB | None = None,
    on_tool_completed: ToolCompletedCB | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    first_token_timeout_s: float | None = None,
    idle_timeout_s: float | None = None,
    on_timeout: Callable[[str, float, str], BaseException] | None = None,
) -> str:
    """Run LLM ↔ tool rounds until a text-only final answer.

    Thin shim over ``voxedge.engine.turn_driver.run_turn``: the pump
    algorithm lives in the driver; this function provides the agent's
    private adaptations (allowlist→schema, prefix-cache injection on
    iter >0, iteration-limit event, ``session`` llm kwarg, dual-write
    history, final-text return). The driver has a single behaviour since
    P2a (name-keyed preamble dedup + all_join template fast-path).

    Returns the final assistant text (also appended to
    ``session.history``). Mutates both ``session.history`` and the
    caller's ``messages`` list in lock-step.

    On cancel / iteration-cap / error: rolls back any messages added
    during this call so ``session.history`` stays strict-valid.
    """
    tools_schema = registry.list_openai_tools(allowed_tools) or None
    rollback_anchor = len(session.history)
    # ``messages`` typically looks like ``[system, *session.history]``.
    # When we mirror a rollback we need to truncate ``messages`` to the
    # same logical anchor: ``messages_offset`` is the count of
    # non-history prefix items (1 for system; 0 if absent).
    messages_offset = max(0, len(messages) - rollback_anchor)

    # Wrap the backend so the driver's ``llm.stream_events(...)`` works
    # even for legacy ``stream``-only test backends.
    llm_for_driver = _ShimLLM(llm)

    base_kwargs: dict[str, Any] = dict(llm_kwargs or {})
    base_kwargs["session"] = session

    def _params_for_round(iter_idx: int) -> dict[str, Any]:
        # iter 0: caller kwargs verbatim. iter >0: also ask the server to
        # save the (grown) prefix KV — mirrors the legacy A1 behaviour
        # (runner.py:149-152). The caller's ``extra_body`` is preserved.
        if iter_idx == 0:
            return {}
        caller_extra = dict(base_kwargs.get("extra_body") or {})
        caller_extra.setdefault("save_system_prompt_kv_cache", True)
        return {"extra_body": caller_extra}

    def _on_iteration_limit() -> None:
        # Roll back the partial last tool round (no terminal assistant text)
        # so it doesn't haunt future turns, then emit the EventBus event.
        logger.warning(
            "tool iteration cap reached (%d); rolling back", max_iterations,
        )
        dropped = session.rollback_to(rollback_anchor)
        del messages[rollback_anchor + messages_offset:]
        bus = getattr(ctx, "event_bus", None)
        if bus is not None:
            try:
                bus.emit(
                    "on_tool_iteration_limit",
                    {
                        "iterations": max_iterations,
                        "dropped": dropped,
                        "sid": session.sid,
                    },
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("event_bus emit failed", exc_info=True)

    def _on_template_misconfig(tool_name: str) -> None:
        logger.warning(
            "tool %r declared response_mode=template with empty "
            "completion_text; falling back to await (running LLM round 2). "
            "Set a completion_text in the tool definition to suppress this "
            "fallback.",
            tool_name,
        )

    try:
        final = await run_turn(
            llm=llm_for_driver,
            registry=_AgentRegistryAdapter(registry),
            msg_sink=_AgentMessageSink(session, messages),
            text_sink=_TokenTextSink(on_assistant_token, on_tool_preamble),
            should_abort=lambda: False,
            ctx=ctx,
            llm_params=base_kwargs,
            max_rounds=max_iterations,
            tools_schema=tools_schema,
            llm_params_for_round=_params_for_round,
            on_iteration_limit=_on_iteration_limit,
            on_tool_started=(
                _wrap_tool_started(on_tool_started)
                if on_tool_started is not None else None
            ),
            on_tool_completed=(
                _wrap_tool_completed(on_tool_completed)
                if on_tool_completed is not None else None
            ),
            first_token_timeout_s=first_token_timeout_s,
            idle_timeout_s=idle_timeout_s,
            on_timeout=on_timeout,
            reraise_errors=True,
            record_template_text=True,
            completion_text_cb=(
                _wrap_completion_text(on_tool_completion_text)
                if on_tool_completion_text is not None else None
            ),
            on_template_misconfig=_on_template_misconfig,
        )
        return final or ""
    except asyncio.CancelledError:
        # Truncate both session.history AND the caller's local messages
        # list, otherwise the next turn sees mismatched state.
        dropped = session.rollback_to(rollback_anchor)
        del messages[rollback_anchor + messages_offset:]
        logger.info("tool round cancelled, rolled back %d messages", dropped)
        raise
    except BaseException:
        # Any non-cancel exception escaping after we appended
        # assistant(tool_calls) + tool result messages would pin an
        # incomplete tool round in history. Roll back symmetrically with
        # cancel, then re-raise so the caller's error path still fires.
        dropped = session.rollback_to(rollback_anchor)
        del messages[rollback_anchor + messages_offset:]
        if dropped:
            logger.info(
                "tool round aborted by exception, rolled back %d messages",
                dropped,
            )
        raise


def _wrap_tool_started(cb: "ToolStartedCB") -> "ToolStartedCB":
    """Wrap ``on_tool_started`` so a raise is swallowed (fail-open),
    matching the original runner behaviour."""
    async def _fn(tc: dict[str, Any]) -> None:
        try:
            await cb(tc)
        except Exception:  # noqa: BLE001
            logger.debug("on_tool_started raised", exc_info=True)
    return _fn


def _wrap_tool_completed(cb: "ToolCompletedCB") -> "ToolCompletedCB":
    async def _fn(tc: dict[str, Any], result: Any, dt_ms: float) -> None:
        try:
            await cb(tc, result, dt_ms)
        except Exception:  # noqa: BLE001
            logger.debug("on_tool_completed raised", exc_info=True)
    return _fn


def _wrap_completion_text(cb: "ToolCompletionTextCB") -> "ToolCompletionTextCB":
    async def _fn(text: str) -> None:
        try:
            await cb(text)
        except Exception:  # noqa: BLE001
            logger.debug("on_tool_completion_text raised", exc_info=True)
    return _fn


class _ShimLLM:
    """Wrap a backend so ``stream_events`` always exists.

    The driver only calls ``stream_events(messages, tools=..., **params)``.
    Legacy/test backends that implement only ``stream`` (text deltas as
    ``str``) get adapted here (mirrors the old ``_open_stream`` helper)."""

    def __init__(self, llm: Any):
        self._llm = llm

    def stream_events(self, messages: list[dict[str, Any]], **kwargs: Any):
        return _open_stream(self._llm, messages, kwargs)

