"""Tests for the multi-turn LLM ↔ tool runner."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ovs_agent.llm import LLMEvent
from ovs_agent.session import Session
from ovs_agent.tools import ToolCallCtx, ToolRegistry, stream_with_tools


class _FakeLLM:
    """Stub backend that returns a scripted sequence of LLMEvent lists,
    one per stream_events() call. Records the kwargs of each call."""

    def __init__(self, script: list[list[LLMEvent]]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []
        self._call_idx = 0

    async def stream_events(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ):
        # Snapshot the kwargs the runner passed in.
        self.calls.append({
            "messages": [dict(m) for m in messages],
            "kwargs": dict(kwargs),
        })
        if self._call_idx >= len(self._script):
            raise RuntimeError("fake LLM script exhausted")
        events = self._script[self._call_idx]
        self._call_idx += 1
        for ev in events:
            yield ev


def _text(t: str) -> LLMEvent:
    return LLMEvent(kind="text", text=t)


def _tc(idx: int, *, id: str | None = None, name: str | None = None,
        arguments: str | None = None) -> LLMEvent:
    return LLMEvent(
        kind="tool_call_delta",
        tool_call_index=idx,
        tool_call_id=id,
        name=name,
        arguments=arguments,
    )


def _finish(reason: str) -> LLMEvent:
    return LLMEvent(kind="finish", finish_reason=reason)


def _make_ctx(session: Session) -> ToolCallCtx:
    return ToolCallCtx(session=session)


# ── (a) text-only, no tools called ────────────────────────────────────


@pytest.mark.asyncio
async def test_text_only_no_tools():
    session = Session()
    registry = ToolRegistry()
    llm = _FakeLLM([[_text("hello"), _text(" world"), _finish("stop")]])

    tokens: list[str] = []

    async def on_tok(t):
        tokens.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools=None,
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    assert final == "hello world"
    assert tokens == ["hello", " world"]
    assert session.history == [
        {"role": "assistant", "content": "hello world"},
    ]
    assert len(llm.calls) == 1


# ── (b) one tool round-trip ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_tool_round_trip():
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def time_now() -> dict:
        return {"now": "2026-01-01T00:00:00"}

    llm = _FakeLLM([
        # Iteration 1: emit a tool_call
        [
            _tc(0, id="c1", name="time_now", arguments=""),
            _tc(0, arguments="{}"),
            _finish("tool_calls"),
        ],
        # Iteration 2: emit the final text
        [_text("it is morning"), _finish("stop")],
    ])

    tokens: list[str] = []

    async def on_tok(t):
        tokens.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"time_now"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    assert final == "it is morning"
    # Session history: assistant_tc + tool_result + assistant_text
    assert len(session.history) == 3
    assert session.history[0]["role"] == "assistant"
    assert session.history[0]["content"] is None
    assert session.history[0]["tool_calls"][0]["function"]["name"] == "time_now"
    assert session.history[1]["role"] == "tool"
    assert session.history[1]["tool_call_id"] == "c1"
    assert "2026-01-01" in session.history[1]["content"]
    assert session.history[2] == {"role": "assistant", "content": "it is morning"}
    # messages list mirrored:
    assert msgs[0]["role"] == "system"
    assert len(msgs) == 1 + 3  # system + same 3 history entries


# ── (c) two consecutive tool rounds ───────────────────────────────────


@pytest.mark.asyncio
async def test_two_tool_rounds():
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def step(n: int) -> dict:
        return {"n": n}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="step", arguments='{"n":1}'), _finish("tool_calls")],
        [_tc(0, id="c2", name="step", arguments='{"n":2}'), _finish("tool_calls")],
        [_text("done"), _finish("stop")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"step"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    assert final == "done"
    # 2 tc + 2 tool + 1 text = 5 entries
    assert len(session.history) == 5
    assert session.history[-1]["content"] == "done"


# ── (d) cancel during tool dispatch rolls back BOTH lists ─────────────


@pytest.mark.asyncio
async def test_cancel_during_tool_dispatch_rolls_back():
    session = Session()
    session.add_user("pre-existing user msg")
    anchor_history = list(session.history)
    registry = ToolRegistry()

    @registry.tool(timeout_s=5.0)
    async def slow() -> dict:
        await asyncio.sleep(10)
        return {}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="slow", arguments="{}"), _finish("tool_calls")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}, *session.history]

    async def runner_coro():
        return await stream_with_tools(
            llm, msgs,
            session=session, registry=registry, allowed_tools={"slow"},
            ctx=_make_ctx(session), on_assistant_token=on_tok,
        )

    task = asyncio.create_task(runner_coro())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Session rolled back to pre-runner state.
    assert session.history == anchor_history
    # Messages list mirrored — only system + pre-existing user msg.
    assert len(msgs) == 1 + len(anchor_history)
    assert msgs[0]["role"] == "system"


# ── (e) iteration cap rolls back ──────────────────────────────────────


@pytest.mark.asyncio
async def test_iteration_cap_rolls_back():
    session = Session()
    session.add_user("u1")
    anchor_history = list(session.history)
    registry = ToolRegistry()

    @registry.tool()
    def loop() -> dict:
        return {"again": True}

    # Always issue a tool_call → trigger cap after max_iterations.
    def iter_script():
        return [
            _tc(0, id="c", name="loop", arguments="{}"),
            _finish("tool_calls"),
        ]

    llm = _FakeLLM([iter_script() for _ in range(5)])

    bus_events: list[tuple[str, dict]] = []

    class _Bus:
        def emit(self, name, data):
            bus_events.append((name, data))

    ctx = ToolCallCtx(session=session, event_bus=_Bus())

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}, *session.history]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"loop"},
        ctx=ctx, on_assistant_token=on_tok, max_iterations=3,
    )
    assert final == ""
    # Session rolled back to anchor.
    assert session.history == anchor_history
    # Messages list mirrored.
    assert len(msgs) == 1 + len(anchor_history)
    # iteration_limit event emitted.
    names = [n for n, _ in bus_events]
    assert "on_tool_iteration_limit" in names


# ── (f) invalid args JSON → error result, loop continues ──────────────


@pytest.mark.asyncio
async def test_invalid_args_json_continues_loop():
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def f(x: int) -> dict:
        return {"x": x}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="f", arguments="not-json"), _finish("tool_calls")],
        [_text("recovered"), _finish("stop")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"f"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    assert final == "recovered"
    # Tool result should carry the JSON-decode error.
    tool_msg = next(m for m in session.history if m.get("role") == "tool")
    import json
    body = json.loads(tool_msg["content"])
    assert body["success"] is False
    assert "invalid arguments JSON" in body["error"]


# ── (g) iter >0 keeps prefix_cache and asks to save new prefix (A1) ───


@pytest.mark.asyncio
async def test_iter_gt_zero_keeps_prefix_cache_and_saves():
    """A1: iter >0 must NOT disable prefix_cache (server cache supports
    multi-turn prefix match) and SHOULD ask the server to save the
    grown prefix for the next iter."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def f() -> dict:
        return {"ok": True}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="f", arguments="{}"), _finish("tool_calls")],
        [_text("done"), _finish("stop")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    _ = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"f"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
    )
    # First call: no extra_body forced; caller's llm_kwargs was empty.
    first_kw = llm.calls[0]["kwargs"]
    assert "extra_body" not in first_kw or "prefix_cache" not in (
        first_kw.get("extra_body") or {}
    )
    # Second call (iter >0): runner must NOT have forced prefix_cache=False,
    # and SHOULD have asked the server to save the larger prefix.
    second_kw = llm.calls[1]["kwargs"]
    extra = second_kw["extra_body"]
    assert extra.get("prefix_cache") is not False
    assert extra["save_system_prompt_kv_cache"] is True


# ── (h) first-token + idle timeout kwargs ─────────────────────────────


@pytest.mark.asyncio
async def test_first_token_timeout_raises():
    session = Session()
    registry = ToolRegistry()

    class _HangLLM:
        async def stream_events(self, messages, **kw):
            await asyncio.sleep(10)
            if False:  # pragma: no cover
                yield None

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    with pytest.raises(asyncio.TimeoutError):
        await stream_with_tools(
            _HangLLM(), msgs,
            session=session, registry=registry, allowed_tools=None,
            ctx=_make_ctx(session), on_assistant_token=on_tok,
            first_token_timeout_s=0.05,
            idle_timeout_s=1.0,
        )


@pytest.mark.asyncio
async def test_first_token_timeout_invokes_on_timeout_hook():
    """Custom on_timeout should be called with (kind, t_used, partial)
    and its return value is raised instead of TimeoutError."""
    session = Session()
    registry = ToolRegistry()

    class _HangLLM:
        async def stream_events(self, messages, **kw):
            await asyncio.sleep(10)
            if False:  # pragma: no cover
                yield None

    class _MyTimeoutError(RuntimeError):
        def __init__(self, kind, t, partial):
            super().__init__(f"{kind}@{t}")
            self.kind = kind

    seen: list[tuple[str, float, str]] = []

    def _on_timeout(kind, t, partial):
        seen.append((kind, t, partial))
        return _MyTimeoutError(kind, t, partial)

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    with pytest.raises(_MyTimeoutError) as exc_info:
        await stream_with_tools(
            _HangLLM(), msgs,
            session=session, registry=registry, allowed_tools=None,
            ctx=_make_ctx(session), on_assistant_token=on_tok,
            first_token_timeout_s=0.05,
            idle_timeout_s=1.0,
            on_timeout=_on_timeout,
        )
    assert exc_info.value.kind == "first_token"
    assert seen and seen[0][0] == "first_token"


@pytest.mark.asyncio
async def test_idle_timeout_after_first_token():
    """After a payload event, a long gap should raise an idle (not
    first-token) timeout."""
    from ovs_agent.llm import LLMEvent
    session = Session()
    registry = ToolRegistry()

    class _SlowAfterFirstLLM:
        async def stream_events(self, messages, **kw):
            yield LLMEvent(kind="text", text="x")
            await asyncio.sleep(10)
            if False:  # pragma: no cover
                yield None

    kinds: list[str] = []

    def _on_timeout(kind, t, partial):
        kinds.append(kind)
        return RuntimeError(f"timeout:{kind}")

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    with pytest.raises(RuntimeError):
        await stream_with_tools(
            _SlowAfterFirstLLM(), msgs,
            session=session, registry=registry, allowed_tools=None,
            ctx=_make_ctx(session), on_assistant_token=on_tok,
            first_token_timeout_s=5.0,
            idle_timeout_s=0.05,
            on_timeout=_on_timeout,
        )
    assert kinds == ["stream_idle"]


@pytest.mark.asyncio
async def test_iter_gt_zero_preserves_caller_extra_body():
    """A caller-supplied extra_body must survive the A1
    save_system_prompt_kv_cache injection on iter >0."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool()
    def f() -> dict:
        return {}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="f", arguments="{}"), _finish("tool_calls")],
        [_text("ok"), _finish("stop")],
    ])

    async def on_tok(t):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"f"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
        llm_kwargs={"extra_body": {"custom_flag": "keep_me"}},
    )
    second_kw = llm.calls[1]["kwargs"]
    assert second_kw["extra_body"]["custom_flag"] == "keep_me"
    assert second_kw["extra_body"]["save_system_prompt_kv_cache"] is True
    # And we must NOT have stomped the cache lookup off.
    assert second_kw["extra_body"].get("prefix_cache") is not False


# ── (k) per-tool preamble_text metadata fires on_tool_preamble ────────


@pytest.mark.asyncio
async def test_tool_preamble_text_fires_callback():
    """A tool registered with ``preamble_text="..."`` must trigger the
    ``on_tool_preamble`` callback after ``on_tool_started`` and before
    the tool body runs. Tools without it must NOT fire the callback."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool(preamble_text="好的。")
    def wave_hand() -> dict:
        return {"ok": True}

    @registry.tool()
    def silent_tool() -> dict:
        return {"ok": True}

    llm = _FakeLLM([
        # Round 1: model calls wave_hand → expect preamble fired
        [
            _tc(0, id="c1", name="wave_hand", arguments="{}"),
            _finish("tool_calls"),
        ],
        # Round 2: model calls silent_tool → expect NO preamble
        [
            _tc(0, id="c2", name="silent_tool", arguments="{}"),
            _finish("tool_calls"),
        ],
        # Round 3: model finishes with text
        [_text("done"), _finish("stop")],
    ])

    events: list[tuple[str, Any]] = []

    async def on_tok(t):
        events.append(("tok", t))

    async def on_started(tc):
        events.append(("started", tc["function"]["name"]))

    async def on_preamble(text):
        events.append(("preamble", text))

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session,
        registry=registry,
        allowed_tools=None,
        ctx=_make_ctx(session),
        on_assistant_token=on_tok,
        on_tool_started=on_started,
        on_tool_preamble=on_preamble,
    )
    assert final == "done"

    # Expected ordering (after early-fire optimisation):
    #   preamble:好的。 (fired on first tool_call name delta),
    #   started:wave_hand, started:silent_tool, tok:done
    started_events = [e for e in events if e[0] == "started"]
    preamble_events = [e for e in events if e[0] == "preamble"]
    assert started_events == [("started", "wave_hand"), ("started", "silent_tool")]
    # Preamble fires exactly once per tool slot, regardless of whether
    # the trigger was early (during streaming) or dispatch-time fallback.
    assert preamble_events == [("preamble", "好的。")]

    # Ordering check: preamble fires EARLY now — before either
    # on_tool_started, as soon as the tool_call name delta arrives.
    idx_wave_started = events.index(("started", "wave_hand"))
    idx_preamble = events.index(("preamble", "好的。"))
    idx_silent_started = events.index(("started", "silent_tool"))
    assert idx_preamble < idx_wave_started < idx_silent_started


@pytest.mark.asyncio
async def test_tool_preamble_backward_compat_no_callback():
    """Callers that don't pass ``on_tool_preamble`` must still work
    (backward compat — voice-arm v1 and the rest of the test suite
    rely on this default)."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool(preamble_text="hello.")
    def t() -> dict:
        return {"ok": True}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="t", arguments="{}"), _finish("tool_calls")],
        [_text("k"), _finish("stop")],
    ])

    async def on_tok(_):
        pass

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    # No on_tool_preamble kwarg — must not raise.
    final = await stream_with_tools(
        llm, msgs,
        session=session,
        registry=registry,
        allowed_tools=None,
        ctx=_make_ctx(session),
        on_assistant_token=on_tok,
    )
    assert final == "k"


@pytest.mark.asyncio
async def test_tool_preamble_callback_failure_is_swallowed():
    """If ``on_tool_preamble`` raises, the tool dispatch must continue
    (fail-open semantics — TTS hiccups must never break tool calls)."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool(preamble_text="oops.")
    def t() -> dict:
        return {"ok": True}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="t", arguments="{}"), _finish("tool_calls")],
        [_text("done"), _finish("stop")],
    ])

    async def on_tok(_):
        pass

    async def on_preamble(_text):
        raise RuntimeError("simulated SLV drop")

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session,
        registry=registry,
        allowed_tools=None,
        ctx=_make_ctx(session),
        on_assistant_token=on_tok,
        on_tool_preamble=on_preamble,
    )
    assert final == "done"


@pytest.mark.asyncio
async def test_response_mode_await_runs_llm_round_2():
    """Default ``await`` mode (backward compat) — runner still calls
    LLM round 2 after dispatch, final text comes from LLM."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool()  # response_mode defaults to "await"
    def wave() -> dict:
        return {"success": True}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="wave", arguments="{}"), _finish("tool_calls")],
        [_text("waved hello"), _finish("stop")],
    ])

    completion_texts: list[str] = []

    async def on_tok(_):
        pass

    async def on_completion_text(t):
        completion_texts.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"wave"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
        on_tool_completion_text=on_completion_text,
    )
    # LLM round 2 produced the final text
    assert final == "waved hello"
    # No completion_text emitted (mode is "await")
    assert completion_texts == []
    # 2 LLM calls (round 1 + round 2)
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_response_mode_template_skips_llm_round_2():
    """``template`` mode — runner skips LLM round 2 and emits fixed
    completion_text via the callback. Final text matches completion_text."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool(response_mode="template", completion_text="挥完了")
    def wave() -> dict:
        return {"success": True}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="wave", arguments="{}"), _finish("tool_calls")],
        # 2nd entry exists only to assert it's NOT consumed
        [_text("SHOULD NOT BE USED"), _finish("stop")],
    ])

    completion_texts: list[str] = []

    async def on_tok(_):
        pass

    async def on_completion_text(t):
        completion_texts.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"wave"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
        on_tool_completion_text=on_completion_text,
    )
    assert final == "挥完了"
    assert completion_texts == ["挥完了"]
    # ONLY 1 LLM call — round 2 was skipped
    assert len(llm.calls) == 1
    # Session history must have the synth assistant text appended for
    # next-turn coherence.
    assert session.history[-1] == {"role": "assistant", "content": "挥完了"}


@pytest.mark.asyncio
async def test_response_mode_template_failure_falls_through_to_llm():
    """If a template-mode tool returns success=False, runner must NOT
    emit completion_text — instead run LLM round 2 so the model can
    react to the error."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool(response_mode="template", completion_text="ok")
    def wave() -> dict:
        return {"success": False, "error": "boom"}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="wave", arguments="{}"), _finish("tool_calls")],
        [_text("sorry, failed"), _finish("stop")],
    ])

    completion_texts: list[str] = []

    async def on_tok(_):
        pass

    async def on_completion_text(t):
        completion_texts.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"wave"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
        on_tool_completion_text=on_completion_text,
    )
    assert final == "sorry, failed"
    assert completion_texts == []  # not emitted on failure
    assert len(llm.calls) == 2  # LLM round 2 happened


@pytest.mark.asyncio
async def test_response_mode_template_empty_completion_text_falls_back_to_await(caplog):
    """Misconfiguration: response_mode=template but completion_text="".

    Without the fallback, the runner would skip LLM round 2 and return
    "" — the user gets no spoken reply at all and session history is
    missing an assistant turn. Fix: detect empty completion_text, log a
    warning, and let LLM round 2 run as if response_mode were "await".
    """
    import logging

    session = Session()
    registry = ToolRegistry()

    @registry.tool(response_mode="template", completion_text="")
    def wave() -> dict:
        return {"success": True}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="wave", arguments="{}"), _finish("tool_calls")],
        [_text("sure, waved"), _finish("stop")],
    ])

    completion_texts: list[str] = []

    async def on_tok(_):
        pass

    async def on_completion_text(t):
        completion_texts.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    with caplog.at_level(logging.WARNING, logger="ovs_agent.tools.runner"):
        final = await stream_with_tools(
            llm, msgs,
            session=session, registry=registry, allowed_tools={"wave"},
            ctx=_make_ctx(session), on_assistant_token=on_tok,
            on_tool_completion_text=on_completion_text,
        )
    # Fell back to await: LLM round 2 ran and its text is returned.
    assert final == "sure, waved"
    assert completion_texts == []  # empty ctext was never emitted
    assert len(llm.calls) == 2  # round 2 happened
    # Warning was logged so operator can fix the misconfig.
    assert any(
        "completion_text" in r.message and "falling back to await" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_response_mode_parallel_runs_llm_round_2_on_fast_result():
    """``parallel`` mode behaves like await in the runner (no special
    branching) — the contract is that the TOOL BODY returns fast and
    LLM round 2 acknowledges the started state. The runner doesn't
    block on a background task; that's the tool's responsibility."""
    session = Session()
    registry = ToolRegistry()

    @registry.tool(response_mode="parallel", completion_text="done")
    def wave() -> dict:
        # Simulate fast-return: real arm_plugin.dispatch_action returns
        # {"started": True} after ~200ms.
        return {"started": True, "success": True}

    llm = _FakeLLM([
        [_tc(0, id="c1", name="wave", arguments="{}"), _finish("tool_calls")],
        [_text("ok, on it"), _finish("stop")],
    ])

    completion_texts: list[str] = []

    async def on_tok(_):
        pass

    async def on_completion_text(t):
        completion_texts.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools={"wave"},
        ctx=_make_ctx(session), on_assistant_token=on_tok,
        on_tool_completion_text=on_completion_text,
    )
    assert final == "ok, on it"
    # parallel mode does NOT auto-emit completion_text (that's template's
    # job); the LLM round 2 covers the acknowledgement.
    assert completion_texts == []
    assert len(llm.calls) == 2


# ── (P2a) name-keyed preamble dedup: same tool twice in one round ─────


@pytest.mark.asyncio
async def test_same_tool_twice_in_one_round_fires_preamble_once():
    """P2a: preamble dedup is keyed by tool NAME (server semantics). When the
    model calls the SAME tool twice in a single round (two tool_call slots,
    same name, each with a preamble_text), the preamble must fire EXACTLY
    once — not once per call slot (the old client ``index`` semantics).
    """
    session = Session()
    registry = ToolRegistry()

    @registry.tool(preamble_text="好的。")
    def wave_hand() -> dict:
        return {"ok": True}

    llm = _FakeLLM([
        # Round 1: TWO tool_calls, both wave_hand (distinct indices).
        [
            _tc(0, id="c1", name="wave_hand", arguments="{}"),
            _tc(1, id="c2", name="wave_hand", arguments="{}"),
            _finish("tool_calls"),
        ],
        # Round 2: final text.
        [_text("done"), _finish("stop")],
    ])

    preambles: list[str] = []

    async def on_tok(_):
        pass

    async def on_preamble(text):
        preambles.append(text)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools=None,
        ctx=_make_ctx(session), on_assistant_token=on_tok,
        on_tool_preamble=on_preamble,
    )
    assert final == "done"
    # Name dedup → preamble fires exactly once despite two same-name calls.
    assert preambles == ["好的。"], preambles


# ── (P2a) all_join template fast-path: mixed round runs round 2 ───────


@pytest.mark.asyncio
async def test_mixed_template_and_await_runs_round2_no_shortcut():
    """P2a: the template fast-path is all_join (server semantics). A round
    with ONE template tool + ONE await tool must NOT short-circuit on the
    template tool (old client ``any_first`` behaviour) — it runs LLM round 2
    so the model can synthesise over both results.
    """
    session = Session()
    registry = ToolRegistry()

    @registry.tool(response_mode="template", completion_text="挥完了")
    def wave() -> dict:
        return {"success": True}

    @registry.tool()  # response_mode defaults to "await"
    def lookup() -> dict:
        return {"success": True, "data": 42}

    llm = _FakeLLM([
        # Round 1: template tool + await tool in one round (distinct slots).
        [
            _tc(0, id="c1", name="wave", arguments="{}"),
            _tc(1, id="c2", name="lookup", arguments="{}"),
            _finish("tool_calls"),
        ],
        # Round 2: the synthesised final text.
        [_text("挥完并查到 42"), _finish("stop")],
    ])

    completion_texts: list[str] = []

    async def on_tok(_):
        pass

    async def on_completion_text(t):
        completion_texts.append(t)

    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    final = await stream_with_tools(
        llm, msgs,
        session=session, registry=registry, allowed_tools=None,
        ctx=_make_ctx(session), on_assistant_token=on_tok,
        on_tool_completion_text=on_completion_text,
    )
    # all_join did NOT fire (mixed round) → LLM round 2 produced the answer.
    assert final == "挥完并查到 42"
    assert len(llm.calls) == 2, "mixed round must run LLM round 2 (no any_first short-circuit)"
    # template completion_text was NOT emitted (fast-path abandoned).
    assert completion_texts == [], completion_texts


def test_tool_dataclass_default_preamble_is_empty():
    """Bare ``@tool()`` registration must default preamble_text to ''
    (not None), so callers can ``if preamble:`` safely."""
    registry = ToolRegistry()

    @registry.tool()
    def t() -> dict:
        return {}

    assert registry._tools["t"].preamble_text == ""
