"""Agent-side defense-in-depth for the SLV /v2v soft-deadlock.

Three independent mitigations (secondary to the server-side root fix):

1. The thinking-watchdog sends ``SLVClient.abort()`` BEFORE reconnecting, so
   the wedged server session gets an in-band teardown hint and releases its
   slot (otherwise the next wake's fresh WS is 4429-rejected → soft-deadlock).
2. A reconnect attempt rejected with WS close code 4429 (session-limit /
   slot-busy) backs off and retries (instead of giving up or hot-looping).
3. A hung remote tool dispatch is bounded by a per-call deadline and returns
   an error result so the server-loop turn proceeds instead of stalling.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovs_agent.app_base import BaseApp
from ovs_agent.slv_client import ServerToolCall, SLVClient
from ovs_agent.state import ConvState
from ovs_agent.event_bus import EventBus


# ── change 1: abort-before-reconnect in the thinking-watchdog ────────────


def _watchdog_app(timeout: float = 0.02) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.events = EventBus()
    app.plugins = []
    app.config = SimpleNamespace(thinking_timeout_s=timeout, pipeline_mode="always_on")
    app._state = ConvState.THINKING
    app._slv_reconnect_count = 0
    app._eos_sent_this_turn = True
    app._asr_watchdog_task = None
    app._thinking_watchdog_task = None
    app._llm_turn_task = None
    app._first_tts_seen = True
    app._sleep_task = None
    # Record the call order across abort()/reconnect() on ONE mock so we can
    # assert abort precedes reconnect.
    app.slv = MagicMock()
    order: list[str] = []

    async def _abort():
        order.append("abort")

    async def _reconnect():
        order.append("reconnect")

    app.slv.abort = AsyncMock(side_effect=_abort)
    app.slv.reconnect = AsyncMock(side_effect=_reconnect)
    app._call_order = order  # type: ignore[attr-defined]
    return app


@pytest.mark.asyncio
async def test_thinking_watchdog_aborts_before_reconnect():
    app = _watchdog_app(timeout=0.02)
    await app._thinking_watchdog()
    assert app._call_order == ["abort", "reconnect"], (
        "watchdog must send SLV abort() BEFORE reconnect() so the wedged "
        f"server session releases its slot; got {app._call_order!r}"
    )
    app.slv.abort.assert_awaited_once()
    app.slv.reconnect.assert_awaited_once()
    assert app._state == ConvState.IDLE


@pytest.mark.asyncio
async def test_thinking_watchdog_guards_missing_slv():
    """No SLV / not-connected case must not raise (pre-connect watchdog)."""
    app = _watchdog_app(timeout=0.02)
    app.slv = None
    # Should complete without AttributeError and still reset state.
    await app._thinking_watchdog()
    assert app._state == ConvState.IDLE


@pytest.mark.asyncio
async def test_thinking_watchdog_noop_when_state_moved_on():
    """If THINKING resolved naturally, neither abort nor reconnect fire."""
    app = _watchdog_app(timeout=0.02)
    app._state = ConvState.SPEAKING
    await app._thinking_watchdog()
    app.slv.abort.assert_not_awaited()
    app.slv.reconnect.assert_not_awaited()


# ── change 3: per-tool-callback dispatch timeout ─────────────────────────


def _dispatch_app() -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.config = SimpleNamespace(
        tool_trigger_guard=False, tool_trigger_guard_log_only=False
    )
    app.session = None
    app.modes = None
    app.events = None
    app._last_user_utterance_text = ""
    app.slv = MagicMock()
    app.slv.send_tool_result = AsyncMock()
    return app


class _SlowRegistry:
    """Registry whose dispatch blocks longer than the call timeout."""

    _tools: dict = {}

    def __init__(self, delay: float):
        self._delay = delay
        self.dispatched = False

    async def dispatch(self, name, arguments, ctx):
        self.dispatched = True
        await asyncio.sleep(self._delay)
        return {"success": True}


class _FastRegistry:
    _tools: dict = {}

    def __init__(self, result):
        self._result = result
        self.dispatched = False

    async def dispatch(self, name, arguments, ctx):
        self.dispatched = True
        return self._result


@pytest.mark.asyncio
async def test_dispatch_timeout_returns_error_result():
    app = _dispatch_app()
    app.tool_registry = _SlowRegistry(delay=5.0)
    # timeout_s well under the handler delay → must time out, not hang.
    evt = ServerToolCall(id="call-1", name="get_frame", arguments={}, timeout_s=0.05)
    await asyncio.wait_for(app._handle_server_tool_call(evt), timeout=2.0)
    app.slv.send_tool_result.assert_awaited_once()
    _args, kwargs = app.slv.send_tool_result.call_args
    assert kwargs.get("ok") is False
    assert "timed out" in (kwargs.get("error") or "").lower()


@pytest.mark.asyncio
async def test_dispatch_timeout_default_when_absent_is_lenient():
    """When timeout_s is absent/invalid, default (30s) applies — a fast tool
    on the success path is byte-identical (ok=True, full result)."""
    app = _dispatch_app()
    result = {"success": True, "value": 42}
    app.tool_registry = _FastRegistry(result)
    evt = ServerToolCall(id="call-2", name="ok_tool", arguments={})
    # Force the "absent" path: a non-positive timeout falls back to default.
    evt.timeout_s = 0.0
    await app._handle_server_tool_call(evt)
    app.slv.send_tool_result.assert_awaited_once_with(
        "call-2", "ok_tool", ok=True, result=result
    )


@pytest.mark.asyncio
async def test_dispatch_success_path_unchanged():
    """Success path with a normal timeout_s: ok=True with the exact result."""
    app = _dispatch_app()
    result = {"started": True}
    app.tool_registry = _FastRegistry(result)
    evt = ServerToolCall(id="call-3", name="grasp_object", arguments={}, timeout_s=15.0)
    await app._handle_server_tool_call(evt)
    app.slv.send_tool_result.assert_awaited_once_with(
        "call-3", "grasp_object", ok=True, result=result
    )


# ── change 2: 4429 backoff-and-retry on reconnect ────────────────────────


class _FakeWS:
    """Minimal WS double for _open_with_retry. ``reject_first`` makes the
    first connection's reader die inside the grace window with close code
    4429 (server slot busy); the second succeeds (reader stays alive)."""

    def __init__(self, *, reject: bool):
        self._reject = reject
        self.closed = False

    async def send(self, payload):
        return None

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._reject:
            import websockets

            # Surface a 4429 close so _reader_loop records _last_close_code.
            raise websockets.ConnectionClosed(
                SimpleNamespace(code=4429, reason="slot busy"), None
            )
        # Healthy: block forever (reader stays alive past the grace window).
        # Use a never-set Event, not asyncio.sleep — sleep is monkeypatched in
        # the test, which would make a "healthy" reader exit immediately.
        await asyncio.Event().wait()
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_reconnect_4429_backs_off_then_succeeds(monkeypatch):
    client = SLVClient("ws://test", {"asr_language": "zh"})
    # Tiny backoffs so the test is fast; still proves the retry path runs.
    monkeypatch.setattr(SLVClient, "_RECONNECT_BACKOFFS", (0.01, 0.01, 0.01))

    connect_calls = {"n": 0}
    sleeps: list[float] = []

    async def _fake_ws_connect(url, **kw):
        connect_calls["n"] += 1
        # First attempt: rejected with 4429. Subsequent: healthy.
        return _FakeWS(reject=(connect_calls["n"] == 1))

    real_sleep = asyncio.sleep

    async def _spy_sleep(d):
        sleeps.append(d)
        await real_sleep(0)  # don't actually wait

    monkeypatch.setattr("ovs_agent.slv_client.ws_connect", _fake_ws_connect)
    monkeypatch.setattr("ovs_agent.slv_client.asyncio.sleep", _spy_sleep)

    async with client._send_lock:
        await client._open_with_retry()

    # Two WS opens: the rejected one + the healthy one.
    assert connect_calls["n"] == 2, "4429 rejection must trigger a retry open"
    # A backoff sleep happened between the rejected attempt and the retry.
    assert any(s == 0.01 for s in sleeps), (
        f"expected a backoff sleep from _RECONNECT_BACKOFFS, got {sleeps!r}"
    )
    # Landed on a healthy WS (reader alive, close code cleared on success).
    assert client._ws is not None
    assert client._last_close_code is None
    await client.close()


@pytest.mark.asyncio
async def test_reconnect_4429_all_attempts_exhausted_raises(monkeypatch):
    from ovs_agent.slv_client import SLVReconnectError

    client = SLVClient("ws://test", {"asr_language": "zh"})
    monkeypatch.setattr(SLVClient, "_RECONNECT_BACKOFFS", (0.01, 0.01))

    async def _always_reject(url, **kw):
        return _FakeWS(reject=True)

    real_sleep = asyncio.sleep

    async def _spy_sleep(d):
        await real_sleep(0)

    monkeypatch.setattr("ovs_agent.slv_client.ws_connect", _always_reject)
    monkeypatch.setattr("ovs_agent.slv_client.asyncio.sleep", _spy_sleep)

    with pytest.raises(SLVReconnectError) as exc:
        async with client._send_lock:
            await client._open_with_retry()
    # The exhausted-attempts error names the 4429 cause.
    assert "4429" in str(exc.value)
    await client.close()
