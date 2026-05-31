"""#38 — voice-arm agent connect/advertise/reconnect state machine.

Covers the boot-time connect retry budget and tool re-advertise after
reconnect, so server-loop parity survives both the SLV session-limiter
slot-release window at boot AND any WS churn at runtime.

  1. boot race regression: connect() fails for >1.75s (longer than the
     runtime reconnect budget) but <75s, then succeeds → boot continues
     and advertise fires.
  2. reconnect re-advertise: every successful reconnect re-advertises
     exactly once (idempotent upsert on SLV).
  3. server-loop OFF no-op: the advertise entry point touches no WS and
     raises nothing when the flag is off.
  4. duplicate advertise upsert: advertising twice in server-loop is a
     no-error upsert (latest schema wins on the client side too).
  5. client-loop mode does not regress: a full boot+dispatch with
     server_loop=False issues zero advertise-related WS sends.
"""
from __future__ import annotations

import asyncio

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.slv_client import ASRFinal, SLVReconnectError
from openvoicestream_agent.state import ConvState
from openvoicestream_agent.tools import ToolRegistry


# ── fakes ────────────────────────────────────────────────────────────


class FakeSLV:
    """SLV double with scriptable connect failures + reconnect counting."""

    def __init__(self, connect_fail_times: int = 0) -> None:
        # Number of connect() calls that should raise before succeeding.
        self._connect_fail_remaining = connect_fail_times
        self.connect_calls = 0
        self.reconnect_calls = 0
        self.advertised: list[dict] = []

    async def connect(self) -> None:
        self.connect_calls += 1
        if self._connect_fail_remaining > 0:
            self._connect_fail_remaining -= 1
            raise SLVReconnectError("session limiter slot busy (4429)")

    async def reconnect(self) -> None:
        self.reconnect_calls += 1

    async def advertise_tools(self, tools, *, system_prompt=None, llm_params=None):
        self.advertised.append(
            {"tools": list(tools), "system_prompt": system_prompt, "llm_params": llm_params}
        )

    # Liveness shims used by other paths (not exercised here).
    def is_reconnecting(self) -> bool:
        return False


async def _noop_broadcast(*args, **kwargs):
    return None


def _make_app(server_loop: bool, slv: FakeSLV, registry: ToolRegistry) -> BaseApp:
    """Build a BaseApp without running __init__ (avoids audio/WS setup)."""
    cfg = Config(
        system_prompt="SYS",
        server_loop=server_loop,
        tools_max_iterations=4,
        llm_model="qwen-test",
    )
    app = BaseApp.__new__(BaseApp)
    app.config = cfg
    app.tool_registry = registry
    app.session = Session()
    app.modes = None
    app.slv = slv
    app._slv_reconnect_count = 0
    app._broadcast = _noop_broadcast  # type: ignore[assignment]
    return app


def _two_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(name="wave", description="wave the arm")
    def _wave(side: str = "left") -> dict:
        return {"ran": "wave", "side": side}

    @reg.tool(name="grip", description="grip")
    def _grip() -> dict:
        return {"ran": "grip"}

    return reg


# ── 1. boot race regression ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_boot_connect_retries_past_runtime_budget(monkeypatch):
    """connect() fails long enough to exhaust the SHORT runtime reconnect
    budget (~1.75s) but well within the boot deadline → boot retries and
    eventually succeeds instead of aborting before advertise.

    We mock connect() to raise SLVReconnectError 5 times — the runtime
    ``_RECONNECT_BACKOFFS`` would have given up after 3 — then succeed.
    asyncio.sleep is patched to a no-op so the test is instant while still
    exercising the real retry loop and its monotonic-clock deadline math.
    """
    slv = FakeSLV(connect_fail_times=5)
    app = _make_app(server_loop=True, slv=slv, registry=_two_tool_registry())

    slept: list[float] = []

    async def _fast_sleep(d):
        slept.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    # Sanity: this many failures exceeds the runtime reconnect budget,
    # which would have aborted boot under the old code.
    from openvoicestream_agent.slv_client import SLVClient
    assert len(SLVClient._RECONNECT_BACKOFFS) < 5

    await app._connect_with_boot_retry()

    # 5 failures + 1 success = 6 connect attempts; boot did NOT abort.
    assert slv.connect_calls == 6
    # Capped exponential backoff: 0.5, 1.0, 2.0, 5.0, then 5.0.
    assert slept == [0.5, 1.0, 2.0, 5.0, 5.0]

    # Boot continues → advertise fires (the regression we are guarding).
    await app._advertise_tools_if_server_loop()
    assert len(slv.advertised) == 1
    assert {t["function"]["name"] for t in slv.advertised[0]["tools"]} == {"wave", "grip"}


@pytest.mark.asyncio
async def test_boot_connect_gives_up_after_deadline(monkeypatch):
    """If connect() keeps failing past the boot deadline, the final error
    escapes (so the operator sees a hard failure rather than a silent hang).
    """
    slv = FakeSLV(connect_fail_times=10_000)  # never succeeds
    app = _make_app(server_loop=True, slv=slv, registry=_two_tool_registry())
    app._BOOT_CONNECT_DEADLINE_S = 3.0  # shrink the budget for the test

    # Advance a fake monotonic clock by the backoff each sleep so the
    # deadline is reached deterministically without real waiting.
    clock = {"t": 0.0}
    monkeypatch.setattr(
        "openvoicestream_agent.app_base.time.monotonic", lambda: clock["t"]
    )

    async def _advance_sleep(d):
        clock["t"] += d

    monkeypatch.setattr(asyncio, "sleep", _advance_sleep)

    with pytest.raises(SLVReconnectError):
        await app._connect_with_boot_retry()


# ── 2. reconnect re-advertise ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconnect_readvertises_exactly_once():
    """A single successful reconnect re-advertises tools exactly once."""
    slv = FakeSLV()
    app = _make_app(server_loop=True, slv=slv, registry=_two_tool_registry())

    await app._readvertise_after_reconnect()

    assert len(slv.advertised) == 1
    assert slv.reconnect_calls == 0  # helper does not reconnect itself


@pytest.mark.asyncio
async def test_sleeping_session_complete_reconnect_readvertises():
    """The SLEEPING + session_complete dispatch path (reconnect site #4)
    re-advertises tools exactly once per reconnect, end to end.
    """
    slv = FakeSLV()
    app = _make_app(server_loop=True, slv=slv, registry=_two_tool_registry())
    app._state = ConvState.SLEEPING
    app._first_tts_seen = True

    await app._dispatch_one(ASRFinal(text="late final", session_complete=True))

    # WS was reopened and tools re-advertised exactly once.
    assert slv.reconnect_calls == 1
    assert len(slv.advertised) == 1

    # A second reconnect re-advertises again (one per reconnect).
    await app._dispatch_one(ASRFinal(text="another", session_complete=True))
    assert slv.reconnect_calls == 2
    assert len(slv.advertised) == 2


@pytest.mark.asyncio
async def test_readvertise_failure_does_not_propagate():
    """An advertise failure during reconnect recovery must be swallowed —
    a transient send error can't be allowed to abort the reconnect path.
    """
    slv = FakeSLV()

    async def _boom(*a, **k):
        raise RuntimeError("WS send failed")

    slv.advertise_tools = _boom  # type: ignore[assignment]
    app = _make_app(server_loop=True, slv=slv, registry=_two_tool_registry())

    # Must not raise.
    await app._readvertise_after_reconnect()


# ── 3. server-loop OFF no-op ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_off_mode_readvertise_is_noop():
    slv = FakeSLV()
    app = _make_app(server_loop=False, slv=slv, registry=_two_tool_registry())

    # Calling twice must not touch the WS and must not raise.
    await app._readvertise_after_reconnect()
    await app._advertise_tools_if_server_loop()
    await app._readvertise_after_reconnect()

    assert slv.advertised == []


@pytest.mark.asyncio
async def test_off_mode_sleeping_reconnect_does_not_advertise():
    """In client-loop mode the SLEEPING session_complete reconnect still
    reconnects but issues ZERO advertise sends (no behaviour change).
    """
    slv = FakeSLV()
    app = _make_app(server_loop=False, slv=slv, registry=_two_tool_registry())
    app._state = ConvState.SLEEPING
    app._first_tts_seen = True

    await app._dispatch_one(ASRFinal(text="late final", session_complete=True))

    assert slv.reconnect_calls == 1
    assert slv.advertised == []


# ── 4. duplicate advertise upsert ─────────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_advertise_upsert_no_error():
    """Advertising twice in server-loop mode is an idempotent upsert: no
    error client-side, and the latest schema is the one sent last.
    """
    slv = FakeSLV()
    reg = _two_tool_registry()
    app = _make_app(server_loop=True, slv=slv, registry=reg)

    await app._advertise_tools_if_server_loop()

    # Mutate the registry, then re-advertise: newest schema wins.
    @reg.tool(name="point", description="point")
    def _point() -> dict:
        return {"ran": "point"}

    await app._readvertise_after_reconnect()

    assert len(slv.advertised) == 2
    first_names = {t["function"]["name"] for t in slv.advertised[0]["tools"]}
    last_names = {t["function"]["name"] for t in slv.advertised[1]["tools"]}
    assert first_names == {"wave", "grip"}
    assert last_names == {"wave", "grip", "point"}  # latest schema wins


# ── 5. client-loop mode does not regress ──────────────────────────────


@pytest.mark.asyncio
async def test_client_loop_boot_path_issues_no_advertise(monkeypatch):
    """A full client-loop boot (connect retry wrapper + advertise entry
    point + mic gate release) issues zero advertise sends and never blocks.
    """
    slv = FakeSLV()
    app = _make_app(server_loop=False, slv=slv, registry=_two_tool_registry())

    # Boot connect succeeds first try.
    await app._connect_with_boot_retry()
    assert slv.connect_calls == 1

    # Simulate the run() boot advertise + gate release without the rest of
    # the heavy run() machinery.
    app._advertise_ready = asyncio.Event()
    try:
        await app._advertise_tools_if_server_loop()
    finally:
        app._advertise_ready.set()

    # Client-loop: zero advertise frames, gate released (mic can forward).
    assert slv.advertised == []
    assert app._advertise_ready.is_set()
