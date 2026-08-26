"""Idle keepalive (P1): a live-but-silent client must not be reaped.

The server's v2v dispatcher treats "no frame for OVS_V2V_IDLE_TIMEOUT_S"
(default 90s) as a half-open client and releases the session slot. In
wake-word mode a quiet room legitimately sends nothing, so that watchdog
fired every 90s and an utterance landing in the reconnect window lost its
first syllables. SLVClient now sends a no-op ping while the send path is
idle.
"""
from __future__ import annotations

import asyncio

import pytest

from ovs_agent.protocol import CLIENT_PING
from ovs_agent.slv_client import SLVClient


class _FakeWS:
    """Records outgoing frames; never closes."""

    def __init__(self) -> None:
        self.sent: list = []

    async def send(self, frame) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        pass


def _client(monkeypatch, interval_s: str) -> tuple[SLVClient, _FakeWS]:
    monkeypatch.setenv("OVS_SLV_KEEPALIVE_S", interval_s)
    c = SLVClient("ws://test/v2v/stream", {"asr": "auto"})
    ws = _FakeWS()
    c._ws = ws
    return c, ws


def _pings(ws: _FakeWS) -> int:
    return sum(1 for f in ws.sent if isinstance(f, str) and CLIENT_PING in f)


@pytest.mark.asyncio
async def test_keepalive_pings_while_idle(monkeypatch):
    c, ws = _client(monkeypatch, "0.05")
    c._touch_send()
    c._start_keepalive()
    try:
        await asyncio.sleep(0.3)
    finally:
        c._stop_keepalive()
    assert _pings(ws) >= 2, f"expected repeated pings, got {ws.sent}"


@pytest.mark.asyncio
async def test_real_traffic_suppresses_keepalive(monkeypatch):
    """An active turn must add no ping traffic — real frames reset the timer."""
    c, ws = _client(monkeypatch, "0.2")
    c._touch_send()
    c._start_keepalive()
    try:
        for _ in range(12):
            await c.send_audio(b"\x00\x00" * 160)
            await asyncio.sleep(0.03)
    finally:
        c._stop_keepalive()
    assert _pings(ws) == 0, f"keepalive fired during a live turn: {ws.sent}"
    assert len(ws.sent) == 12


@pytest.mark.asyncio
async def test_keepalive_never_revives_a_dead_ws(monkeypatch):
    """Reconnect ownership stays with reconnect()/dispatch.

    A keepalive that called connect() would open a session behind their
    back and race the server's single-session limiter.
    """
    c, _ws = _client(monkeypatch, "0.05")
    c._ws = None
    connects = 0

    async def _boom() -> None:
        nonlocal connects
        connects += 1

    c.connect = _boom  # type: ignore[method-assign]
    c._touch_send()
    c._start_keepalive()
    try:
        await asyncio.sleep(0.25)
    finally:
        c._stop_keepalive()
    assert connects == 0


@pytest.mark.asyncio
async def test_keepalive_silent_while_reconnecting(monkeypatch):
    c, ws = _client(monkeypatch, "0.05")
    c._reconnecting = True
    c._touch_send()
    c._start_keepalive()
    try:
        await asyncio.sleep(0.25)
    finally:
        c._stop_keepalive()
    assert _pings(ws) == 0


@pytest.mark.asyncio
async def test_keepalive_disabled_by_env(monkeypatch):
    c, ws = _client(monkeypatch, "0")
    c._touch_send()
    c._start_keepalive()
    try:
        await asyncio.sleep(0.2)
    finally:
        c._stop_keepalive()
    assert c._keepalive_task is None
    assert _pings(ws) == 0


@pytest.mark.asyncio
async def test_close_stops_keepalive(monkeypatch):
    c, _ws = _client(monkeypatch, "0.05")
    c._touch_send()
    c._start_keepalive()
    task = c._keepalive_task
    assert task is not None
    await c.close()
    assert c._keepalive_task is None
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()


def test_keepalive_clears_the_tightest_deployed_watchdog():
    """Margin must be sized against the tightest DEPLOYED idle timeout.

    The server defaults to 90s, but deploy/docker-compose.jetson-rebot.yml
    pins OVS_V2V_IDLE_TIMEOUT_S=45 on the arm stack. Sizing the cadence
    against 90s leaves only 1.5x margin there and one late ping reaps a
    live session.
    """
    c = SLVClient("ws://test/v2v/stream", {})
    assert c._KEEPALIVE_DEFAULT_S * 3 <= c._TIGHTEST_DEPLOYED_IDLE_TIMEOUT_S


def test_tightest_deployed_idle_timeout_matches_the_repo():
    """Pin the constant against the actual compose files.

    If someone lowers OVS_V2V_IDLE_TIMEOUT_S further, this fails rather
    than silently erasing the keepalive margin.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    found = []
    for path in list(root.glob("deploy/*.yml")) + list(root.glob("deploy/*.yaml")):
        for m in re.finditer(
            r"OVS_V2V_IDLE_TIMEOUT_S\s*[=:]\s*([0-9.]+)", path.read_text()
        ):
            found.append(float(m.group(1)))
    assert found, "no OVS_V2V_IDLE_TIMEOUT_S found under deploy/"
    c = SLVClient("ws://test/v2v/stream", {})
    assert min(found) >= c._TIGHTEST_DEPLOYED_IDLE_TIMEOUT_S, (
        f"a deploy config now sets OVS_V2V_IDLE_TIMEOUT_S={min(found)}, below "
        f"the {c._TIGHTEST_DEPLOYED_IDLE_TIMEOUT_S}s the keepalive cadence "
        f"was sized against"
    )


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "Infinity"])
def test_non_finite_interval_falls_back_to_default(monkeypatch, raw):
    """float() accepts nan/inf; neither is a usable sleep interval.

    inf never pings (session reaped as if there were no keepalive at all);
    nan makes asyncio.sleep() raise and silently kills the task.
    """
    monkeypatch.setenv("OVS_SLV_KEEPALIVE_S", raw)
    c = SLVClient("ws://test/v2v/stream", {})
    assert c._keepalive_interval_s == c._KEEPALIVE_DEFAULT_S


@pytest.mark.parametrize("raw", ["", "   ", "abc"])
def test_malformed_interval_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv("OVS_SLV_KEEPALIVE_S", raw)
    c = SLVClient("ws://test/v2v/stream", {})
    assert c._keepalive_interval_s == c._KEEPALIVE_DEFAULT_S


@pytest.mark.asyncio
async def test_non_finite_interval_does_not_kill_the_task(monkeypatch):
    """Regression: asyncio.sleep(nan) raising would leave no keepalive."""
    c, ws = _client(monkeypatch, "nan")
    c._touch_send()
    # Force the loop to be due immediately regardless of the fallback cadence.
    c._last_send_ts = 0.0
    c._start_keepalive()
    try:
        await asyncio.sleep(0.15)
        assert c._keepalive_task is not None
        assert not c._keepalive_task.done(), "keepalive task died"
    finally:
        c._stop_keepalive()
    assert _pings(ws) >= 1
