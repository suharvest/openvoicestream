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


def test_keepalive_interval_leaves_margin_under_server_watchdog():
    """30s vs the server's 90s default — a dropped ping must not reap us."""
    c = SLVClient("ws://test/v2v/stream", {})
    assert c._KEEPALIVE_DEFAULT_S * 3 <= 90.0
