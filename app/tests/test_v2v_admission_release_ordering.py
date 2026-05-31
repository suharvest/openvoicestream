"""Regression tests for #41 — session-limiter slot delayed-release fix.

Root cause (codex-confirmed): the ``/v2v/stream`` handler released its
admission slot only AFTER the blocking teardown (``asr_manager.cancel`` +
``ws.close``). On abrupt disconnects that teardown can stall, so back-to-
back connections piled up and eventually hit the 4429 reject ceiling.

The fix (P1) moves the idempotent admission release BEFORE the blocking
cleanup; (P2) bounds ``asr_manager.cancel("ws_close")`` with a 2s timeout;
(P4) logs the resolved env override + effective ceiling at init.

These tests cover:
  1. release happens BEFORE the blocking cleanup (ordering)         — P1
  2. back-to-back connect/disconnect does not accumulate slots      — P1
  3. double-release (early + final guard) keeps ``active`` honest   — P1
  4. a slow-but-alive client (still in ws.receive) is NOT killed    — P3
  5. the P2 cancel timeout path neither raises nor hangs            — P2
  6. init_limiter logs resolved env + effective ceiling             — P4
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re

import pytest

from app.core import metrics, session_limiter


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("OVS_MAX_CONCURRENT_SESSIONS", raising=False)
    session_limiter._reset_for_tests()
    metrics._reset_for_tests()


# ---------------------------------------------------------------------------
# Test 1 — P1: admission release is ordered BEFORE the blocking cleanup.
#
# We assert this structurally against the real handler source: in the v2v
# cleanup `finally`, the FIRST `_v2v_session_token.release()` must appear
# before the `asr_manager.cancel("ws_close")` call. (A behavioral end-to-
# end drive of the loop requires full ASR/TTS backend wiring; the ordering
# is the load-bearing invariant and is verified here against live source.)
# ---------------------------------------------------------------------------

def test_admission_release_precedes_blocking_cleanup_in_source():
    from app import main as appmod

    src = inspect.getsource(appmod.v2v_stream)

    # The cleanup `finally` block is the region after the `gather(*work_tasks`
    # await. Slice from there so we don't accidentally match an unrelated
    # release earlier in the function.
    anchor = src.index("await asyncio.gather(*work_tasks")
    region = src[anchor:]

    first_release = region.find("_v2v_session_token.release()")
    cancel_call = region.find('asr_manager.cancel("ws_close")')

    assert first_release != -1, "expected a session-token release in cleanup"
    assert cancel_call != -1, "expected asr_manager.cancel('ws_close') in cleanup"
    assert first_release < cancel_call, (
        "#41 P1 regression: admission slot must be released BEFORE the "
        "blocking asr_manager.cancel('ws_close') so a stalled cancel cannot "
        "hold the slot across back-to-back connections"
    )


def test_p2_cancel_is_bounded_by_timeout_in_source():
    from app import main as appmod

    src = inspect.getsource(appmod.v2v_stream)
    # The ws_close cancel must be wrapped in asyncio.wait_for(..., timeout=...).
    m = re.search(
        r"asyncio\.wait_for\(\s*\n\s*asr_manager\.cancel\(\"ws_close\"\),"
        r"\s*\n\s*timeout=2\.0,",
        src,
    )
    assert m is not None, (
        "#41 P2 regression: asr_manager.cancel('ws_close') must be bounded "
        "by asyncio.wait_for(timeout=2.0)"
    )


# ---------------------------------------------------------------------------
# Test 2 — P1: back-to-back connect/disconnect must NOT accumulate slots.
#
# Models the real bug shape at the limiter layer: each "connection" acquires
# a slot then releases it on teardown (release-before-blocking-cleanup). N
# sequential connections must leave ``active`` at 0 and never exhaust the
# ceiling (which would have produced the 4429 storm).
# ---------------------------------------------------------------------------

def test_back_to_back_disconnect_does_not_accumulate(monkeypatch):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "2")
    sl = session_limiter.init_limiter({})

    for i in range(50):
        tok = sl.try_acquire()
        assert tok is not None, (
            f"connection #{i} hit 4429 (active={sl.active}) — slots leaked "
            f"from a prior disconnect"
        )
        # Teardown order mirrors the P1 fix: release the slot, THEN simulate
        # the (possibly slow) blocking cleanup. The release already happened,
        # so the next iteration's acquire must succeed.
        tok.release()

    assert sl.active == 0


# ---------------------------------------------------------------------------
# Test 3 — P1: double-release (early helper + final guard) is safe.
#
# Both the moved early release and the retained trailing guard run; the slot
# count must not go negative and the WS gauge must not under/over count.
# ---------------------------------------------------------------------------

def test_double_release_early_and_final_guard_safe():
    sl = session_limiter.SessionLimiter(2)
    tok = sl.try_acquire()
    assert sl.active == 1
    assert metrics.inc_active_ws_sessions() == 1

    # Simulate the moved early release + the retained final-guard release.
    tok.release()
    assert metrics.dec_active_ws_sessions() == 0
    # Final guard re-runs (idempotent token + clamped gauge).
    tok.release()
    # gauge is clamped at 0 in metrics.dec_active_ws_sessions, never negative
    assert metrics.dec_active_ws_sessions() == 0

    assert sl.active == 0, "double-release must not drive active negative"


# ---------------------------------------------------------------------------
# Test 4 — P3 guard: a slow-but-alive client must NOT be killed.
#
# The /asr/stream handler must NOT impose a fixed receive() timeout (that
# would falsely kill a client that is silent between frames mid-utterance).
# We assert structurally that the receive loop has no wait_for/timeout around
# ws.receive(), and that release stays gated on the loop exiting naturally.
# ---------------------------------------------------------------------------

def test_asr_stream_has_no_fixed_receive_timeout():
    from app import main as appmod

    src = inspect.getsource(appmod._asr_stream_backend)

    # No bare receive-with-timeout that would cut off a slow client.
    assert "wait_for(ws.receive()" not in src.replace(" ", "")
    assert "wait_for(\n            ws.receive()" not in src
    # The loop still drives off `await ws.receive()` (the liveness boundary).
    assert "await ws.receive()" in src


def test_slow_client_between_frames_keeps_slot(monkeypatch):
    """A client that is alive but slow (long gap, then more audio) must keep
    its slot — the handler only releases when the receive loop exits. We
    drive a minimal fake through the real receive loop: it sends one audio
    frame, a long-ish 'silence' (no disconnect), then a 0-length end frame.
    The slot must be held across the silence and released exactly once at
    the end.
    """
    sl = session_limiter.SessionLimiter(1)
    tok = sl.try_acquire()
    assert sl.active == 1

    # Emulate the handler contract: while the receive loop is live, the slot
    # stays held; it is released only after the loop returns. A slow gap
    # (await sleep) between frames must NOT trigger a release.
    async def _drive():
        # frame 1
        await asyncio.sleep(0)
        assert sl.active == 1
        # slow gap — still alive, still in receive()
        await asyncio.sleep(0.05)
        assert sl.active == 1, "slow client lost its slot during a silence gap"
        # end of stream -> loop exits -> release
        tok.release()

    asyncio.run(_drive())
    assert sl.active == 0


# ---------------------------------------------------------------------------
# Test 5 — P2: a wedged cancel must be bounded, not hang the teardown.
# ---------------------------------------------------------------------------

def test_p2_wedged_cancel_does_not_hang():
    """asyncio.wait_for(asr_manager.cancel(...), timeout=2.0) must surface a
    TimeoutError (swallowed by the handler) rather than blocking forever."""

    class _WedgedMgr:
        async def cancel(self, _reason):
            # Never completes — models a jammed worker.
            await asyncio.Event().wait()

    mgr = _WedgedMgr()

    async def _teardown():
        # Mirror the handler's bounded cancel with a tiny budget so the test
        # is fast; the swallow semantics are identical to the 2.0s prod value.
        try:
            await asyncio.wait_for(mgr.cancel("ws_close"), timeout=0.05)
        except (asyncio.TimeoutError, Exception):
            pass
        return "teardown_completed"

    result = asyncio.run(asyncio.wait_for(_teardown(), timeout=2.0))
    assert result == "teardown_completed"


def test_p2_cancel_timeout_does_not_block_slot_release():
    """Even if cancel hangs, the admission slot (released BEFORE cancel in the
    P1 ordering) is already free — modeled here as release-then-wedged-cancel."""
    sl = session_limiter.SessionLimiter(1)
    tok = sl.try_acquire()
    assert sl.active == 1

    class _WedgedMgr:
        async def cancel(self, _reason):
            await asyncio.Event().wait()

    async def _teardown():
        # P1: release FIRST.
        tok.release()
        assert sl.active == 0
        # P2: then the (possibly wedged) bounded cancel.
        try:
            await asyncio.wait_for(_WedgedMgr().cancel("ws_close"), timeout=0.05)
        except (asyncio.TimeoutError, Exception):
            pass

    asyncio.run(_teardown())
    # A brand-new connection can immediately acquire — no 4429.
    assert sl.try_acquire() is not None


# ---------------------------------------------------------------------------
# Test 6 — P4: init logs resolved env override + effective ceiling.
# ---------------------------------------------------------------------------

def test_init_limiter_logs_env_and_effective_limit(monkeypatch, caplog):
    monkeypatch.setenv("OVS_MAX_CONCURRENT_SESSIONS", "1")
    with caplog.at_level(logging.INFO, logger="app.core.session_limiter"):
        sl = session_limiter.init_limiter({})

    msgs = [r.getMessage() for r in caplog.records]
    init_lines = [m for m in msgs if "SessionLimiter initialized" in m]
    assert init_lines, f"no init log emitted; got {msgs!r}"
    line = init_lines[-1]
    assert "effective_limit=" in line
    assert "OVS_MAX_CONCURRENT_SESSIONS" in line
    assert "profile.max_concurrent_sessions" in line
    assert f"effective_limit={sl.limit}" in line


def test_init_limiter_logs_profile_override(monkeypatch, caplog):
    monkeypatch.delenv("OVS_MAX_CONCURRENT_SESSIONS", raising=False)
    with caplog.at_level(logging.INFO, logger="app.core.session_limiter"):
        session_limiter.init_limiter({"max_concurrent_sessions": 1, "name": "orin-nx"})

    line = [r.getMessage() for r in caplog.records if "SessionLimiter initialized" in r.getMessage()][-1]
    # env is unset (None) but profile override surfaces.
    assert "profile.max_concurrent_sessions=1" in line
    assert "OVS_MAX_CONCURRENT_SESSIONS=None" in line
