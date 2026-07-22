"""Regression: ASR wedge self-heal when the session manager self-drops to
IDLE while server.main still holds ``asr_active`` at the dead generation.

Root cause (2026-07-02, Orin-NX exhibition)
===========================================
An ASR turn hit the 45s wall-clock cap → ``asr_manager.cancel("turn_timeout")``
→ cancel executor timed out → worker restart → manager lands in **IDLE** at
generation N. But ``server.main``'s per-connection ``asr_active`` /
``asr_active_gen`` still point at generation N. For a backend that treats a
fresh VAD speech-start as *barge-in-only* while ``asr_active`` is True
(``prefer_backend_endpoint_vad``), ``on_speech_start`` is never called, so the
generation never advances and **every** subsequent ``finalize`` is rejected
(state != ACTIVE) → ``suppressing discarded asr_final from gen=N current_gen=N``
on a loop. ASR stays dead until a full container restart.

Fix (this commit, server/main.py)
----------------------------------
Both the barge-in-only speech-start gate and the suppressed-finalize path now
consult the manager's ACTUAL state via ``_asr_manager_idle()``:
  * speech-start: if the manager is IDLE, fall through to ``on_speech_start``
    (re-open a fresh generation) instead of barge-in-only.
  * suppressed finalize: if the manager is IDLE, clear ``asr_active`` so the
    next speech-start re-opens.

IDLE-only (NOT merely "not ACTIVE"): FINALIZING / CANCELLING / ERROR_REBUILD are
transient states a HEALTHY turn passes through — keying off those would misfire
the fix on a live utterance. This test drives a real ``ASRSessionManager`` into
the desync state and asserts the pre-fix logic wedges, the post-fix logic
self-heals, and neither a healthy ACTIVE turn nor a transient FINALIZING state
triggers the fix.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server.core.asr_backend import ASRBackend, ASRCapability, TranscriptionResult
from server.core.asr_session_manager import ASRSessionManager, SessionState


def _asynctest(fn):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = fn.__name__
    return wrapper


# Mirror of server.main._asr_manager_idle (a nested closure, not importable).
def _asr_manager_idle(mgr) -> bool:
    st = getattr(mgr, "state", None)
    return str(getattr(st, "value", st)) == "idle"


class _Stream:
    def __init__(self, text: str = "hi"):
        self._text = text
        self.cancelled = False

    def accept_waveform(self, sr: int, samples) -> None:  # noqa: ANN001
        pass

    def get_partial(self):
        return "", False

    def finalize(self):
        return self._text, None

    def cancel(self) -> None:
        self.cancelled = True

    def cancel_and_finalize(self):
        self.cancelled = True
        return ""


class _Backend(ASRBackend):
    """Minimal streaming backend; hands out a fresh stream each open."""

    def __init__(self):
        self.restart_calls = 0
        self.opened = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return "fake-selfheal"

    @property
    def capabilities(self):  # type: ignore[override]
        return {ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return 16000

    def is_ready(self) -> bool:  # type: ignore[override]
        return True

    def preload(self) -> None:  # type: ignore[override]
        return None

    def transcribe(self, audio_bytes: bytes, language: str = "auto"):  # type: ignore[override]
        return TranscriptionResult(text="", duration=0.0, inference_time=0.0,
                                   rtf=0.0, n_tokens=0, per_token_ms=0.0, backend=self.name)

    def create_stream(self, language: str = "auto"):  # type: ignore[override]
        self.opened += 1
        return _Stream()

    def restart_worker(self) -> None:
        self.restart_calls += 1


# ── mirrored decision blocks from asr_out_task / dispatcher ────────────
def _speech_start_is_bargein_only(state: dict, mgr, *, post_fix: bool) -> bool:
    """True => treat VAD speech-start as barge-in only (do NOT re-open).
    False => fall through to on_speech_start (re-open a fresh generation)."""
    prefers_backend_vad = True  # qwen3-style backend-owned endpointing
    if post_fix:
        return bool(state["asr_active"] and prefers_backend_vad and not _asr_manager_idle(mgr))
    return bool(state["asr_active"] and prefers_backend_vad)  # pre-fix


def _reconcile_after_suppressed_final(state: dict, mgr) -> None:
    """Mirror of the post-fix suppressed-finalize reconcile block."""
    if _asr_manager_idle(mgr):
        state["asr_active"] = False
        state["asr_audio_samples_accepted"] = 0
        state["asr_turn_started_at"] = None
        state["endpoint_pending"] = None
        state["endpoint_pending_gen"] = None


async def _drive_to_desync():
    """Return (mgr, state, backend) with manager IDLE@gen1 but server active@1."""
    backend = _Backend()
    mgr = ASRSessionManager(backend=backend, language="auto", sample_rate=16000)
    gen = await mgr.on_speech_start()               # gen=1, ACTIVE
    state = {
        "asr_active": True, "asr_active_gen": gen,
        "asr_audio_samples_accepted": 100, "asr_turn_started_at": 123.0,
        "endpoint_pending": "client_eos", "endpoint_pending_gen": gen,
    }
    await mgr.cancel("turn_timeout")                # manager → IDLE, gen stays 1
    assert _asr_manager_idle(mgr)                   # desync: server still active@1
    assert state["asr_active"] is True
    return mgr, state, backend


@_asynctest
async def test_prefix_logic_wedges_on_desync():
    mgr, state, _ = await _drive_to_desync()
    # Pre-fix: speech-start is barge-in-only → on_speech_start never called →
    # generation frozen → permanent wedge.
    assert _speech_start_is_bargein_only(state, mgr, post_fix=False) is True


@_asynctest
async def test_postfix_speech_start_reopens_on_desync():
    mgr, state, _ = await _drive_to_desync()
    # Post-fix: manager IDLE → NOT barge-in-only → re-open.
    assert _speech_start_is_bargein_only(state, mgr, post_fix=True) is False
    # And re-opening actually recovers a fresh ACTIVE generation.
    new_gen = await mgr.on_speech_start()
    assert new_gen == 2
    assert _asr_manager_idle(mgr) is False


@_asynctest
async def test_postfix_suppressed_final_reconciles():
    mgr, state, _ = await _drive_to_desync()
    _reconcile_after_suppressed_final(state, mgr)
    # asr_active cleared so the next speech-start opens a fresh turn.
    assert state["asr_active"] is False
    assert state["asr_turn_started_at"] is None
    assert state["endpoint_pending"] is None


@_asynctest
async def test_healthy_active_turn_is_untouched():
    """Control: a genuinely ACTIVE turn triggers neither fix."""
    backend = _Backend()
    mgr = ASRSessionManager(backend=backend, language="auto", sample_rate=16000)
    gen = await mgr.on_speech_start()
    state = {
        "asr_active": True, "asr_active_gen": gen,
        "asr_audio_samples_accepted": 100, "asr_turn_started_at": 123.0,
        "endpoint_pending": None, "endpoint_pending_gen": None,
    }
    assert _asr_manager_idle(mgr) is False
    assert _speech_start_is_bargein_only(state, mgr, post_fix=True) is True  # unchanged
    _reconcile_after_suppressed_final(state, mgr)
    assert state["asr_active"] is True   # NOT cleared while active
    assert state["asr_turn_started_at"] == 123.0


@_asynctest
async def test_transient_nonidle_states_do_not_misfire():
    """A healthy turn passing through FINALIZING / CANCELLING / ERROR_REBUILD
    must NOT be treated as a desync (IDLE-only trigger, backend-agnostic)."""
    backend = _Backend()
    mgr = ASRSessionManager(backend=backend, language="auto", sample_rate=16000)
    await mgr.on_speech_start()
    state = {
        "asr_active": True, "asr_active_gen": 1,
        "asr_audio_samples_accepted": 100, "asr_turn_started_at": 123.0,
        "endpoint_pending": None, "endpoint_pending_gen": None,
    }
    for transient in (SessionState.FINALIZING, SessionState.CANCELLING,
                      SessionState.ERROR_REBUILD):
        mgr._state = transient  # white-box: simulate the transient window
        assert _asr_manager_idle(mgr) is False
        # barge-in stays barge-in-only (no spurious re-open)
        assert _speech_start_is_bargein_only(state, mgr, post_fix=True) is True
        # reconcile does NOT tear down the live turn
        _reconcile_after_suppressed_final(state, mgr)
        assert state["asr_active"] is True


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok {_name}")
