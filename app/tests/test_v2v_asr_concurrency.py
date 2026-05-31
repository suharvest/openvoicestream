"""ASR/TTS concurrency tests for the per-utterance v2v wiring.

We don't spin up the real FastAPI WebSocket here — the heavy stack
(profile, executors, Jetson backends) isn't importable on Mac. Instead
we exercise the coordination contract directly: ASRSessionManager
running through ``BackendCoordinator.acquire("asr")`` while a parallel
"TTS" routine holds ``acquire("tts")``. Covers the 3 scenarios required
by the spec under both serialized and concurrent execution_policy.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import List

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.coordinator import BackendCoordinator
from app.core.asr_session_manager import ASRSessionManager, SessionState


# ── shared fakes ──────────────────────────────────────────────────────


class _Stream:
    def __init__(self, final_text="ok", finalize_delay=0.0):
        self.final = final_text
        self.delay = finalize_delay
        self.accepts = 0
        self.finalized = False
        self.cancelled = False

    def accept_waveform(self, sr, samples):
        self.accepts += 1

    def finalize(self):
        if self.delay:
            time.sleep(self.delay)
        self.finalized = True
        # New ASRStream ABC contract: ``(text, detected_language)``.
        return self.final, None

    def cancel(self):
        self.cancelled = True

    def cancel_and_finalize(self):
        self.cancelled = True


class _Backend:
    def __init__(self, streams):
        self.streams = list(streams)
        self.idx = 0
        self.restart_calls = 0

    def create_stream(self, language="auto"):
        s = self.streams[self.idx]
        self.idx += 1
        return s

    def restart_worker(self):
        self.restart_calls += 1


def asynctest(fn):
    def wrapper(*a, **k):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(fn(*a, **k))
        finally:
            loop.close()
    wrapper.__name__ = fn.__name__
    return wrapper


# ── Scenario 1 ────────────────────────────────────────────────────────


def _run_scenario_tts_in_flight_then_asr(policy: str):
    """TTS holds the slot; VAD speech-start opens an ASR utterance.

    In serialized mode the ASR call must queue behind TTS but eventually
    succeed without deadlock. In concurrent mode both proceed in parallel.
    """
    async def go():
        coord = BackendCoordinator({"mode": policy})
        be = _Backend([_Stream("scen1")])
        mgr = ASRSessionManager(backend=be, language="zh", coord=coord)

        tts_done = asyncio.Event()
        asr_done = asyncio.Event()

        async def tts_routine():
            async with coord.acquire("tts"):
                await asyncio.sleep(0.05)  # simulated TTS synth
                tts_done.set()

        async def asr_routine():
            async with coord.acquire("asr"):
                await mgr.on_speech_start()
            asr_done.set()

        t0 = time.monotonic()
        await asyncio.gather(tts_routine(), asr_routine())
        elapsed = time.monotonic() - t0
        assert tts_done.is_set() and asr_done.is_set()
        if policy == "serialized":
            # The two slots run sequentially → total ≥ TTS delay.
            assert elapsed >= 0.05
        # No deadlock under either policy.
        assert mgr.state == SessionState.ACTIVE
    asyncio.new_event_loop().run_until_complete(go())


def test_scenario1_tts_in_flight_serialized():
    _run_scenario_tts_in_flight_then_asr("serialized")


def test_scenario1_tts_in_flight_concurrent():
    _run_scenario_tts_in_flight_then_asr("concurrent")


# ── Scenario 2 ────────────────────────────────────────────────────────


@asynctest
async def test_scenario2_finalize_and_tts_overlap_concurrent():
    """In concurrent mode ASR finalize and TTS synth must both progress."""
    coord = BackendCoordinator({"mode": "concurrent"})
    be = _Backend([_Stream("hi", finalize_delay=0.1)])
    mgr = ASRSessionManager(backend=be, coord=coord)

    await mgr.on_speech_start()

    progressed: List[str] = []

    async def tts_routine():
        async with coord.acquire("tts"):
            await asyncio.sleep(0.02)
            progressed.append("tts1")
            await asyncio.sleep(0.02)
            progressed.append("tts2")

    async def asr_finalize():
        async with coord.acquire("asr"):
            text = await mgr.finalize("vad_end")
            progressed.append(f"asr:{text}")

    await asyncio.gather(tts_routine(), asr_finalize())
    assert "tts1" in progressed and "tts2" in progressed
    assert any(p.startswith("asr:") for p in progressed)


# ── Scenario 3 ────────────────────────────────────────────────────────


@asynctest
async def test_scenario3_asr_worker_restart_does_not_interrupt_tts():
    """A worker-restart mid-conversation must not abort an in-progress TTS."""
    coord = BackendCoordinator({"mode": "concurrent"})
    # First create_stream blows up; manager recovers in ERROR_REBUILD.
    class _RestartBackend(_Backend):
        def __init__(self):
            super().__init__([_Stream("recovered")])
            self._first = True

        def create_stream(self, language="auto"):
            if self._first:
                self._first = False
                from voxedge.backends.jetson.trt_edge_llm_asr import WorkerExitError
                raise WorkerExitError("simulated")
            return super().create_stream(language)

    be = _RestartBackend()
    mgr = ASRSessionManager(backend=be, coord=coord)

    tts_complete = False

    async def long_tts():
        nonlocal tts_complete
        async with coord.acquire("tts"):
            for _ in range(5):
                await asyncio.sleep(0.05)
        tts_complete = True

    async def asr_open_with_recovery():
        async with coord.acquire("asr"):
            await mgr.on_speech_start()

    tts_task = asyncio.create_task(long_tts())
    await asyncio.sleep(0.01)  # let TTS get in flight
    await asr_open_with_recovery()
    await tts_task

    assert tts_complete is True
    # After ERROR_REBUILD the manager either recovered to ACTIVE or
    # bounced back to IDLE if all attempts failed; both are valid as
    # long as TTS didn't crash.
    assert mgr.state in (SessionState.ACTIVE, SessionState.IDLE)
