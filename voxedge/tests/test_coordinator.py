"""Slot-layer concurrency abstraction tests (migrated from app/core, spec §3.1).

Covers:
  * ceiling = min(asr.max_concurrent, tts.max_concurrent) — several combos.
  * coordinator mode resolution: exclusive honored, concurrent→serialized
    downgrade when either backend can't run parallel.
  * serialized mode mutually-excludes ASR/TTS slot work; concurrent overlaps.
  * the engine still runs with no coordinator (backward compatible) and runs
    with one wired.
"""
from __future__ import annotations

import asyncio

import numpy as np

from voxedge.backends.base import ASRBackend, TTSBackend
from voxedge.backends.mock import MockASR, MockLLM, MockTTS, MockVAD
from voxedge.engine import (
    BackendCoordinator,
    ConcurrencyCapability,
    ConversationEngine,
)
from voxedge.engine.capability_resolver import capability_of, resolve
from voxedge.transport import InProcessTransport


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


def _cap(max_concurrent, supports_parallel) -> ConcurrencyCapability:
    return ConcurrencyCapability(
        supports_parallel=supports_parallel, max_concurrent=max_concurrent
    )


# ───────────────────────────────────────────────────────────────────────
# 1. ceiling = min(asr, tts)
# ───────────────────────────────────────────────────────────────────────


def test_ceiling_min_of_asr_tts():
    # (asr_max, tts_max) -> expected ceiling (None == +inf)
    cases = [
        ((1, 1), 1),
        ((2, 2), 2),
        ((4, 2), 2),
        ((2, 4), 2),
        ((1, 4), 1),
        ((None, 2), 2),   # asr=inf -> tts wins
        ((2, None), 2),   # tts=inf -> asr wins
        ((None, None), None),  # both inf -> no fixed cap
    ]
    for (asr_max, tts_max), expected in cases:
        asr = MockASR(concurrency=_cap(asr_max, (asr_max or 2) > 1))
        tts = MockTTS(concurrency=_cap(tts_max, (tts_max or 2) > 1))
        r = resolve(asr_backend=asr, tts_backend=tts, requested_mode="concurrent")
        assert r.ceiling == expected, (asr_max, tts_max, r.ceiling)


def test_capability_of_dict_form_and_default():
    # ABC default (no override) → conservative serialized single-slot.
    cap = capability_of(MockASR())
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False
    # Plain dict form from a backend that reports concurrent mode.
    cap2 = capability_of(MockTTS(concurrency={"max_concurrency": 3, "mode": "concurrent"}))
    assert cap2.max_concurrent == 3
    assert cap2.supports_parallel is True
    # None backend → default.
    assert capability_of(None).max_concurrent == 1


# ───────────────────────────────────────────────────────────────────────
# 2. coordinator mode resolution (spec §4)
# ───────────────────────────────────────────────────────────────────────


def test_mode_downgrade_concurrent_to_serialized():
    # ASR parallel-capable, TTS single-slot → concurrent must downgrade.
    asr = MockASR(concurrency=_cap(2, True))
    tts = MockTTS(concurrency=_cap(1, False))
    r = resolve(asr_backend=asr, tts_backend=tts, requested_mode="concurrent")
    assert r.coordinator_mode == "serialized"

    coord = BackendCoordinator.from_backends(asr=asr, tts=tts, requested_mode="concurrent")
    assert coord.mode == "serialized"


def test_mode_concurrent_when_both_parallel():
    asr = MockASR(concurrency=_cap(2, True))
    tts = MockTTS(concurrency=_cap(2, True))
    r = resolve(asr_backend=asr, tts_backend=tts, requested_mode="concurrent")
    assert r.coordinator_mode == "concurrent"
    coord = BackendCoordinator.from_backends(asr=asr, tts=tts, requested_mode="concurrent")
    assert coord.mode == "concurrent"


def test_mode_exclusive_always_honored():
    # Even with both parallel-capable, exclusive is honored as-is.
    asr = MockASR(concurrency=_cap(4, True))
    tts = MockTTS(concurrency=_cap(4, True))
    r = resolve(asr_backend=asr, tts_backend=tts, requested_mode="exclusive")
    assert r.coordinator_mode == "exclusive"


def test_mode_serialized_passthrough():
    asr = MockASR(concurrency=_cap(2, True))
    tts = MockTTS(concurrency=_cap(2, True))
    r = resolve(asr_backend=asr, tts_backend=tts, requested_mode="serialized")
    assert r.coordinator_mode == "serialized"


# ───────────────────────────────────────────────────────────────────────
# 3. acquire() exclusion semantics
# ───────────────────────────────────────────────────────────────────────


async def _overlap_probe(coord: BackendCoordinator) -> bool:
    """Returns True if an ASR slot and a TTS slot overlapped in time."""
    state = {"in_flight": 0, "max_in_flight": 0}

    async def hold(slot: str):
        async with coord.acquire(slot):
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            # yield control so the other task can attempt to enter.
            await asyncio.sleep(0.02)
            state["in_flight"] -= 1

    await asyncio.gather(hold("asr"), hold("tts"))
    return state["max_in_flight"] > 1


@run_async
async def test_serialized_mutually_excludes():
    coord = BackendCoordinator(mode="serialized")
    overlapped = await _overlap_probe(coord)
    assert overlapped is False, "serialized mode must not overlap asr/tts"


@run_async
async def test_concurrent_overlaps():
    coord = BackendCoordinator(mode="concurrent")
    overlapped = await _overlap_probe(coord)
    assert overlapped is True, "concurrent mode must allow asr/tts overlap"


@run_async
async def test_exclusive_unloads_dormant_backend():
    unloaded: list[str] = []

    class _ASR(ASRBackend):
        @property
        def name(self):
            return "a"

        @property
        def capabilities(self):
            return set()

        @property
        def sample_rate(self):
            return 16000

        def is_ready(self):
            return True

        def preload(self):
            pass

        def transcribe(self, audio_bytes, language="auto"):
            raise NotImplementedError

        def unload(self):
            unloaded.append("asr")

    class _TTS(TTSBackend):
        @property
        def name(self):
            return "t"

        @property
        def capabilities(self):
            return set()

        @property
        def sample_rate(self):
            return 16000

        def is_ready(self):
            return True

        def preload(self):
            pass

        def synthesize(self, text, **kw):
            raise NotImplementedError

        def unload(self):
            unloaded.append("tts")

    coord = BackendCoordinator.from_backends(
        asr=_ASR(), tts=_TTS(), requested_mode="exclusive"
    )
    assert coord.mode == "exclusive"
    # acquire asr, then tts → switching slot unloads the dormant asr backend.
    async with coord.acquire("asr"):
        pass
    async with coord.acquire("tts"):
        pass
    assert unloaded == ["asr"]


# ───────────────────────────────────────────────────────────────────────
# 4. engine wiring — no coordinator vs coordinator present
# ───────────────────────────────────────────────────────────────────────


def _pcm(loud: bool, n: int = 512) -> bytes:
    arr = (np.ones(n, dtype=np.int16) * 8000) if loud else np.zeros(n, dtype=np.int16)
    return arr.tobytes()


async def _drive_full_turn(engine: ConversationEngine):
    transport = InProcessTransport()
    run_coro = engine.run(transport)
    task = asyncio.create_task(run_coro)
    # speech then silence (drives VAD endpoint), then end input.
    for _ in range(2):
        await transport.feed_audio(_pcm(True))
    for _ in range(3):
        await transport.feed_audio(_pcm(False))
    await transport.feed_event({"type": "text", "text": "hi there."})
    await transport.feed_event({"type": "tts_flush"})
    transport.end_input()
    await asyncio.wait_for(task, timeout=10.0)
    return transport.drain_events_nowait(), transport.drain_audio_nowait()


@run_async
async def test_engine_no_coordinator_runs():
    engine = ConversationEngine(
        {"asr": MockASR(), "tts": MockTTS(), "vad": MockVAD(), "llm": MockLLM()},
    )
    assert engine.coordinator is None
    events, audio = await _drive_full_turn(engine)
    types = [e.get("type") for e in events]
    assert "asr_final" in types
    assert any(t == "tts_started" for t in types)
    assert len(audio) > 0


@run_async
async def test_engine_with_serialized_coordinator_runs():
    asr = MockASR(concurrency=_cap(2, True))
    tts = MockTTS(concurrency=_cap(1, False))
    coord = BackendCoordinator.from_backends(asr=asr, tts=tts, requested_mode="concurrent")
    assert coord.mode == "serialized"
    engine = ConversationEngine(
        {"asr": asr, "tts": tts, "vad": MockVAD(), "llm": MockLLM()},
        coordinator=coord,
    )
    events, audio = await _drive_full_turn(engine)
    types = [e.get("type") for e in events]
    # Serialized coordinator must not break the normal flow.
    assert "asr_final" in types
    assert any(t == "tts_started" for t in types)
    assert len(audio) > 0
