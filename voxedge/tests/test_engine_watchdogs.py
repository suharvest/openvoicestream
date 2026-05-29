"""M1/M3/M4 + M2-semantics regression tests.

Guards the 5 production-critical behaviours re-introduced into the voxedge
engine (codex review 2026-05-29):

  M1  ASR per-turn wall-clock timeout → cancel + error, no spurious final.
  M3  TTS per-chunk + per-sentence watchdog → abort wedged synth + error.
  M4  backend ``PoolSaturatedError`` → typed ``pool_saturated`` event (4429).
  M2  ASRSessionManager generation/finalize semantics: partials don't leak
      across a barge-in, and a stale (cancelled) finalize is suppressed.

All mock-backend only — no CUDA, no env.
"""
from __future__ import annotations

import asyncio

import numpy as np

from voxedge.backends.mock import MockASR, MockLLM, MockTTS, MockVAD
from voxedge.engine import ConversationEngine
from voxedge.engine.asr_session_manager import ASRSessionManager, SessionState
from voxedge.transport import InProcessTransport


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


def _pcm(loud: bool, n: int = 512) -> bytes:
    arr = (np.ones(n, dtype=np.int16) * 8000) if loud else np.zeros(n, dtype=np.int16)
    return arr.tobytes()


async def _run(transport, run_coro, timeout=10.0):
    await asyncio.wait_for(run_coro, timeout=timeout)
    return transport.drain_events_nowait(), transport.drain_audio_nowait()


# ── M1: ASR per-turn wall-clock timeout ───────────────────────────────────


@run_async
async def test_m1_asr_turn_timeout_cancels_and_errors():
    """An active ASR turn that never reaches an endpoint must hit the
    per-turn deadline → typed error, no asr_final, session unwinds."""
    engine = ConversationEngine(
        backends={
            "asr": MockASR(transcript="never finishes"),
            "vad": MockVAD(silence_chunks=2),
        },
        multi_utterance=False,
        timeouts={"asr_turn": 0.2},  # M1 deadline, constructor-injected
    )
    transport = InProcessTransport()
    # Loud audio only → speech_start, turn goes ACTIVE; NEVER silent → no
    # speech_end → the turn never endpoints on its own.
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=True))
    # Do NOT end_input immediately — keep the audio loop alive a touch so the
    # turn stays active past the deadline. The asr_out_task watchdog fires
    # regardless once elapsed > 0.2s.
    await asyncio.sleep(0.4)
    transport.end_input()

    events, _ = await _run(transport, engine.run(transport))
    types = [e["type"] for e in events]

    # The deadline error was surfaced.
    errs = [e for e in events if e["type"] == "error"]
    assert errs, f"expected a timeout error, got {types}"
    assert any("deadline" in (e.get("error") or "") for e in errs), errs
    # No spurious asr_final was emitted for the wedged turn.
    assert "asr_final" not in types, f"unexpected asr_final in {types}"


# ── M3: TTS per-chunk watchdog ─────────────────────────────────────────────


@run_async
async def test_m3_tts_chunk_watchdog_aborts_wedged_synth():
    """A synth that produces no chunk within the chunk-timeout must be
    aborted with an error, and the engine must not hang."""
    engine = ConversationEngine(
        backends={"tts": MockTTS(sample_rate=16000, first_chunk_block_s=5.0)},
        multi_utterance=False,
        # chunk watchdog short; sentence deadline a bit longer so the chunk
        # watchdog is the one that fires.
        timeouts={"tts_chunk": 0.2, "tts_sentence": 5.0},
    )
    transport = InProcessTransport()
    await transport.feed_event({"type": "text", "text": "Hello there."})
    await transport.feed_event({"type": "tts_flush"})
    transport.end_input()

    events, _ = await _run(transport, engine.run(transport), timeout=5.0)
    errs = [e for e in events if e["type"] == "error"]
    assert errs, f"expected a tts watchdog error, got {[e['type'] for e in events]}"
    assert any("no chunks" in (e.get("error") or "") for e in errs), errs
    # tts_started was emitted (drain entered) but the sentence still 'done'd
    # so the loop didn't wedge.
    types = [e["type"] for e in events]
    assert "tts_started" in types
    assert "tts_sentence_done" in types


# ── M3: TTS per-sentence deadline (wedge before first chunk watchdog) ──────


@run_async
async def test_m3_tts_sentence_deadline_fires():
    """If the chunk watchdog is longer than the sentence deadline, the outer
    per-sentence deadline must still abort a wedged synth."""
    engine = ConversationEngine(
        backends={"tts": MockTTS(sample_rate=16000, first_chunk_block_s=5.0)},
        multi_utterance=False,
        timeouts={"tts_chunk": 5.0, "tts_sentence": 0.2},
    )
    transport = InProcessTransport()
    await transport.feed_event({"type": "text", "text": "Hello there."})
    await transport.feed_event({"type": "tts_flush"})
    transport.end_input()

    events, _ = await _run(transport, engine.run(transport), timeout=5.0)
    errs = [e for e in events if e["type"] == "error"]
    assert errs, f"expected a per-sentence deadline error, got {events}"
    assert any("per-sentence deadline" in (e.get("error") or "") for e in errs), errs


# ── M4: PoolSaturatedError → typed pool_saturated (TTS path) ───────────────


@run_async
async def test_m4_tts_pool_saturated_typed_error():
    """A backend that raises PoolSaturatedError must surface a typed
    ``pool_saturated`` (status 4429) event, not a generic ``tts: ...``."""
    engine = ConversationEngine(
        backends={"tts": MockTTS(saturate=True, max_slots=3)},
        multi_utterance=False,
    )
    transport = InProcessTransport()
    await transport.feed_event({"type": "text", "text": "Hi."})
    await transport.feed_event({"type": "tts_flush"})
    transport.end_input()

    events, _ = await _run(transport, engine.run(transport))
    sat = [e for e in events if e.get("error") == "pool_saturated"]
    assert sat, f"expected pool_saturated, got {events}"
    assert sat[0].get("status") == 4429
    assert sat[0].get("max_slots") == 3
    # Must NOT be reported as a generic synth error.
    assert not any((e.get("error") or "").startswith("tts:") for e in events), events


# ── M4: PoolSaturatedError → typed pool_saturated (ASR open path) ──────────


@run_async
async def test_m4_asr_pool_saturated_on_open():
    """A saturated ASR backend (reject at stream open) must surface a typed
    pool_saturated and NOT flag the turn active."""
    engine = ConversationEngine(
        backends={
            "asr": MockASR(saturate_on_create=True, max_slots=2),
            "vad": MockVAD(silence_chunks=2),
        },
        multi_utterance=False,
    )
    transport = InProcessTransport()
    for _ in range(2):
        await transport.feed_audio(_pcm(loud=True))
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=False))
    transport.end_input()

    events, _ = await _run(transport, engine.run(transport))
    sat = [e for e in events if e.get("error") == "pool_saturated"]
    assert sat, f"expected pool_saturated, got {[e['type'] for e in events]}"
    assert sat[0].get("status") == 4429
    assert sat[0].get("max_slots") == 2


# ── M2: ASRSessionManager generation/finalize semantics ────────────────────


@run_async
async def test_m2_partial_does_not_leak_across_generation():
    """A partial snapshot taken against generation N must carry gen N, and a
    new utterance must bump the generation so stale-gen partials can be
    dropped by the caller (engine's BUG-4 gate)."""
    mgr = ASRSessionManager(MockASR(transcript="one two three"))
    gen1 = await mgr.on_speech_start()
    await mgr.accept_audio(np.zeros(160, dtype=np.float32))
    g, partial, _ = await mgr.get_partial_for_generation()
    assert g == gen1 and partial  # partial belongs to gen1

    # Barge-in: a new utterance preempts gen1 → fresh generation.
    gen2 = await mgr.on_speech_start()
    assert gen2 == gen1 + 1
    g2, _, _ = await mgr.get_partial_for_generation()
    assert g2 == gen2  # snapshot now reflects the new generation only


@run_async
async def test_m2_finalize_accepted_then_stale_suppressed():
    """finalize_with_status returns accepted=True for a live turn; a finalize
    attempted after the turn already finalized (IDLE) is suppressed
    (accepted=False, empty text) instead of double-emitting."""
    mgr = ASRSessionManager(MockASR(transcript="done", language="English"))
    await mgr.on_speech_start()
    gen, text, accepted, lang = await mgr.finalize_with_status("vad_end")
    assert accepted is True and text == "done" and lang == "English"
    assert mgr.state == SessionState.IDLE

    # A second finalize with no active turn must be a suppressed no-op.
    gen2, text2, accepted2, lang2 = await mgr.finalize_with_status("vad_end")
    assert accepted2 is False and text2 == "" and lang2 is None


@run_async
async def test_m2_cancel_suppresses_finalize_race():
    """A finalize that races a cancel must be discarded (accepted=False)."""
    mgr = ASRSessionManager(MockASR(transcript="hi"))
    await mgr.on_speech_start()
    await mgr.cancel("bargein")
    assert mgr.state == SessionState.IDLE
    # finalize after cancel → no active stream → suppressed.
    gen, text, accepted, lang = await mgr.finalize_with_status("vad_end")
    assert accepted is False and text == ""
