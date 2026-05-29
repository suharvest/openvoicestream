"""Phase 1a acceptance test.

Proves the voxedge architecture runs an entire fake conversation end-to-end
on a laptop with NO CUDA: mock backends + InProcessTransport, driven purely
through the public Transport API. Asserts the server event sequence matches
the V2V protocol contract.
"""
from __future__ import annotations

import asyncio

import numpy as np

from voxedge.backends.mock import MockASR, MockLLM, MockTTS, MockVAD
from voxedge.engine import ConversationEngine
from voxedge.transport import InProcessTransport

# No pytest-asyncio dependency (keep voxedge core dep-free). Each async test
# body is wrapped with asyncio.run via this tiny decorator — the InProcess
# transport + queues must be constructed *inside* the running loop, so the
# coroutine factory pattern is required.


def run_async(coro_fn):
    """Decorate a zero-arg async test fn so pytest runs it via asyncio.run."""

    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


def _pcm(loud: bool, n: int = 512) -> bytes:
    """int16 PCM chunk: loud = speech, silent = below VAD threshold."""
    if loud:
        arr = (np.ones(n, dtype=np.int16) * 8000)
    else:
        arr = np.zeros(n, dtype=np.int16)
    return arr.tobytes()


async def _collect_events(transport: InProcessTransport, run_coro):
    """Run the engine to completion, then drain outbound events + audio.

    The InProcess outbound queues are unbounded, so the engine never blocks on
    send — it runs to a clean stop (session close), and we collect afterwards.
    A timeout guards against a regression that wedges the loop.
    """
    await asyncio.wait_for(run_coro, timeout=10.0)
    events = transport.drain_events_nowait()
    audio = transport.drain_audio_nowait()
    return events, audio


# ───────────────────────────────────────────────────────────────────────


@run_async
async def test_single_utterance_asr_only():
    """Feed loud-then-silent audio → VAD segments → asr_partial → asr_endpoint
    → asr_final. Single-utterance ends the session."""
    engine = ConversationEngine(
        backends={
            "asr": MockASR(transcript="hello world", language="English"),
            # VAD drives the segmentation this test asserts (speech_start /
            # speech_end → endpoint). Without a VAD backend the engine takes
            # its no-VAD lazy-open path (app/main.py:2901-2921) which never
            # emits vad_event and relies on client asr_eos for the endpoint —
            # a different contract than the one under test here.
            "vad": MockVAD(silence_chunks=2),
        },
        multi_utterance=False,
    )
    transport = InProcessTransport()

    # 3 loud chunks (speech) then 3 silent (VAD silence_chunks=2 → speech_end)
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=True))
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=False))
    transport.end_input()

    events, _audio = await _collect_events(transport, engine.run(transport))
    types = [e["type"] for e in events]

    assert "vad_event" in types
    assert any(e["type"] == "vad_event" and e["event"] == "speech_start" for e in events)
    assert any(e["type"] == "vad_event" and e["event"] == "speech_end" for e in events)
    assert "asr_partial" in types
    assert "asr_endpoint" in types
    finals = [e for e in events if e["type"] == "asr_final"]
    assert len(finals) == 1
    assert finals[0]["text"] == "hello world"
    assert finals[0].get("language") == "English"


@run_async
async def test_client_text_drives_tts():
    """TTS-only path: CLIENT_TEXT → sentence buffer → tts_started + PCM +
    tts_sentence_done + tts_done. Exercises the sample-rate header too."""
    engine = ConversationEngine(
        backends={"tts": MockTTS(sample_rate=24000)},
        multi_utterance=False,
    )
    transport = InProcessTransport()
    await transport.feed_event({"type": "text", "text": "Hello there. How are you?"})
    await transport.feed_event({"type": "tts_flush"})
    transport.end_input()

    events, audio = await _collect_events(transport, engine.run(transport))
    types = [e["type"] for e in events]

    assert "tts_started" in types
    assert "tts_sentence_done" in types
    assert "tts_done" in types
    # First audio frame is the 4-byte little-endian sample-rate header.
    assert len(audio) >= 2
    import struct
    assert struct.unpack("<I", audio[0])[0] == 24000
    # At least some PCM bytes followed.
    assert sum(len(a) for a in audio[1:]) > 0


@run_async
async def test_full_loop_asr_llm_tts_multiturn():
    """Full closed loop, multi-utterance: audio → asr_final → LLM → TTS.

    Asserts the ordered contract: asr_final(session_complete=False) then TTS
    events, then on session close asr_final(session_complete=True) + final
    tts_done(session_complete=True)."""
    engine = ConversationEngine(
        backends={
            "asr": MockASR(transcript="what time is it"),
            "vad": MockVAD(silence_chunks=2),
            "llm": MockLLM(reply="It is noon."),
            "tts": MockTTS(),
        },
        multi_utterance=True,
    )
    transport = InProcessTransport()

    # Turn 1: speech then silence.
    for _ in range(2):
        await transport.feed_audio(_pcm(loud=True))
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=False))
    # Close the session via client_eos so multi-utterance terminates.
    await transport.feed_event({"type": "asr_eos"})
    transport.end_input()

    events, audio = await _collect_events(transport, engine.run(transport))
    types = [e["type"] for e in events]

    # ASR produced a final.
    finals = [e for e in events if e["type"] == "asr_final"]
    assert finals, f"no asr_final in {types}"
    assert finals[-1]["text"] == "what time is it"

    # LLM→TTS loop produced TTS output.
    assert "tts_started" in types
    assert any(e["type"] == "tts_sentence_done" for e in events)
    # Final tts_done carries session_complete=True in multi-utterance.
    done = [e for e in events if e["type"] == "tts_done"]
    assert done, "no tts_done"
    assert done[-1].get("session_complete") is True
    # Some audio was synthesized.
    assert len(audio) >= 1


@run_async
async def test_event_ordering_asr_final_before_tts():
    """Within a turn, asr_final must precede the TTS audio it triggers."""
    engine = ConversationEngine(
        backends={
            "asr": MockASR(transcript="ping"),
            "vad": MockVAD(silence_chunks=2),
            "llm": MockLLM(reply="pong."),
            "tts": MockTTS(),
        },
        multi_utterance=False,
    )
    transport = InProcessTransport()
    for _ in range(2):
        await transport.feed_audio(_pcm(loud=True))
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=False))
    transport.end_input()

    events, _audio = await _collect_events(transport, engine.run(transport))
    types = [e["type"] for e in events]
    assert "asr_final" in types and "tts_started" in types
    assert types.index("asr_final") < types.index("tts_started")
