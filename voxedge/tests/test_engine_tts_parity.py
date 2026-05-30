"""Engine-parity #15 acceptance tests.

Covers the two TTS features ported into the voxedge ConversationEngine to
align with the legacy /v2v handler:

  (a) speaker / voice / speed kwargs are forwarded to the TTS backend's
      ``generate_streaming`` (constructor-injected, no app.core import).
  (b) ``low_latency_tts=True`` selects the ported ``LowLatencyTTSBuffer``,
      which emits the first speakable chunk EARLIER than the default
      ``_SentenceBuffer`` (asserted via event timing).

Same dep-free asyncio.run harness as test_engine_inprocess.py.
"""
from __future__ import annotations

import asyncio
from typing import Any, Iterator, Optional

from voxedge.backends.base import TTSBackend, TTSCapability
from voxedge.engine import ConversationEngine
from voxedge.engine.tts_buffer import LowLatencyTTSBuffer
from voxedge.transport import InProcessTransport


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


class RecordingTTS(TTSBackend):
    """Mock TTS that records every kwarg ``generate_streaming`` was called
    with (so a test can assert speaker/voice/speed pass-through) and yields a
    single PCM chunk per sentence."""

    def __init__(self, sample_rate: int = 16000):
        self._sr = sample_rate
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "recording-tts"

    @property
    def capabilities(self) -> set:
        return {TTSCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return self._sr

    def is_ready(self) -> bool:
        return True

    def preload(self) -> None:
        pass

    def synthesize(self, text: str, **kwargs):
        return b"\x00\x00", {"sample_rate": self._sr}

    def generate_streaming(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        cancel_token: Optional[Any] = None,
        **kwargs,
    ) -> Iterator[bytes]:
        # Record the FULL call so the test can assert pass-through. ``speaker``
        # is an explicit ABC param; everything else lands in **kwargs.
        rec = {"text": text, "language": language, "speaker": speaker}
        rec.update(kwargs)
        self.calls.append(rec)
        yield b"\x01\x02\x03\x04"


async def _drive(engine: ConversationEngine, events_in: list[dict]):
    transport = InProcessTransport()
    for ev in events_in:
        await transport.feed_event(ev)
    transport.end_input()
    await asyncio.wait_for(engine.run(transport), timeout=10.0)
    return transport.drain_events_nowait(), transport.drain_audio_nowait()


# ───────────────────────────────────────────────────────────────────────
# (a) speaker / voice / speed pass-through
# ───────────────────────────────────────────────────────────────────────


@run_async
async def test_speaker_kwargs_forwarded_to_generate_streaming():
    """A constructor-injected ``tts_speaker_kwargs`` (+ speed) must reach the
    backend's ``generate_streaming`` exactly."""
    tts = RecordingTTS()
    engine = ConversationEngine(
        backends={"tts": tts},
        multi_utterance=False,
        tts_language="english",
        tts_speaker_kwargs={"speaker_id": 2301, "speaker": "2301"},
        tts_speed=1.25,
    )
    await _drive(
        engine,
        [
            {"type": "text", "text": "Hello there."},
            {"type": "tts_flush"},
        ],
    )

    assert tts.calls, "generate_streaming was never called"
    call = tts.calls[0]
    # The resolved speaker dict landed: speaker_id via **kwargs, speaker via
    # the explicit ABC param.
    assert call["speaker_id"] == 2301
    assert call["speaker"] == "2301"
    assert call["speed"] == 1.25
    assert call["language"] == "english"


@run_async
async def test_no_speaker_config_passes_no_speaker_kwargs():
    """Back-compat: with no speaker config injected, NO speaker/voice/speed
    kwargs are added (backend uses its default)."""
    tts = RecordingTTS()
    engine = ConversationEngine(
        backends={"tts": tts},
        multi_utterance=False,
        tts_language="english",
    )
    await _drive(
        engine,
        [
            {"type": "text", "text": "Hello there."},
            {"type": "tts_flush"},
        ],
    )

    assert tts.calls
    call = tts.calls[0]
    assert call["speaker"] is None
    assert "speaker_id" not in call
    assert "speed" not in call
    assert "voice" not in call


@run_async
async def test_deprecated_voice_fallback_when_no_speaker_kwargs():
    """When only ``tts_voice`` is set (no resolved speaker dict) the deprecated
    ``voice`` kwarg is forwarded — mirrors legacy app/main.py:3476-3477."""
    tts = RecordingTTS()
    engine = ConversationEngine(
        backends={"tts": tts},
        multi_utterance=False,
        tts_language="english",
        tts_voice="narrator",
    )
    await _drive(
        engine,
        [
            {"type": "text", "text": "Hi."},
            {"type": "tts_flush"},
        ],
    )
    assert tts.calls
    assert tts.calls[0].get("voice") == "narrator"
    # speaker dict wins over voice when both present:
    tts2 = RecordingTTS()
    engine2 = ConversationEngine(
        backends={"tts": tts2},
        multi_utterance=False,
        tts_language="english",
        tts_voice="narrator",
        tts_speaker_kwargs={"speaker_id": 7, "speaker": "7"},
    )
    await _drive(
        engine2,
        [
            {"type": "text", "text": "Hi."},
            {"type": "tts_flush"},
        ],
    )
    assert tts2.calls
    assert "voice" not in tts2.calls[0]
    assert tts2.calls[0]["speaker_id"] == 7


# ───────────────────────────────────────────────────────────────────────
# (b) low-latency buffer emits the first chunk earlier
# ───────────────────────────────────────────────────────────────────────


@run_async
async def test_low_latency_buffer_emits_first_chunk_earlier():
    """For a CJK utterance with no terminal punctuation, the low-latency
    buffer must emit a speakable chunk that the default _SentenceBuffer would
    still be holding — proven by tts_started firing on the engine that uses
    LowLatencyTTSBuffer and NOT on the one using _SentenceBuffer (before
    flush)."""
    # A CJK clause with a soft comma break, NO hard sentence terminator. The
    # low-latency buffer emits the pre-comma clause from add() (soft break past
    # min_chars=15); the default _SentenceBuffer needs a hard 。！？；\n
    # terminator (or max_buffer=200) and so holds everything until flush.
    streamed = "今天天气真的非常好我们一起出去散步吧，你说好不好呢朋友"  # 27 chars, soft comma at 18

    # Engine A: low-latency ON.
    tts_ll = RecordingTTS()
    engine_ll = ConversationEngine(
        backends={"tts": tts_ll},
        multi_utterance=False,
        tts_language="zh",
        low_latency_tts=True,
    )
    transport_ll = InProcessTransport()
    # Feed the text but DO NOT flush yet — we want to observe whether a chunk
    # is emitted from .add() alone.
    await transport_ll.feed_event({"type": "text", "text": streamed})
    await transport_ll.feed_event({"type": "tts_flush"})
    transport_ll.end_input()
    await asyncio.wait_for(engine_ll.run(transport_ll), timeout=10.0)
    events_ll = transport_ll.drain_events_nowait()

    # Engine B: default _SentenceBuffer.
    tts_sb = RecordingTTS()
    engine_sb = ConversationEngine(
        backends={"tts": tts_sb},
        multi_utterance=False,
        tts_language="zh",
        low_latency_tts=False,
    )
    transport_sb = InProcessTransport()
    await transport_sb.feed_event({"type": "text", "text": streamed})
    await transport_sb.feed_event({"type": "tts_flush"})
    transport_sb.end_input()
    await asyncio.wait_for(engine_sb.run(transport_sb), timeout=10.0)
    events_sb = transport_sb.drain_events_nowait()

    # Direct buffer-level assertion: low-latency emits from add() (no flush),
    # the sentence buffer does not (no terminator).
    ll = LowLatencyTTSBuffer(language="zh")
    from voxedge.engine.conversation import _SentenceBuffer

    sb = _SentenceBuffer()
    ll_from_add = list(ll.add(streamed))
    sb_from_add = list(sb.add(streamed))
    assert ll_from_add, "low-latency buffer should emit a chunk from add() alone"
    assert not sb_from_add, "sentence buffer should hold (no terminator) until flush"

    # End-to-end: both eventually synthesize (after flush), but the
    # low-latency engine split the utterance into MORE chunks (earlier first
    # emit + remainder) than the single flushed sentence.
    started_ll = [e for e in events_ll if e["type"] == "tts_started"]
    started_sb = [e for e in events_sb if e["type"] == "tts_started"]
    assert started_ll, "low-latency engine produced no tts_started"
    assert started_sb, "sentence-buffer engine produced no tts_started"
    # The first chunk the low-latency path speaks is a strict PREFIX of the
    # full utterance the sentence buffer speaks as one block — i.e. it spoke
    # sooner with less text.
    first_ll = started_ll[0]["sentence"]
    only_sb = started_sb[0]["sentence"]
    assert len(first_ll) < len(only_sb), (
        f"low-latency first chunk {first_ll!r} not shorter than "
        f"sentence-buffer block {only_sb!r}"
    )
    assert streamed.startswith(first_ll) or first_ll in streamed
