"""Mock backends for CUDA-free, laptop-only verification.

These let the whole ``ConversationEngine`` run end-to-end on a Mac with no
GPU — the Phase 1a acceptance goal. They implement the voxedge ABCs with
trivial deterministic behaviour.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Optional

import numpy as np

from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    LLMBackend,
    LLMEvent,
    TranscriptionResult,
    TTSBackend,
    TTSCapability,
    VADBackend,
    VADSession,
)


# ═══════════════════════════════════════════════════════════════════════
# Mock ASR
# ═══════════════════════════════════════════════════════════════════════


class MockASRStream(ASRStream):
    """Accumulates fed audio; emits a deterministic transcript on finalize.

    Partials grow with each chunk so the engine's partial-polling path is
    exercised. ``transcript`` lets a test pin the final text.
    """

    def __init__(self, transcript: str = "hello world", language: Optional[str] = "English"):
        self._transcript = transcript
        self._language = language
        self._chunks = 0
        self._cancelled = False

    def accept_waveform(self, sample_rate: int, samples: "np.ndarray") -> None:
        if self._cancelled:
            return
        self._chunks += 1

    def get_partial(self) -> tuple[str, bool]:
        if self._cancelled or self._chunks == 0:
            return "", False
        # Reveal one word of the transcript per chunk fed, as a partial.
        words = self._transcript.split()
        n = min(self._chunks, len(words))
        return " ".join(words[:n]), False

    def finalize(self) -> tuple[str, Optional[str]]:
        if self._cancelled:
            return "", None
        return self._transcript, self._language

    def cancel_and_finalize(self) -> None:
        self._cancelled = True


class MockASR(ASRBackend):
    """Streaming + offline mock ASR. No env, no GPU."""

    def __init__(
        self,
        transcript: str = "hello world",
        language: Optional[str] = "English",
        sample_rate: int = 16000,
    ):
        self._transcript = transcript
        self._language = language
        self._sr = sample_rate
        self._ready = False

    @property
    def name(self) -> str:
        return "mock_asr"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING, ASRCapability.LANGUAGE_ID}

    @property
    def sample_rate(self) -> int:
        return self._sr

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._ready = True

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        return TranscriptionResult(text=self._transcript, language=self._language)

    def create_stream(self, language: str = "auto") -> ASRStream:
        return MockASRStream(transcript=self._transcript, language=self._language)


# ═══════════════════════════════════════════════════════════════════════
# Mock TTS
# ═══════════════════════════════════════════════════════════════════════


class MockTTS(TTSBackend):
    """Emits deterministic PCM: one chunk per sentence, length proportional
    to text length so callers can assert audio was produced."""

    def __init__(self, sample_rate: int = 16000, chunk_bytes_per_char: int = 4):
        self._sr = sample_rate
        self._cbpc = chunk_bytes_per_char
        self._ready = False

    @property
    def name(self) -> str:
        return "mock_tts"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return {TTSCapability.BASIC_TTS, TTSCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return self._sr

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._ready = True

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        pcm = self._pcm_for(text)
        return pcm, {"sample_rate": self._sr, "text": text}

    def generate_streaming(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        cancel_token: Optional[Any] = None,
        **kwargs,
    ) -> Iterator[bytes]:
        # Emit the PCM in a few sub-chunks so barge-in (cancel_token) can be
        # exercised mid-stream.
        pcm = self._pcm_for(text)
        step = max(2, len(pcm) // 3) & ~1  # even byte boundary (int16)
        for i in range(0, len(pcm), step or len(pcm)):
            if cancel_token is not None and cancel_token.is_set():
                return
            yield pcm[i : i + step]

    def _pcm_for(self, text: str) -> bytes:
        n_samples = max(1, len(text) * self._cbpc // 2)
        return (np.zeros(n_samples, dtype=np.int16)).tobytes()


# ═══════════════════════════════════════════════════════════════════════
# Mock VAD
# ═══════════════════════════════════════════════════════════════════════


class MockVADSession(VADSession):
    """Deterministic VAD: emits SPEECH_START on the first non-silent chunk,
    SPEECH_END once ``silence_chunks`` consecutive silent chunks are seen.

    "Silent" = all samples below ``threshold`` magnitude. This lets a test
    drive segmentation purely from the audio it feeds.
    """

    def __init__(self, silence_chunks: int = 2, threshold: float = 0.01):
        self._silence_needed = max(1, silence_chunks)
        self._threshold = threshold
        self._in_speech = False
        self._silence = 0

    def process(self, samples: "np.ndarray") -> Optional[str]:
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        loud = bool(np.any(np.abs(samples) >= self._threshold))
        if loud:
            self._silence = 0
            if not self._in_speech:
                self._in_speech = True
                return self.SPEECH_START
            return None
        # silent chunk
        if self._in_speech:
            self._silence += 1
            if self._silence >= self._silence_needed:
                self._in_speech = False
                self._silence = 0
                return self.SPEECH_END
        return None

    def reset(self) -> None:
        self._in_speech = False
        self._silence = 0


class MockVAD(VADBackend):
    def __init__(self, silence_chunks: int = 2, threshold: float = 0.01):
        self._silence_chunks = silence_chunks
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "mock_vad"

    def create_session(
        self, sample_rate: int = 16000, silence_ms: int = 400, **kwargs
    ) -> VADSession:
        return MockVADSession(
            silence_chunks=self._silence_chunks, threshold=self._threshold
        )


# ═══════════════════════════════════════════════════════════════════════
# Mock LLM
# ═══════════════════════════════════════════════════════════════════════


class MockLLM(LLMBackend):
    """Echoes a canned reply word-by-word as text deltas."""

    def __init__(self, reply: str = "Sure, here is your answer."):
        self._reply = reply

    async def stream(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        for word in self._reply.split():
            yield word + " "


__all__ = [
    "MockASR",
    "MockASRStream",
    "MockTTS",
    "MockVAD",
    "MockVADSession",
    "MockLLM",
]
