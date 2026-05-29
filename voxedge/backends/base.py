"""voxedge backend abstract base classes.

These mirror the **public** API of the production backend ABCs in
``app/core/`` and ``agent/openvoicestream_agent/llm/`` but are deliberately
DECOUPLED from env / profile loading: every constructor takes explicit
parameters only and nothing here reads ``os.environ`` or calls
``current_profile()``.

Source-of-truth mapping (read-only reference, NOT imported):
  * ASRBackend / ASRStream  ← app/core/asr_backend.py:36-147
  * TTSBackend              ← app/core/tts_backend.py:31-142
  * VADBackend / VADSession ← app/core/vad.py:90-109, 244-257
  * LLMBackend / LLMEvent   ← agent/openvoicestream_agent/llm/base.py

The LLM contract is reproduced here (not imported) on purpose: importing the
``agent`` package would drag in openai / httpx and break the "pure Python, no
heavy deps" guarantee. When the agent layer is itself packaged into voxedge
(spec §0.1 layer ②) these can be unified.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Iterator, Literal, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# ASR
# ═══════════════════════════════════════════════════════════════════════


class ASRCapability(str, Enum):
    """Mirrors app/core/asr_backend.py:21-26."""

    OFFLINE = "offline"
    STREAMING = "streaming"
    TIMESTAMPS = "timestamps"
    MULTI_LANGUAGE = "multi_language"
    LANGUAGE_ID = "language_id"


@dataclass
class TranscriptionResult:
    """Mirrors app/core/asr_backend.py:29-33."""

    text: str
    language: Optional[str] = None
    meta: Optional[dict] = None


class ASRStream(ABC):
    """A streaming ASR session that accumulates audio and produces text.

    Public API mirrors app/core/asr_backend.py:36-91.
    """

    @abstractmethod
    def accept_waveform(self, sample_rate: int, samples: "np.ndarray") -> None:
        """Feed audio samples (float32, [-1,1]) into the stream."""
        ...

    @abstractmethod
    def finalize(self) -> tuple[str, Optional[str]]:
        """Signal end-of-audio. Returns ``(final_text, detected_language)``.

        ``detected_language`` is the human-readable language name if the
        backend supports language ID, otherwise ``None``.
        """
        ...

    def get_partial(self) -> tuple[str, bool]:
        """Return ``(partial_text, is_endpoint)``. Default: no partials."""
        return "", False

    def prepare_finalize(self) -> None:
        """Optional: pre-encode remaining audio so finalize() runs decoder only."""

    def cancel_and_finalize(self) -> None:
        """Hard-cancel in-flight decode + skip residual tail encode (barge-in)."""

    def cancel(self) -> None:
        """Symmetric alias for :meth:`cancel_and_finalize`."""
        self.cancel_and_finalize()

    def close(self) -> None:
        """Release per-stream resources. Default no-op; safe to call twice."""


class ASRBackend(ABC):
    """Streaming/offline ASR backend.

    Public API mirrors app/core/asr_backend.py:94-147. The production
    factory ``create_asr_backend()`` reads ``current_profile()`` — voxedge
    drops that: callers construct the concrete backend with explicit params.
    """

    # Backends whose unload() truly releases GPU/NPU resources set True.
    supports_hot_reload: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> set[ASRCapability]:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def preload(self) -> None:
        """Load models and warm up. Called once before use."""
        ...

    @abstractmethod
    def transcribe(
        self, audio_bytes: bytes, language: str = "auto"
    ) -> TranscriptionResult:
        """One-shot offline transcription."""
        ...

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Create a streaming ASR session. Requires STREAMING capability."""
        raise NotImplementedError(f"{self.name} does not support streaming")

    def has_capability(self, cap: ASRCapability) -> bool:
        return cap in self.capabilities

    def unload(self) -> None:
        """Release GPU/NPU resources. Default no-op."""

    def concurrency_capability(self) -> Any:
        """Describe runtime concurrency properties (N, mode).

        In the production code this is a classmethod returning a typed
        ``ConcurrencyCapability`` read from a profile dict. In voxedge it is
        an instance method (no profile/env coupling) returning a plain dict
        ``{"max_concurrency": int, "mode": str}``. Default is conservative.
        """
        return {"max_concurrency": 1, "mode": "serialized"}


# ═══════════════════════════════════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════════════════════════════════


class TTSCapability(str, Enum):
    """Mirrors app/core/tts_backend.py:21-28."""

    BASIC_TTS = "basic_tts"
    VOICE_CLONE = "voice_clone"
    VOICE_CLONE_ICL = "voice_clone_icl"
    STREAMING = "streaming"
    MULTI_SPEAKER = "multi_speaker"
    MULTI_LANGUAGE = "multi_language"


class TTSBackend(ABC):
    """Text-to-speech backend.

    Public API mirrors app/core/tts_backend.py:31-142. The production
    ``model_id`` property reads ``OVS_TTS_MODEL_ID`` from env — voxedge takes
    ``model_id`` as an explicit constructor param on concrete backends
    instead (no env read here).
    """

    supports_hot_reload: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> set[TTSCapability]:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def preload(self) -> None:
        """Load models and warm up. Called once before use."""
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize text to WAV bytes. Returns ``(wav_bytes, metadata)``."""
        ...

    def generate_streaming(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        cancel_token: Optional[Any] = None,
        **kwargs,
    ) -> Iterator[bytes]:
        """Generator yielding raw PCM chunks. Requires STREAMING capability.

        Signature per spec §3: explicit keyword-only ``language`` /
        ``speaker`` / ``cancel_token`` instead of the production code's
        ad-hoc ``**kwargs`` so adapters have a stable contract. ``cancel_token``
        is any object with an ``is_set()`` method (e.g. ``threading.Event``);
        the backend should stop yielding once it returns True.
        """
        raise NotImplementedError(f"Backend '{self.name}' does not support streaming")

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize with voice cloning. Requires VOICE_CLONE capability."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support voice cloning"
        )

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        raise NotImplementedError(
            f"Backend '{self.name}' does not support speaker embedding extraction"
        )

    def has_capability(self, cap: TTSCapability) -> bool:
        return cap in self.capabilities

    def unload(self) -> None:
        """Release GPU/NPU resources. Default no-op."""

    def concurrency_capability(self) -> Any:
        """See :meth:`ASRBackend.concurrency_capability`."""
        return {"max_concurrency": 1, "mode": "serialized"}


# ═══════════════════════════════════════════════════════════════════════
# VAD
# ═══════════════════════════════════════════════════════════════════════


class VADSession(ABC):
    """Per-connection VAD state. Mirrors app/core/vad.py:90-109.

    Feed PCM chunks via :meth:`process`; get speech-start / speech-end
    transition events back.
    """

    SPEECH_START: Literal["speech_start"] = "speech_start"
    SPEECH_END: Literal["speech_end"] = "speech_end"
    NONE = None

    @abstractmethod
    def process(self, samples: "np.ndarray") -> Optional[str]:
        """Feed one chunk of int16/float32 PCM (16 kHz mono assumed).

        Returns ``"speech_start"`` (onset), ``"speech_end"`` (endpoint), or
        ``None`` (no transition this chunk).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state (e.g. after a forced finalize)."""
        ...


class VADBackend(ABC):
    """Factory for per-connection VAD sessions.

    voxedge folds the production module-level ``create_vad(...)`` factory
    (app/core/vad.py:244-257) into a backend object whose ``create_session``
    takes explicit ``sample_rate`` / ``silence_ms`` — no ``SILERO_VAD_ONNX_PATH``
    env read (the model path is a constructor arg on concrete backends).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def create_session(
        self, sample_rate: int = 16000, silence_ms: int = 400, **kwargs
    ) -> VADSession:
        ...


# ═══════════════════════════════════════════════════════════════════════
# LLM  (contract reproduced from agent/openvoicestream_agent/llm/base.py)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LLMEvent:
    """One unit of streaming output from an LLM backend.

    Equivalent to agent/openvoicestream_agent/llm/base.py:9-30. Reproduced
    here (not imported) so voxedge core stays free of the agent package's
    openai/httpx deps. When layer ② is packaged this can re-export the
    canonical definition.
    """

    kind: Literal["text", "tool_call_delta", "finish"]
    text: Optional[str] = None
    tool_call_index: Optional[int] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    finish_reason: Optional[str] = None


class LLMBackend(ABC):
    """Streaming LLM backend.

    Equivalent ABC to agent/openvoicestream_agent/llm/base.py:33-118:
      * ``stream_events`` (preferred) — yields :class:`LLMEvent`.
      * ``stream`` (back-compat) — yields plain text deltas.
      * ``warmup`` / ``aclose`` lifecycle hooks (default no-op, idempotent).
    """

    async def stream_events(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[LLMEvent]:
        """Yield :class:`LLMEvent`s. Default delegates to :meth:`stream`."""
        async for tok in self.stream(messages, **kw):
            if tok:
                yield LLMEvent(kind="text", text=tok)
        yield LLMEvent(kind="finish", finish_reason="stop")

    @abstractmethod
    async def stream(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        """Yield already-decoded text deltas."""
        if False:  # pragma: no cover
            yield ""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release held network/transport resources. Default no-op."""
        return None

    async def warmup(
        self,
        *,
        system_prompt: str = "",
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: bool = False,
        timeout_s: Optional[float] = 60.0,
    ) -> dict[str, Any]:
        """Optional fire-and-forget pre-flight warmup. Default returns {}."""
        return {}


__all__ = [
    "ASRCapability",
    "TranscriptionResult",
    "ASRStream",
    "ASRBackend",
    "TTSCapability",
    "TTSBackend",
    "VADSession",
    "VADBackend",
    "LLMEvent",
    "LLMBackend",
]
