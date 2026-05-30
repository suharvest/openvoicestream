"""TTS backend abstraction with capability discovery.

Backends expose different capabilities (voice clone, streaming, etc.).
Clients check capabilities before calling optional endpoints.
"""

from __future__ import annotations

import importlib
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple

from app.core.concurrency_capability import ConcurrencyCapability

logger = logging.getLogger(__name__)


class TTSCapability(str, Enum):
    """Capabilities a TTS backend may support."""
    BASIC_TTS = "basic_tts"
    VOICE_CLONE = "voice_clone"
    VOICE_CLONE_ICL = "voice_clone_icl"
    STREAMING = "streaming"
    MULTI_SPEAKER = "multi_speaker"
    MULTI_LANGUAGE = "multi_language"


class TTSBackend(ABC):
    """Base class for all TTS backends."""

    # PR5 / FIX_A: backends opt in by overriding this class attribute. Default
    # False keeps reload safe — a Jetson in-process TRT backend that holds
    # CUDA buffers, ORT sessions, or pybind engines cannot reliably release
    # GPU memory inside the same process (spike measured <6% RSS drop on
    # matcha/kokoro/qwen3). Only backends whose unload() truly frees the
    # underlying resources (subprocess workers, CPU-only models) should set
    # this to True. BackendManager.reload() refuses with HTTP 400 otherwise.
    supports_hot_reload: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. 'sherpa', 'qwen3_trt')."""
        ...

    @property
    def model_id(self) -> str:
        """Model-scope key for speaker tables.

        Reads ``OVS_TTS_MODEL_ID`` from the environment. Falls back to
        ``self.name`` (the backend identifier) with a warning so that
        production profiles always set this explicitly.
        """
        mid = os.environ.get("OVS_TTS_MODEL_ID")
        if mid:
            return mid
        logger.warning(
            "OVS_TTS_MODEL_ID not set; falling back to backend name %r. "
            "Speaker tables may be incorrect. Add tts_model_id to your profile.",
            self.name,
        )
        return self.name

    @property
    @abstractmethod
    def capabilities(self) -> set[TTSCapability]:
        """Set of supported capabilities."""
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
        """Load models and warm up. Called once at startup."""
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
        """Synthesize text to WAV bytes. Returns (wav_bytes, metadata)."""
        ...

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
        """Extract speaker embedding from WAV audio. Requires VOICE_CLONE capability."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support speaker embedding extraction"
        )

    def generate_streaming(self, text: str, **kwargs):
        """Generator yielding PCM chunks. Requires STREAMING capability."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support streaming"
        )

    def has_capability(self, cap: TTSCapability) -> bool:
        return cap in self.capabilities

    def unload(self) -> None:
        """Release GPU/NPU resources. See ASRBackend.unload() for semantics."""
        pass

    @classmethod
    def concurrency_capability(
        cls, profile: Optional[dict] = None
    ) -> ConcurrencyCapability:
        """Describe runtime concurrency properties.

        Classmethod (not instance property) so the scheduler can read the
        ceiling before ``preload()``. Default is conservative (N=1,
        serialized) — backends opt in by overriding. See
        ``docs/specs/concurrency-capability-framework.md`` Section 2.
        """
        return ConcurrencyCapability.default()


_TTS_REGISTRY: Dict[str, Tuple[str, str]] = {
    "jetson.trt_edge_llm": ("app.backends.jetson.trt_edge_llm_tts", "TRTEdgeLLMTTSBackend"),
    "jetson.matcha_trt":   ("voxedge.backends.jetson.matcha_trt",   "MatchaTRTBackend"),
    "jetson.kokoro_trt":   ("voxedge.backends.jetson.kokoro_trt",   "KokoroTRTBackend"),
    "jetson.qwen3_trt":    ("voxedge.backends.jetson.qwen3_trt",    "Qwen3TRTBackend"),
    "jetson.moss_tts_nano":("voxedge.backends.jetson.moss_tts_nano","MossTtsNanoBackend"),
    "cpu.sherpa":          ("voxedge.backends.sherpa.tts",          "SherpaTTSBackend"),
    "rk.tts":              ("voxedge.backends.rk.tts",              "RKTTSBackend"),
}


def _lazy_import(module_path: str, attr: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def create_tts_backend() -> TTSBackend:
    """Factory: instantiate the TTS backend declared by the loaded profile.

    Reads ``tts_backend`` from app.core.profile_loader.current_profile().
    Raises ValueError when no profile is loaded, the key is missing, or
    the value is not in the registry.
    """
    from app.core.profile_loader import current_profile
    spec = current_profile().get("tts_backend")
    if not spec:
        raise ValueError("Profile must declare 'tts_backend'")
    if spec not in _TTS_REGISTRY:
        raise ValueError(f"Unknown tts_backend: {spec!r}")
    module_path, cls_name = _TTS_REGISTRY[spec]
    logger.info("Creating TTS backend %s (%s.%s)", spec, module_path, cls_name)
    cls = _lazy_import(module_path, cls_name)
    # voxedge backends are env-free: build their config from env/profile in the
    # product layer (preserves voxedge's zero-env property). Other specs keep
    # their legacy os.environ-reading __init__.
    if spec == "jetson.matcha_trt":
        from app.core.voxedge_backend_config import build_matcha_tts_config
        return cls(config=build_matcha_tts_config(profile=current_profile()))
    if spec == "jetson.kokoro_trt":
        from app.core.voxedge_backend_config import build_kokoro_trt_config
        return cls(config=build_kokoro_trt_config(profile=current_profile()))
    if spec == "jetson.qwen3_trt":
        from app.core.voxedge_backend_config import build_qwen3_trt_config
        return cls(config=build_qwen3_trt_config(profile=current_profile()))
    if spec == "jetson.moss_tts_nano":
        from app.core.voxedge_backend_config import build_moss_tts_nano_config
        return cls(config=build_moss_tts_nano_config(profile=current_profile()))
    if spec == "cpu.sherpa":
        from app.core.voxedge_backend_config import build_sherpa_tts_config
        return cls(config=build_sherpa_tts_config(profile=current_profile()))
    if spec == "rk.tts":
        from app.core.voxedge_backend_config import build_rk_tts_config
        return cls(config=build_rk_tts_config(profile=current_profile()))
    return cls()
