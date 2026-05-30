"""voxedge Rockchip (RKNN/RKLLM) ASR/TTS adapters.

adapted from app/backends/rk/*.py + app/core/rk_*.py (2026-05-30), dedup after
registry switch.

Heavy runtime (``rkvoice_stream`` / the rknn runtime, aarch64-only) is imported
lazily inside backend ``__init__`` / methods, so importing this package
succeeds even on a machine without the optional ``voxedge[rk]`` extra or any RK
runtime (e.g. a Mac dev box).
"""

from __future__ import annotations

from .artifacts import RKArtifactConfig, RKArtifactError, ensure_rk_artifacts
from .asr import RKASRBackend, RKASRConfig
from .runtime import RKRuntimeConfig, RKRuntimeError, check_rk_runtime
from .tts import RKTTSBackend, RKTTSConfig

__all__ = [
    "RKASRBackend",
    "RKASRConfig",
    "RKTTSBackend",
    "RKTTSConfig",
    "RKRuntimeConfig",
    "RKRuntimeError",
    "check_rk_runtime",
    "RKArtifactConfig",
    "RKArtifactError",
    "ensure_rk_artifacts",
]
