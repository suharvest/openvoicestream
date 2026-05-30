"""voxedge sherpa-onnx CPU ASR/TTS adapters.

adapted from app/backends/cpu/sherpa*.py (2026-05-30), dedup after registry
switch.

Heavy runtime (``sherpa-onnx``) is imported lazily inside backend methods, so
importing this package succeeds even when the optional ``voxedge[sherpa]``
extra is not installed.
"""

from __future__ import annotations

from .asr import SherpaASRBackend, SherpaASRConfig, SherpaASRStream
from .tts import SherpaTTSBackend, SherpaTTSConfig

__all__ = [
    "SherpaASRBackend",
    "SherpaASRConfig",
    "SherpaASRStream",
    "SherpaTTSBackend",
    "SherpaTTSConfig",
]
