"""voxedge TRT-Edge-LLM ASR/TTS adapters.

adapted from app/backends/jetson/trt_edge_llm_*.py + app/core/worker_io.py
(2026-05-30), dedup after registry switch.

The Python side spawns a prebuilt C++ worker subprocess and talks JSON-line
IPC — it never imports ``tensorrt`` / CUDA directly. Combined with lazy
``onnxruntime`` / ``soundfile`` / ``webrtcvad`` imports inside methods, this
package imports cleanly on a machine with no GPU (Mac), even without the
optional ``voxedge[trt]`` extra installed.
"""

from __future__ import annotations

from .asr import TRTEdgeLLMASRBackend, TRTEdgeLLMASRConfig
from .ipc import audio_bytes_to_mel, run_binary, write_safetensors
from .tts import TRTEdgeLLMTTSBackend, TRTEdgeLLMTTSConfig
from .worker_io import WorkerExitError, WorkerIO

__all__ = [
    "TRTEdgeLLMASRBackend",
    "TRTEdgeLLMASRConfig",
    "TRTEdgeLLMTTSBackend",
    "TRTEdgeLLMTTSConfig",
    "WorkerIO",
    "WorkerExitError",
    "audio_bytes_to_mel",
    "run_binary",
    "write_safetensors",
]
