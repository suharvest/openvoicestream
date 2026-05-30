"""Common subprocess and I/O utilities for TRT-Edge-LLM backends — voxedge adapter.

adapted from app/backends/jetson/trt_edge_llm_ipc.py (2026-05-30), dedup after
registry switch.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * The production module reads ~30 ``os.environ.get(...)`` paths at module
    import time (binaries, plugin, engine/artifact dirs) and exposes them as
    module-level constants. voxedge has ZERO module-scope env reads. The path
    resolution moves into the ASR/TTS config dataclasses (``asr.py`` /
    ``tts.py``); this module keeps only the env-free, numpy-only helpers:
    ``run_binary`` / ``write_safetensors`` / ``audio_bytes_to_mel`` /
    ``write_temp_json`` / ``write_temp_wav``.
  * ``run_binary`` and ``_build_env`` take the plugin path / extra env as
    explicit arguments instead of reading ``EDGELLM_PLUGIN_PATH`` from the
    module-level constant.
  * ``audio_bytes_to_mel`` takes ``min_audio_frames`` as a parameter (was the
    module-level ``EDGE_LLM_ASR_MIN_AUDIO_FRAMES`` env read).

Nothing here imports ``tensorrt`` / CUDA: the TRT-Edge-LLM backends spawn a
C++ worker subprocess and talk JSON-line IPC, so the Python side is numpy-only
and imports cleanly on a machine with no GPU.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# GPU subprocess gate: serialise binary launches to avoid concurrent GPU init OOM
_gpu_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Binary runner
# ---------------------------------------------------------------------------


def _build_env(plugin_path: Optional[str] = None, extra_env: Optional[dict] = None) -> dict:
    """Return a copy of the *current* process env, optionally with the plugin
    path + caller-supplied overrides applied.

    The production helper read ``PLUGIN_PATH`` from a module constant. voxedge
    takes it as an explicit argument so this module holds no env state.
    """
    env = os.environ.copy()
    if plugin_path:
        env["EDGELLM_PLUGIN_PATH"] = plugin_path
    if extra_env:
        env.update(extra_env)
    return env


def run_binary(
    binary_path: str,
    args: list[str],
    timeout: int = 120,
    check: bool = True,
    plugin_path: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Run a TRT-Edge-LLM binary and return the CompletedProcess.

    Raises RuntimeError on non-zero exit (unless ``check=False``). The caller
    passes ``plugin_path`` (was the module-level ``PLUGIN_PATH`` constant) so
    this function reads no env of its own.
    """
    cmd = [binary_path] + args
    logger.info("Running (acquiring GPU lock): %s", " ".join(cmd[:4]))
    with _gpu_lock:
        logger.info("GPU lock acquired, launching: %s", os.path.basename(binary_path))
        try:
            result = subprocess.run(
                cmd,
                env=_build_env(plugin_path, extra_env),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"{os.path.basename(binary_path)} timed out after {timeout}s"
            ) from e

    if check and result.returncode != 0:
        stderr_snip = result.stderr[:1000] if result.stderr else "(empty)"
        raise RuntimeError(
            f"{os.path.basename(binary_path)} failed (exit={result.returncode}): "
            f"{stderr_snip}"
        )
    return result


# ---------------------------------------------------------------------------
# Safetensors writer (zero external deps)
# ---------------------------------------------------------------------------

_SAFETENSORS_DTYPE_MAP = {
    np.float16: "F16",
    np.float32: "F32",
    np.int32: "I32",
    np.int64: "I64",
    np.int8: "I8",
    np.uint8: "U8",
    np.bool_: "BOOL",
}


def write_safetensors(tensor: np.ndarray, name: str, path: str) -> None:
    """Write a single numpy array to a standard safetensors file.

    The tensor is written as-is (caller must cast to desired dtype first).
    """
    header = {
        name: {
            "dtype": _SAFETENSORS_DTYPE_MAP.get(
                tensor.dtype.type, str(tensor.dtype)
            ),
            "shape": list(tensor.shape),
            "data_offsets": [0, tensor.nbytes],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(tensor.tobytes())


# ---------------------------------------------------------------------------
# Mel-spectrogram computation (numpy-only, no librosa needed)
# ---------------------------------------------------------------------------

# Whisper / Qwen3 ASR constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10
# Production read EDGE_LLM_ASR_MIN_AUDIO_FRAMES from env at module scope;
# voxedge takes it as a parameter (default identical to the env default).
DEFAULT_MIN_AUDIO_FRAMES = 100


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank() -> np.ndarray:
    """Build Slaney-norm mel filterbank [n_mels, n_fft//2+1]."""
    n_freq = N_FFT // 2 + 1
    low_mel = _hz_to_mel(np.float64(FMIN))
    high_mel = _hz_to_mel(np.float64(FMAX))
    mel_points = np.linspace(low_mel, high_mel, N_MELS + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)

    bin = np.floor((n_freq - 1) * hz_points / FMAX).astype(np.int32)
    bin = np.clip(bin, 0, n_freq - 1)

    fb = np.zeros((N_MELS, n_freq), dtype=np.float64)
    for m in range(1, N_MELS + 1):
        left = int(bin[m - 1])
        center = int(bin[m])
        right = int(bin[m + 1])
        if left != center:
            for i in range(left, center):
                fb[m - 1, i] = (i - left) / (center - left)
        if center != right:
            for i in range(center, right):
                fb[m - 1, i] = (right - i) / (right - center)

    # Slaney norm: normalize each filter to unit area
    widths = hz_points[2:] - hz_points[:-2]
    fb *= (2.0 / widths)[:, np.newaxis]
    return fb.astype(np.float32)


# Build once at module level (cache) — pure compute, no env.
_MEL_FILTERBANK = _build_mel_filterbank()


def audio_bytes_to_mel(
    audio_bytes: bytes,
    target_sr: int = SAMPLE_RATE,
    min_audio_frames: int = DEFAULT_MIN_AUDIO_FRAMES,
) -> np.ndarray:
    """Convert WAV bytes to log-mel spectrogram.

    Returns float32 array of shape ``[1, 128, T]`` (batch, mel, time),
    using a narrow numpy port of Whisper/Qwen3-ASR feature extraction.

    Dynamic range clamp uses max-8dB (old working behavior) instead of the
    fixed -4dB Whisper clamp (which was producing wrong mel for Qwen3 ASR).
    """
    import wave

    # -- Read WAV --
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        new_len = int(round(len(audio) * target_sr / sr))
        src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)

    # -- Centered STFT with periodic Hann window --
    pad = N_FFT // 2
    if audio.shape[0] <= 1:
        audio = np.pad(audio, (0, 2 - audio.shape[0]), mode="constant")
    audio = np.pad(audio, (pad, pad), mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
    )

    # Drop final frame (Whisper convention)
    stft = np.fft.rfft(frames * window[np.newaxis, :], n=N_FFT, axis=1)
    magnitudes = np.abs(stft[:-1].T).astype(np.float32) ** 2.0

    mel_spec = _MEL_FILTERBANK @ magnitudes
    # -- Log compression (old working: max-8dB dynamic range) --
    log_spec = np.log10(np.maximum(mel_spec, MEL_FLOOR))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    if log_spec.shape[1] < min_audio_frames:
        pad_width = min_audio_frames - log_spec.shape[1]
        log_spec = np.pad(log_spec, ((0, 0), (0, pad_width)), mode="constant")

    return log_spec[np.newaxis, :, :].astype(np.float32)  # [1, 128, T]


# ---------------------------------------------------------------------------
# Temp-file helpers
# ---------------------------------------------------------------------------


def write_temp_json(data: dict, suffix: str = ".json") -> str:
    """Write a JSON dict to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False
    )
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def write_temp_wav(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Write audio bytes to a temporary WAV file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, delete=False
    )
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name
