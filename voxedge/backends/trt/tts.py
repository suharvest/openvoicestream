"""TTS backend via TRT-Edge-LLM C++ worker (qwen3_tts_worker / qwen3_tts_inference).

adapted from app/backends/jetson/trt_edge_llm_tts.py + app/core/worker_io.py
(2026-05-30), dedup after registry switch.

The Python side spawns a C++ worker subprocess and talks JSON-line IPC, so it
imports cleanly on a machine with no CUDA / tensorrt. Audio output is WAV via
the Code2Wav (vocoder) engine.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (TTSBackend / TTSCapability)
    and ``ConcurrencyCapability`` from ``voxedge.engine.concurrency_capability``.
  * ALL module-scope ``os.environ.get(...)`` reads (sampling defaults, artifact
    dirs, worker/segmentation flags, streaming-profile chunk frames, model_id
    /OVS_TTS_MODEL_ID) replaced by an explicit ``TRTEdgeLLMTTSConfig`` dataclass
    injected at construction time. voxedge has ZERO module-scope or hardcoded
    env reads.
  * ``WorkerIO`` imported from the sibling ``voxedge.backends.trt.worker_io``;
    ``resolve_speaker_kwargs`` from ``._util`` (env-free, registry-free).
  * The ``product_explicit_kv`` mode (in-process ``backends.qwen3_trt`` pybind
    backend) is retained but its dynamic ``importlib`` stays lazy (method
    scope) so this module imports without that optional product overlay; the
    speaker-encoder ONNX path uses lazy ``import onnxruntime`` / ``soundfile``.
  * ``concurrency_capability`` is an instance method (voxedge base contract)
    reading ``config.worker_concurrency`` instead of env/profile; the N>1
    ``--max_slots`` conditional (main fix b1cb1a5) is preserved.

Supports: BASIC_TTS, MULTI_LANGUAGE, STREAMING (+ VOICE_CLONE in worker mode).
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Optional

from voxedge.backends.base import TTSBackend, TTSCapability
from voxedge.engine.concurrency_capability import ConcurrencyCapability

from ._util import resolve_speaker_kwargs
from .ipc import run_binary
from .worker_io import WorkerExitError, WorkerIO

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var (first non-empty wins for alias groups) → field
#   EDGE_LLM_TTS_BIN                                  → tts_binary
#   EDGE_LLM_TTS_WORKER_BIN                           → worker_binary
#   EDGELLM_PLUGIN_PATH                              → plugin_path
#   EDGE_LLM_TTS_TALKER_DIR                           → talker_dir
#   EDGE_LLM_TTS_CP_DIR                              → code_predictor_dir
#   EDGE_LLM_TTS_TOKENIZER_DIR                       → tokenizer_dir
#   EDGE_LLM_TTS_CODE2WAV_DIR                        → code2wav_dir
#   QWEN3_SPEAKER_ENCODER                            → speaker_encoder
#   OVS_TTS_MODEL_ID                                → model_id ("trt_edgellm")
#   OVS_TTS_BACKEND/EDGE_LLM_TTS_BACKEND            → backend_mode ("edgellm_worker")
#   EDGE_LLM_TTS_WORKER                              → use_worker (True)
#   OVS_TTS_WORKER_CONCURRENCY                       → worker_concurrency (1) ← N>1 gates --max_slots
#   EDGE_LLM_QWEN3_PROFILE/OVS_QWEN3_PROFILE        → qwen3_runtime_profile ("highperf")
#   EDGE_LLM_TTS_PERF_PROFILE                        → perf_profile ("quality")
#   EDGE_LLM_TTS_STATEFUL_CODE2WAV                   → stateful_code2wav (None→profile-derived)
#   OVS_TTS_SEED                                     → seed (42)
#   OVS_TTS_TALKER_TEMPERATURE/TTS_TALKER_TEMPERATURE → talker_temperature (0.9)
#   OVS_TTS_TALKER_TOP_K/TTS_TALKER_TOP_K            → talker_top_k (50)
#   OVS_TTS_TOP_P/TTS_TOP_P                          → talker_top_p (1.0)
#   OVS_TTS_PREDICTOR_TEMPERATURE/...                → predictor_temperature (0.9)
#   OVS_TTS_PREDICTOR_TOP_K/...                      → predictor_top_k (50)
#   OVS_TTS_PREDICTOR_TOP_P/...                      → predictor_top_p (1.0)
#   TTS_MAX_AUDIO_LENGTH                              → max_audio_length (1024)
#   TTS_MIN_AUDIO_LENGTH                              → min_audio_length (30)
#   TTS_REPETITION_PENALTY                            → repetition_penalty (1.05)
#   TTS_CODEC_EOS_LOGIT_OFFSET                        → codec_eos_logit_offset (0.0)
#   EDGE_LLM_TTS_SEGMENT_TEXT                         → segment_text (True)
#   EDGE_LLM_TTS_SEGMENT_MAX_CHARS/..._CJK_...       → segment_max_chars_latin/cjk (120/48)
#   EDGE_LLM_TTS_SEGMENT_PAUSE_MS/HARD_...           → segment_pause_ms/hard (80/120)
#   EDGE_LLM_TTS_STREAMING_PROFILE                   → streaming_profile ("continuous_playback")
#   EDGE_LLM_TTS_FIRST_CHUNK_FRAMES etc.             → chunk-frame overrides (None→profile-derived)


_HIGHPERF_PROFILES = ("highperf", "perf", "performance", "v2v")
_FAST_PERF_PROFILES = ("fast", "v2v", "low_latency")


@dataclass
class TRTEdgeLLMTTSConfig:
    """Explicit construction-time config for :class:`TRTEdgeLLMTTSBackend`.

    Every field default is identical to the production env default. Artifact
    path fields default to empty strings (production resolved them from
    ``~/...`` artifact trees via env at import); a working backend MUST supply
    them. ``stateful_code2wav`` defaults to ``None`` → derived from
    ``qwen3_runtime_profile`` (mirrors the production env-default expression).
    """

    # Binaries / engines / plugin / artifact dirs.
    tts_binary: str = ""
    worker_binary: str = ""
    plugin_path: str = ""
    talker_dir: str = ""
    code_predictor_dir: str = ""
    tokenizer_dir: str = ""
    code2wav_dir: str = ""
    speaker_encoder: str = ""

    model_id: str = "trt_edgellm"
    backend_mode: str = "edgellm_worker"
    use_worker: bool = True
    worker_concurrency: int = 1  # N>1 gates --max_slots (main fix b1cb1a5)
    qwen3_runtime_profile: str = "highperf"
    perf_profile: str = "quality"
    stateful_code2wav: Optional[bool] = None
    seed: int = 42

    # Sampling.
    talker_temperature: float = 0.9
    talker_top_k: int = 50
    talker_top_p: float = 1.0
    predictor_temperature: float = 0.9
    predictor_top_k: int = 50
    predictor_top_p: float = 1.0
    max_audio_length: int = 1024
    min_audio_length: int = 30
    repetition_penalty: float = 1.05
    codec_eos_logit_offset: float = 0.0

    # Text segmentation.
    segment_text: bool = True
    segment_max_chars_latin: int = 120
    segment_max_chars_cjk: int = 48
    segment_pause_ms: int = 80
    segment_hard_pause_ms: int = 120

    # Streaming.
    streaming_profile: str = "continuous_playback"

    # product_explicit_kv mode paths (only used when backend_mode is that).
    product_model_base: str = "/home/harvest/voice_test/models/qwen3-tts"
    product_overlay_dir: str = "/home/harvest/voice_test/app_overlay"

    # Extra env passed through to the worker subprocess.
    extra_worker_env: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.worker_concurrency = max(1, int(self.worker_concurrency))
        self.qwen3_runtime_profile = (
            (self.qwen3_runtime_profile or "highperf").strip().lower().replace("-", "_")
        )
        self.backend_mode = (
            (self.backend_mode or "edgellm_worker").strip().lower().replace("-", "_")
        )
        self.perf_profile = (self.perf_profile or "quality").strip().lower()

    # -- derived helpers (no env reads) --

    def highperf_enabled(self) -> bool:
        return self.qwen3_runtime_profile in _HIGHPERF_PROFILES

    def stateful_code2wav_enabled(self) -> bool:
        if self.stateful_code2wav is not None:
            return bool(self.stateful_code2wav)
        return self.highperf_enabled()

    def fast_perf_profile(self) -> bool:
        return self.perf_profile in _FAST_PERF_PROFILES


class PoolSaturatedError(RuntimeError):
    """TTS worker rejected a request because every decoder slot is busy.

    The C++ ``qwen3_tts_worker`` launched with ``--max_slots N`` returns a
    ``status:4429`` payload when an N+1st concurrent request arrives. A plain
    RuntimeError (NOT routed into the destructive worker-restart path).
    """

    status: int = 4429

    def __init__(self, message: str, max_slots: Optional[int] = None) -> None:
        super().__init__(message)
        self.max_slots = max_slots


def _tts_pool_saturated_error(event: dict) -> Optional[PoolSaturatedError]:
    """Return a PoolSaturatedError if ``event`` is a worker saturation reject."""
    if not isinstance(event, dict):
        return None
    if event.get("event") != "error" and event.get("ok") is not False:
        return None
    msg = ""
    for key in ("error", "message", "reason", "detail"):
        v = event.get(key)
        if isinstance(v, str) and v:
            msg = v
            break
    low = msg.lower()
    if (
        event.get("status") == 4429
        or "pool_saturated" in low
        or "too_many_tts" in low
        or "too many tts" in low
    ):
        return PoolSaturatedError(msg or str(event), max_slots=event.get("max_slots"))
    return None


_SPEAKER_ENC_SESSION = None  # cached ort session — onnx load is non-trivial


def _qwen3_speaker_embed_inproc(audio_wav_bytes: bytes, encoder_path: str) -> bytes:
    """In-process 1024-d speaker embedding from a reference WAV.

    Pure numpy mel + ONNX Runtime. ``onnxruntime`` / ``soundfile`` are imported
    lazily so this module imports without those optional deps.
    """
    global _SPEAKER_ENC_SESSION
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf

    data, sr_in = sf.read(io.BytesIO(audio_wav_bytes), always_2d=False, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)
    if sr_in != 24000:
        n_in = len(data)
        n_out = int(round(n_in * 24000 / sr_in))
        spec = np.fft.rfft(data)
        n_spec_out = n_out // 2 + 1
        if n_spec_out < len(spec):
            spec = spec[:n_spec_out]
        else:
            spec = np.concatenate(
                [spec, np.zeros(n_spec_out - len(spec), dtype=spec.dtype)]
            )
        data = (np.fft.irfft(spec, n=n_out) * (n_out / n_in)).astype(np.float32)
    sr = 24000

    N_FFT, HOP, WIN, N_MEL = 1024, 256, 1024, 128
    FMIN, FMAX = 0.0, 12000.0

    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _slaney_mel_filterbank() -> "np.ndarray":
        mel_pts = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MEL + 2)
        hz_pts = _mel_to_hz(mel_pts)
        bin_freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
        fb = np.zeros((N_MEL, N_FFT // 2 + 1), dtype=np.float32)
        for i in range(N_MEL):
            lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
            lt = (bin_freqs >= lo) & (bin_freqs <= mid)
            rt = (bin_freqs >= mid) & (bin_freqs <= hi)
            fb[i, lt] = (bin_freqs[lt] - lo) / (mid - lo + 1e-12)
            fb[i, rt] = (hi - bin_freqs[rt]) / (hi - mid + 1e-12)
        enorm = 2.0 / (hz_pts[2:] - hz_pts[:-2])
        fb *= enorm[:, None]
        return fb

    mel_basis = _slaney_mel_filterbank()
    hann = np.hanning(WIN).astype(np.float32)
    pad = (N_FFT - HOP) // 2
    y = np.pad(data, pad, mode="reflect")
    num_frames = 1 + (len(y) - WIN) // HOP
    if num_frames < 1:
        raise ValueError(f"reference audio too short: {len(data)/sr:.2f}s (need >0.5s)")
    frames = np.lib.stride_tricks.sliding_window_view(y, WIN)[::HOP][:num_frames] * hann
    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9).astype(np.float32)
    mel_spec = mag @ mel_basis.T
    mel_spec = np.log(np.clip(mel_spec, 1e-5, None)).astype(np.float32)

    if _SPEAKER_ENC_SESSION is None or _SPEAKER_ENC_SESSION[0] != encoder_path:
        sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        _SPEAKER_ENC_SESSION = (encoder_path, sess)
    sess = _SPEAKER_ENC_SESSION[1]
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: mel_spec[None, ...]})
    emb = out[0].squeeze().astype(np.float32)
    if emb.shape != (1024,):
        raise RuntimeError(f"unexpected speaker embedding shape: {emb.shape}")
    return emb.tobytes()


def _code2wav_engine_path(code2wav_dir: str) -> str:
    """Return the Code2Wav engine path used by current Qwen3 artifact sets."""
    candidates = [
        os.path.join(code2wav_dir, "code2wav.engine"),
        os.path.join(code2wav_dir, "code2wav_stateful.engine"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _detect_language(text: str) -> str:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return "chinese"
        if 0x3040 <= cp <= 0x30FF:
            return "japanese"
        if 0xAC00 <= cp <= 0xD7AF:
            return "korean"
    return "english"


def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            return True
    return False


def _split_tts_text(
    text: str,
    max_chars: Optional[int] = None,
    *,
    max_chars_latin: int = 120,
    max_chars_cjk: int = 48,
) -> list[str]:
    """Split long TTS text into independently stable synthesis requests."""
    normalized = " ".join(text.split()) if not _contains_cjk(text) else text.strip()
    if not normalized:
        return []

    if max_chars is None:
        max_chars = max_chars_cjk if _contains_cjk(normalized) else max_chars_latin
    max_chars = max(8, int(max_chars))

    hard_breaks = set("。！？!?；;\n")
    soft_breaks = set("，,、：:")
    segments: list[str] = []
    current: list[str] = []
    is_cjk = _contains_cjk(normalized)
    max_overrun = 0 if is_cjk else max(2, min(8, max_chars // 2))
    abbreviations = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
        "etc.", "e.g.", "i.e.",
    }

    def is_nonterminal_period(buffer: str, next_ch: str) -> bool:
        stripped = buffer.strip().lower()
        if next_ch.isdigit() and len(stripped) >= 2 and stripped[-2].isdigit():
            return True
        return any(stripped.endswith(abbrev) for abbrev in abbreviations)

    def flush() -> None:
        part = "".join(current).strip()
        current.clear()
        if part:
            segments.append(part)

    for idx, ch in enumerate(normalized):
        next_ch = normalized[idx + 1] if idx + 1 < len(normalized) else ""
        current.append(ch)
        if ch in hard_breaks:
            if not is_cjk and ch == "." and is_nonterminal_period("".join(current), next_ch):
                continue
            flush()
            continue
        current_text = "".join(current).strip()
        if len(current_text) >= max_chars:
            text_so_far = "".join(current)
            cut = max(text_so_far.rfind(p) for p in soft_breaks)
            if cut >= max_chars // 3:
                head = text_so_far[: cut + 1].strip()
                tail = text_so_far[cut + 1:].lstrip()
                current.clear()
                if head:
                    segments.append(head)
                if tail:
                    current.extend(tail)
            else:
                if is_cjk and len(current_text) < max_chars + max_overrun:
                    continue
                flush()
    flush()

    if not is_cjk:
        packed: list[str] = []
        for part in segments:
            if len(part) <= max_chars:
                packed.append(part)
                continue
            words = part.split(" ")
            buf: list[str] = []
            for word in words:
                candidate = " ".join(buf + [word]).strip()
                if buf and len(candidate) > max_chars:
                    packed.append(" ".join(buf))
                    buf = [word]
                else:
                    buf.append(word)
            if buf:
                packed.append(" ".join(buf))
        segments = packed

    merged: list[str] = []
    min_chars = max(4, min(12, max_chars // 3))
    for part in segments:
        if merged and all(ch in hard_breaks or ch in soft_breaks for ch in part):
            merged[-1] = f"{merged[-1]}{part}"
            continue
        sep = "" if merged and _contains_cjk(part + merged[-1]) else " "
        if merged and len(part) < min_chars and len(merged[-1]) + len(sep) + len(part) <= max_chars:
            merged[-1] = f"{merged[-1]}{sep}{part}"
        else:
            merged.append(part)
    return merged


def _segment_pause_ms(segment: str, pause_ms: int = 80, hard_pause_ms: int = 120) -> int:
    """Silence to insert *after* a synthesized segment when concatenating."""
    if not segment:
        return 0
    stripped = segment.rstrip()
    if stripped.endswith(("。", "！", "？", "!", "?", ";", "；")):
        return max(0, hard_pause_ms)
    if stripped.endswith(("，", ",", "、", "：", ":")):
        return max(0, pause_ms)
    return 0


def _concat_wav_bytes(parts: list[bytes], pauses_ms: Optional[list[int]] = None) -> bytes:
    non_empty = [part for part in parts if part]
    if not non_empty:
        return b""
    if len(non_empty) == 1:
        return non_empty[0]

    params = None
    frames: list[bytes] = []
    for idx, part in enumerate(non_empty):
        with wave.open(io.BytesIO(part), "rb") as reader:
            current = reader.getparams()
            comparable = (current.nchannels, current.sampwidth, current.framerate, current.comptype, current.compname)
            if params is None:
                params = comparable
            elif comparable != params:
                raise RuntimeError(f"Cannot concatenate WAV segments with different formats: {comparable} != {params}")
            frames.append(reader.readframes(reader.getnframes()))
            if pauses_ms and idx < len(non_empty) - 1:
                pause_samples = int(current.framerate * max(0, pauses_ms[idx]) / 1000)
                if pause_samples > 0:
                    frames.append(b"\x00" * pause_samples * current.nchannels * current.sampwidth)

    nchannels, sampwidth, framerate, comptype, compname = params
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(nchannels)
        writer.setsampwidth(sampwidth)
        writer.setframerate(framerate)
        writer.setcomptype(comptype, compname)
        for frame_bytes in frames:
            writer.writeframes(frame_bytes)
    return out.getvalue()


def _wav_duration_and_samples(wav_bytes: bytes) -> tuple[float, int]:
    if not wav_bytes:
        return 0.0, 0
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        samples = reader.getnframes()
        rate = reader.getframerate()
    return (samples / rate if rate > 0 else 0.0), samples


def _event_request_id(event: dict) -> str | None:
    rid = event.get("request_id")
    if rid:
        return rid
    rid = event.get("id")
    return rid if rid else None


def _pcm16_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class TRTEdgeLLMTTSBackend(TTSBackend):
    """TTS via TRT-Edge-LLM qwen3_tts_worker subprocess."""

    def concurrency_capability(self) -> ConcurrencyCapability:
        """Declare the TTS slot-pool ceiling.

        WorkerIO multiplexes N in-flight requests with a single subprocess.
        Reads ``config.worker_concurrency`` (was env
        ``OVS_TTS_WORKER_CONCURRENCY`` / profile ``tts_worker_concurrency``).
        N>1 enables ``supports_parallel``.
        """
        n = max(1, int(self._config.worker_concurrency))
        return ConcurrencyCapability(
            supports_parallel=n > 1,
            max_concurrent=n,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    @property
    def supports_hot_reload(self) -> bool:  # type: ignore[override]
        mode = getattr(self, "_resolved_mode", None) or self._backend_mode()
        if mode in ("product_explicit_kv", "explicit_kv"):
            return False
        return True

    def __init__(self, config: Optional[TRTEdgeLLMTTSConfig] = None):
        self._config = config or TRTEdgeLLMTTSConfig()
        self._ready = False
        self._product_backend = None
        self._resolved_mode: Optional[str] = None
        self._worker: Optional[subprocess.Popen] = None
        self._worker_lock = threading.Lock()
        self._worker_io: Optional[WorkerIO] = None
        self._worker_concurrency: int = max(1, int(self._config.worker_concurrency))
        self._worker_ready_meta: dict = {}
        self._worker_stderr_tail = deque(maxlen=80)
        # Artifact paths captured from the injected config (was resolved from
        # the current env at __init__ in the production copy).
        self._talker_dir = self._config.talker_dir
        self._code_predictor_dir = self._config.code_predictor_dir
        self._tokenizer_dir = self._config.tokenizer_dir
        self._speaker_encoder = self._config.speaker_encoder
        self._code2wav_dir = self._config.code2wav_dir
        self._worker_binary = self._config.worker_binary
        self._qwen3_runtime_profile = self._config.qwen3_runtime_profile

    # -- TTSBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def model_id(self) -> str:
        """Model-scope key — injected via config (was ``OVS_TTS_MODEL_ID``)."""
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE, TTSCapability.STREAMING}
        if self._product_backend is not None:
            caps |= self._product_backend.capabilities
        else:
            caps.add(TTSCapability.VOICE_CLONE)
        return caps

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def _backend_mode(self) -> str:
        return self._config.backend_mode

    def _load_product_explicit_kv_backend(self):
        model_base = self._config.product_model_base
        overlay = self._config.product_overlay_dir
        overlay_backends = os.path.join(overlay, "backends")
        for path in (overlay, overlay_backends):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        os.environ.setdefault("QWEN3_MODEL_BASE", model_base)
        os.environ.setdefault("QWEN3_MODEL_DIR", os.path.join(model_base, "onnx"))
        os.environ.setdefault("QWEN3_SHERPA_DIR", os.path.join(model_base, "onnx"))
        os.environ.setdefault("QWEN3_TALKER_ENGINE", os.path.join(model_base, "engines", "talker_decode_bf16.engine"))
        os.environ.setdefault("QWEN3_CP_ENGINE", os.path.join(model_base, "engines", "cp_bf16.engine"))
        os.environ.setdefault("TTS_TALKER_CUDA_GRAPH", "0")

        module = importlib.import_module("backends.qwen3_trt")
        module = importlib.reload(module)
        backend = module.Qwen3TRTBackend()
        logger.info(
            "Using product_explicit_kv TTS backend (model_base=%s, talker=%s)",
            model_base,
            os.environ.get("QWEN3_TALKER_ENGINE"),
        )
        backend.preload()
        return backend

    def preload(self) -> None:
        """Verify all required files exist."""
        mode = self._backend_mode()
        self._resolved_mode = mode
        if mode in ("product_explicit_kv", "explicit_kv"):
            self._product_backend = self._load_product_explicit_kv_backend()
            self._ready = True
            return
        if mode not in ("edgellm", "edgellm_worker", "official"):
            raise ValueError(
                f"Unsupported backend_mode {mode!r}; expected edgellm_worker "
                "or product_explicit_kv"
            )

        required = [
            (self._worker_binary if self._use_worker() else self._config.tts_binary, "TTS binary"),
            (self._config.plugin_path, "TRT-Edge-LLM plugin"),
            (os.path.join(self._talker_dir, "config.json"), "talker config"),
            (os.path.join(self._tokenizer_dir, "tokenizer.json"), "tokenizer"),
            (os.path.join(self._talker_dir, "llm.engine"), "talker engine"),
        ]
        missing = []
        for path, label in required:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")
        if missing:
            raise FileNotFoundError(
                "TTS preload failed — missing:\n  " + "\n  ".join(missing)
            )

        c2w_path = _code2wav_engine_path(self._code2wav_dir)
        if os.path.exists(c2w_path):
            logger.info("Code2Wav engine found at %s", c2w_path)
        else:
            logger.warning(
                "Code2Wav not found at %s — will output RVQ codes only", c2w_path
            )

        logger.info(
            "TTS backend preload OK (profile=%s binary=%s talker=%s)",
            self._qwen3_runtime_profile,
            self._worker_binary if self._use_worker() else self._config.tts_binary,
            self._talker_dir,
        )
        if self._use_worker():
            self._ensure_worker()
        self._ready = True

    def unload(self) -> None:
        """Kill the resident worker subprocess so GPU memory is fully released."""
        if (
            not self._ready
            and self._worker is None
            and self._product_backend is None
        ):
            return
        try:
            with self._worker_lock:
                old = self._worker
                self._worker = None
                self._worker_io = None
                if old is not None:
                    try:
                        old.terminate()
                        old.wait(timeout=5)
                    except Exception:
                        try:
                            old.kill()
                        except Exception:
                            pass
                self._worker_stderr_tail.clear()
                self._worker_ready_meta = {}
        except Exception:
            logger.exception("TRTEdgeLLMTTSBackend.unload failed; continuing")
        finally:
            if self._product_backend is not None:
                try:
                    self._product_backend.unload()
                except Exception:
                    logger.exception("product_backend.unload failed; continuing")
                self._product_backend = None
            self._resolved_mode = None
            self._ready = False

    def _use_worker(self) -> bool:
        return bool(self._config.use_worker)

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        env.update(self._config.extra_worker_env)
        env["EDGELLM_PLUGIN_PATH"] = self._config.plugin_path
        env.setdefault("EDGE_LLM_TTS_CUDA_GRAPH", "0")
        env.setdefault("EDGE_LLM_TTS_LAZY_CODE2WAV", "0")
        env.setdefault(
            "EDGE_LLM_TTS_STATEFUL_CODE2WAV",
            "1" if self._config.highperf_enabled() else "0",
        )
        if self._config.stateful_code2wav_enabled():
            env.setdefault("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", "0")
            env.setdefault("QWEN3_TTS_CP_DECODE_CUDA_GRAPH", "1")
            env.setdefault("QWEN3_TTS_ACTIVE_CP_GROUPS", "13")
        else:
            env.setdefault("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", "3")
        return env

    def _worker_stderr_snip(self) -> str:
        return "".join(self._worker_stderr_tail)[-2000:] or "(empty)"

    def _drain_worker_stderr(self) -> None:
        worker = self._worker
        if worker is None or worker.stderr is None:
            return
        for line in worker.stderr:
            self._worker_stderr_tail.append(line)
            if "[JV_MEM]" in line:
                logger.info("TTS worker: %s", line.rstrip())
            else:
                logger.debug("TTS worker stderr: %s", line.rstrip())

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        self._worker_concurrency = max(1, int(self._config.worker_concurrency))
        cmd = [
            self._worker_binary,
            "--talkerEngineDir", self._talker_dir,
            "--codePredictorEngineDir", self._code_predictor_dir,
            "--tokenizerDir", self._tokenizer_dir,
            "--code2wavEngineDir", self._code2wav_dir,
        ]
        # Only emit --max_slots when N>1 (main fix b1cb1a5): at N=1 omit it for
        # legacy single-slot byte-equivalent behavior + back-compat with older
        # worker binaries that reject the unknown flag.
        if self._worker_concurrency and self._worker_concurrency > 1:
            cmd += ["--max_slots", str(self._worker_concurrency)]
        self._worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        threading.Thread(target=self._drain_worker_stderr, name="tts-worker-stderr", daemon=True).start()
        assert self._worker.stdout is not None
        ready_line = self._worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"TTS worker failed to start: {self._worker_stderr_snip()}")
        ready = json.loads(ready_line)
        if ready.get("event") != "ready":
            raise RuntimeError(f"TTS worker did not become ready: {ready}")
        self._worker_ready_meta = ready
        self._worker_concurrency = max(1, int(self._config.worker_concurrency))
        self._worker_io = WorkerIO(self._worker, self._worker_concurrency)

    def _restart_worker_locked(self, reason: str) -> None:
        """Restart the resident TTS worker. Called from inside ``_worker_lock``."""
        logger.warning("Restarting TTS worker: %s", reason)
        old = self._worker
        self._worker = None
        self._worker_io = None
        if old is not None:
            try:
                old.terminate()
                old.wait(timeout=5)
            except Exception:
                try:
                    old.kill()
                except Exception:
                    pass
        self._worker_stderr_tail.clear()
        self._ensure_worker()

    def _synthesize_worker(self, text: str, language: Optional[str], **kwargs) -> tuple[bytes, dict]:
        if self._config.stateful_code2wav_enabled():
            return self._synthesize_worker_via_stream(text, language=language, **kwargs)
        req_id = uuid.uuid4().hex
        with tempfile.NamedTemporaryFile(prefix="trt_edgellm_tts_", suffix=".wav", delete=False) as f:
            output_file = f.name
        request = {
            "id": req_id,
            "text": text,
            "output_file": output_file,
            "language": language or _detect_language(text),
            "talker_temperature": self._config.talker_temperature,
            "talker_top_k": self._config.talker_top_k,
            "talker_top_p": self._config.talker_top_p,
            "repetition_penalty": self._config.repetition_penalty,
            "codec_eos_logit_offset": self._config.codec_eos_logit_offset,
            "predictor_temperature": self._config.predictor_temperature,
            "predictor_top_k": self._config.predictor_top_k,
            "predictor_top_p": self._config.predictor_top_p,
            "max_audio_length": kwargs.get("max_audio_length", self._config.max_audio_length),
            "min_audio_length": kwargs.get("min_audio_length", self._config.min_audio_length),
            "seed": int(kwargs.get("seed", self._config.seed)),
        }
        voice_kwargs = self._resolve_voice_kwargs(kwargs)
        speaker_embedding = voice_kwargs.get("speaker_embedding")
        if speaker_embedding:
            request["speaker_embedding_b64"] = base64.b64encode(speaker_embedding).decode("ascii")
        else:
            self._add_speaker_request_fields(request, voice_kwargs)
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker_io is not None
        t0 = time.time()
        response = None
        try:
            for event in self._worker_io.request(request):
                response = event
        except WorkerExitError as exc:
            self._worker = None
            self._worker_io = None
            raise RuntimeError(f"TTS worker exited before response: {self._worker_stderr_snip()}") from exc
        elapsed = time.time() - t0
        if response is None:
            self._worker = None
            self._worker_io = None
            raise RuntimeError(f"TTS worker returned no events: {self._worker_stderr_snip()}")
        saturated = _tts_pool_saturated_error(response)
        if saturated is not None:
            raise saturated
        if not response.get("ok"):
            raise RuntimeError(f"TTS worker failed: {response}")
        with open(response["output_file"], "rb") as f:
            wav_bytes = f.read()
        try:
            os.unlink(response["output_file"])
        except OSError:
            pass
        audio_s = float(response.get("audio_s", 0.0))
        meta = {
            "inference_time_s": round(elapsed, 3),
            "sample_rate": int(response.get("sample_rate", 24000)),
            "duration_s": audio_s,
            "samples": int(response.get("samples", 0)),
            "rtf": round(float(response.get("rtf", 0.0)), 3),
            "generation_ms": round(float(response.get("generation_ms", 0.0)), 1),
            "code2wav_ms": round(float(response.get("code2wav_ms", 0.0)), 1),
            "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        }
        return wav_bytes, meta

    def _synthesize_worker_via_stream(
        self, text: str, language: Optional[str] = None, **kwargs
    ) -> tuple[bytes, dict]:
        """Aggregate streaming PCM chunks into a single WAV."""
        t0 = time.time()
        done_meta: dict = {}
        stream_kwargs = dict(kwargs)
        stream_kwargs["segment_text"] = False
        stream_kwargs["language"] = language
        pcm = bytearray()
        for chunk in self._generate_streaming_single(text, meta_out=done_meta, **stream_kwargs):
            pcm.extend(chunk)
        elapsed = time.time() - t0
        sample_rate = int(done_meta.get("sample_rate", 24000))
        wav_bytes = _pcm16_to_wav(bytes(pcm), sample_rate=sample_rate)
        meta = {
            "inference_time_s": round(elapsed, 3),
            "sample_rate": sample_rate,
            "duration_s": float(done_meta.get("audio_s", 0.0)),
            "samples": int(done_meta.get("samples", len(pcm) // 2)),
            "rtf": round(float(done_meta.get("rtf", 0.0)), 3),
            "generation_ms": round(float(done_meta.get("generation_ms", 0.0)), 1),
            "code2wav_ms": round(float(done_meta.get("code2wav_ms", 0.0)), 1),
            "first_chunk_ms": round(float(done_meta.get("first_chunk_ms", 0.0)), 1),
            "chunk_count": int(done_meta.get("chunk_count", 0)),
            "stateful_code2wav": bool(done_meta.get("stateful_code2wav", True)),
            "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        }
        return wav_bytes, meta

    def generate_streaming(self, text: str, **kwargs):
        """Yield raw PCM int16 chunks from the resident EdgeLLM TTS worker."""
        if self._product_backend is not None:
            yield from self._product_backend.generate_streaming(text, **kwargs)
            return

        if self._config.segment_text and kwargs.get("segment_text", True):
            segments = _split_tts_text(
                text,
                kwargs.get("segment_max_chars"),
                max_chars_latin=self._config.segment_max_chars_latin,
                max_chars_cjk=self._config.segment_max_chars_cjk,
            )
            if len(segments) > 1:
                segment_kwargs = dict(kwargs)
                segment_kwargs["segment_text"] = False
                segment_kwargs.setdefault("seed", self._config.seed)
                for segment in segments:
                    yield from self.generate_streaming(segment, **segment_kwargs)
                return

        yield from self._generate_streaming_single(text, **kwargs)

    def _generate_streaming_single(self, text: str, meta_out: Optional[dict] = None, **kwargs):
        """Yield raw PCM int16 chunks for one already-bounded TTS request."""
        retry_empty = bool(kwargs.pop("_retry_empty", True))
        req_id = uuid.uuid4().hex
        streaming_profile = str(
            kwargs.get("streaming_profile", self._config.streaming_profile)
        ).lower()
        if streaming_profile in ("v2v", "voice_to_voice", "eos_to_first_audio"):
            default_first_chunk_frames = 1
            default_chunk_frames = 97
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 97
            default_adaptive_chunks = False
        elif streaming_profile in ("instant_feedback", "low_latency"):
            default_first_chunk_frames = 1
            default_chunk_frames = 25
            default_chunk_growth_frames = 50
            default_max_chunk_frames = 150
            default_adaptive_chunks = True
        elif streaming_profile in ("playback", "smooth", "buffered"):
            default_first_chunk_frames = 20
            default_chunk_frames = 20
            default_chunk_growth_frames = 30
            default_max_chunk_frames = 120
            default_adaptive_chunks = True
        else:
            default_first_chunk_frames = 50
            default_chunk_frames = 97
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 97
            default_adaptive_chunks = False
        if self._config.stateful_code2wav_enabled():
            if self._config.fast_perf_profile():
                default_first_chunk_frames = 4
            elif self._config.perf_profile == "balanced":
                default_first_chunk_frames = 6
            else:
                default_first_chunk_frames = 7
            default_chunk_frames = 10
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 10
            default_adaptive_chunks = False
        request = {
            "id": req_id,
            "text": text,
            "output_file": f"/tmp/trt_edgellm_tts_stream_{req_id}.wav",
            "language": kwargs.get("language") or _detect_language(text),
            "talker_temperature": self._config.talker_temperature,
            "talker_top_k": self._config.talker_top_k,
            "talker_top_p": self._config.talker_top_p,
            "repetition_penalty": self._config.repetition_penalty,
            "codec_eos_logit_offset": self._config.codec_eos_logit_offset,
            "predictor_temperature": self._config.predictor_temperature,
            "predictor_top_k": self._config.predictor_top_k,
            "predictor_top_p": self._config.predictor_top_p,
            "max_audio_length": kwargs.get("max_audio_length", self._config.max_audio_length),
            "min_audio_length": kwargs.get("min_audio_length", self._config.min_audio_length),
            "seed": int(kwargs.get("seed", self._config.seed)),
            "stream": True,
            "stream_only": True,
            "first_chunk_frames": kwargs.get("first_chunk_frames", default_first_chunk_frames),
            "chunk_frames": kwargs.get("chunk_frames", default_chunk_frames),
            "adaptive_chunks": kwargs.get("adaptive_chunks", default_adaptive_chunks),
            "max_chunk_frames": kwargs.get("max_chunk_frames", default_max_chunk_frames),
            "chunk_growth_frames": kwargs.get("chunk_growth_frames", default_chunk_growth_frames),
            "chunk_format": "pcm_s16le",
            "chunk_transport": "base64",
        }
        voice_kwargs = self._resolve_voice_kwargs(kwargs)
        speaker_embedding = voice_kwargs.get("speaker_embedding")
        if speaker_embedding:
            request["speaker_embedding_b64"] = base64.b64encode(speaker_embedding).decode("ascii")
        else:
            self._add_speaker_request_fields(request, voice_kwargs)

        retry_after_empty = False
        emitted_chunks = 0
        done_event: dict | None = None
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker_io is not None
            worker_io = self._worker_io
        try:
            for event in worker_io.request(request):
                event_rid = _event_request_id(event)
                if event_rid is not None and event_rid != req_id and event_rid != "__worker__":
                    logger.debug(
                        "TTS worker event id mismatch: expected=%s got=%s event=%s",
                        req_id, event_rid, event.get("event"),
                    )
                if event.get("event") == "cancelled":
                    logger.info(
                        "TTS worker acknowledged cancel for %s (reason=%s)",
                        req_id, event.get("reason"),
                    )
                    return
                saturated = _tts_pool_saturated_error(event)
                if saturated is not None:
                    raise saturated
                if not event.get("ok"):
                    raise RuntimeError(f"TTS streaming worker failed: {event}")
                if event.get("event") == "chunk":
                    if event.get("chunk_transport") == "base64":
                        emitted_chunks += 1
                        yield base64.b64decode(event.get("audio_b64", ""))
                    elif event.get("chunk_file"):
                        with open(event["chunk_file"], "rb") as f:
                            payload = f.read()
                        try:
                            os.unlink(event["chunk_file"])
                        except OSError:
                            pass
                        if event.get("chunk_format") == "wav" and len(payload) > 44:
                            payload = payload[44:]
                        emitted_chunks += 1
                        yield payload
                elif event.get("event") == "done":
                    done_event = event
                    if meta_out is not None and isinstance(meta_out, dict):
                        meta_out.update(event)
                    if (
                        retry_empty
                        and self._config.stateful_code2wav_enabled()
                        and emitted_chunks == 0
                    ):
                        retry_after_empty = True
                        with self._worker_lock:
                            self._restart_worker_locked(
                                f"stateful stream returned 0 chunks for request {req_id}"
                            )
                    break
        except GeneratorExit:
            logger.info(
                "generator exit during TTS stream; cancelling worker for %s", req_id
            )
            try:
                worker_io.cancel(req_id)
            except Exception:
                logger.debug("worker_io.cancel() failed during GeneratorExit", exc_info=True)
            raise
        except WorkerExitError as exc:
            self._worker = None
            self._worker_io = None
            raise RuntimeError(
                f"TTS worker exited during stream: {self._worker_stderr_snip()}"
            ) from exc
        if retry_after_empty:
            logger.warning(
                "Retrying TTS stream after empty stateful result (done=%s stderr_tail=%s)",
                done_event, self._worker_stderr_snip(),
            )
            yield from self._generate_streaming_single(
                text, meta_out=meta_out, _retry_empty=False, **kwargs
            )

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Run TTS inference via subprocess. Returns (wav_bytes, meta_dict)."""
        if not self._ready:
            raise RuntimeError("TTS backend not preloaded")
        if self._product_backend is not None:
            return self._synthesize_single(
                text, speaker_id=speaker_id, speed=speed,
                pitch_shift=pitch_shift, language=language, **kwargs,
            )

        if self._config.segment_text and kwargs.get("segment_text", True):
            segments = _split_tts_text(
                text,
                kwargs.get("segment_max_chars"),
                max_chars_latin=self._config.segment_max_chars_latin,
                max_chars_cjk=self._config.segment_max_chars_cjk,
            )
            if len(segments) > 1:
                segment_kwargs = dict(kwargs)
                segment_kwargs["segment_text"] = False
                segment_kwargs.setdefault("seed", self._config.seed)
                wav_parts: list[bytes] = []
                segment_meta: list[dict] = []
                total_elapsed = 0.0
                total_duration = 0.0
                total_samples = 0
                for segment in segments:
                    wav, meta = self.synthesize(
                        segment, speaker_id=speaker_id, speed=speed,
                        pitch_shift=pitch_shift, language=language, **segment_kwargs,
                    )
                    wav_parts.append(wav)
                    segment_meta.append({"text": segment, **meta})
                    total_elapsed += float(meta.get("inference_time_s", 0.0))
                    wav_duration, wav_samples = _wav_duration_and_samples(wav)
                    total_duration += wav_duration
                    total_samples += wav_samples

                pauses_ms = [
                    _segment_pause_ms(
                        segment,
                        pause_ms=self._config.segment_pause_ms,
                        hard_pause_ms=self._config.segment_hard_pause_ms,
                    )
                    for segment in segments[:-1]
                ]
                wav_bytes = _concat_wav_bytes(wav_parts, pauses_ms)
                meta = {
                    "inference_time_s": round(total_elapsed, 3),
                    "sample_rate": self.sample_rate,
                    "duration_s": round(total_duration + sum(pauses_ms) / 1000.0, 3),
                    "samples": total_samples + int(self.sample_rate * sum(pauses_ms) / 1000.0),
                    "rtf": round(total_elapsed / total_duration, 3) if total_duration > 0 else 0.0,
                    "segmented": True,
                    "segment_count": len(segments),
                    "segment_pauses_ms": pauses_ms,
                    "segments": segment_meta,
                }
                return wav_bytes, meta

        return self._synthesize_single(
            text, speaker_id=speaker_id, speed=speed,
            pitch_shift=pitch_shift, language=language, **kwargs,
        )

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if self._product_backend is not None:
            return self._product_backend.clone_voice(
                text=text, speaker_embedding=speaker_embedding,
                language=language, speed=speed, **kwargs,
            )
        if len(speaker_embedding) % 4 != 0:
            raise ValueError("speaker_embedding must be a float32 byte vector")
        return self.synthesize(
            text, speed=speed, language=language,
            speaker_embedding=speaker_embedding, **kwargs,
        )

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        if self._product_backend is not None:
            return self._product_backend.extract_speaker_embedding(audio_wav_bytes)
        if not self._speaker_encoder or not os.path.exists(self._speaker_encoder):
            raise NotImplementedError(f"speaker encoder not found: {self._speaker_encoder}")
        return _qwen3_speaker_embed_inproc(audio_wav_bytes, self._speaker_encoder)

    def _synthesize_single(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Run one already-bounded TTS request."""
        if self._product_backend is not None:
            return self._product_backend.synthesize(
                text, speaker_id=speaker_id, speed=speed,
                pitch_shift=pitch_shift, language=language, **kwargs,
            )
        if self._use_worker():
            return self._synthesize_worker(text, language, speaker_id=speaker_id, **kwargs)

        input_data = {
            "requests": [
                {
                    "messages": [{"role": "user", "content": text}],
                    "speaker": "",
                }
            ],
            "batch_size": 1,
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "talker_temperature": self._config.talker_temperature,
            "talker_top_k": self._config.talker_top_k,
            "talker_top_p": self._config.talker_top_p,
            "repetition_penalty": self._config.repetition_penalty,
            "codec_eos_logit_offset": self._config.codec_eos_logit_offset,
            "predictor_temperature": self._config.predictor_temperature,
            "predictor_top_k": self._config.predictor_top_k,
            "predictor_top_p": self._config.predictor_top_p,
            "max_audio_length": kwargs.get("max_audio_length", self._config.max_audio_length),
            "min_audio_length": kwargs.get("min_audio_length", self._config.min_audio_length),
        }
        voice_kwargs = self._resolve_voice_kwargs({"speaker_id": speaker_id, **kwargs})
        self._add_speaker_request_fields(input_data["requests"][0], voice_kwargs)

        with tempfile.TemporaryDirectory(prefix="trt_edgellm_tts_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            output_path = os.path.join(tmpdir, "output.json")
            audio_dir = os.path.join(tmpdir, "audio_out")
            os.makedirs(audio_dir, exist_ok=True)

            with open(input_path, "w") as f:
                json.dump(input_data, f)

            cli_args = [
                "--inputFile", input_path,
                "--talkerEngineDir", self._talker_dir,
                "--codePredictorEngineDir", self._code_predictor_dir,
                "--tokenizerDir", self._tokenizer_dir,
                "--outputFile", output_path,
                "--outputAudioDir", audio_dir,
            ]

            c2w_path = _code2wav_engine_path(self._code2wav_dir)
            if os.path.exists(c2w_path):
                cli_args += ["--code2wavEngineDir", self._code2wav_dir]

            t0 = time.time()
            result = run_binary(
                self._config.tts_binary, cli_args, timeout=120,
                plugin_path=self._config.plugin_path,
            )
            elapsed = time.time() - t0

            if result.returncode != 0 or not os.path.exists(output_path):
                raise RuntimeError(
                    f"TTS subprocess failed (exit={result.returncode}): "
                    f"stdout={result.stdout[-300:]}, stderr={result.stderr[-300:]}"
                )

            with open(output_path) as f:
                output_data = json.load(f)

            responses = output_data.get("responses", [])
            if not responses:
                raise RuntimeError(f"TTS produced no responses: {output_data}")

            r = responses[0]
            audio_file = r.get("audio_file")
            wav_bytes = b""
            meta = {"inference_time_s": round(elapsed, 3), "sample_rate": 24000}

            if audio_file and os.path.exists(audio_file):
                with open(audio_file, "rb") as f:
                    wav_bytes = f.read()
                meta["duration_s"] = r.get("audio_duration_ms", 0) / 1000.0
                meta["samples"] = r.get("audio_samples", 0)
            else:
                logger.warning("No audio WAV in output, returning RVQ codes only")
                meta["rvq_file"] = r.get("rvq_file")
                if not meta.get("rvq_file"):
                    raise RuntimeError(
                        f"TTS output has neither audio nor RVQ: {list(r.keys())}"
                    )

            return wav_bytes, meta

    def _resolve_voice_kwargs(self, kwargs: dict) -> dict:
        sid = kwargs.get("speaker_id", kwargs.get("sid"))
        forward = {k: v for k, v in kwargs.items() if k not in ("speaker_id", "sid")}
        return resolve_speaker_kwargs(self.model_id, speaker_id=sid, **forward)

    @staticmethod
    def _add_speaker_request_fields(request: dict, voice_kwargs: dict) -> None:
        if not voice_kwargs:
            return
        if "speaker_id" in voice_kwargs:
            request["speaker_id"] = int(voice_kwargs["speaker_id"])
        if "speaker" in voice_kwargs:
            request["speaker"] = str(voice_kwargs["speaker"])
