"""TTS backend via TRT-Edge-LLM C++ binary (qwen3_tts_inference).

Calls the binary per-request with temp-file I/O.
Supports: BASIC_TTS, MULTI_LANGUAGE
Audio output: WAV via Code2Wav (vocoder) engine.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import base64
import io
import wave
from collections import deque
from typing import Iterator, Optional
import importlib
import uuid

from app.core.tts_backend import TTSBackend, TTSCapability
from app.core.tts_speakers import resolve_speaker_kwargs
from app.core.worker_io import WorkerIO, WorkerExitError

from app.backends.jetson.trt_edge_llm_ipc import (
    TTS_BINARY,
    PLUGIN_PATH,
    qwen3_highperf_enabled,
    qwen3_runtime_profile,
    resolve_tts_talker_dir,
    resolve_tts_code_predictor_dir,
    resolve_tts_tokenizer_dir,
    resolve_tts_code2wav_dir,
    resolve_tts_worker_binary,
    run_binary,
)

logger = logging.getLogger(__name__)


class PoolSaturatedError(RuntimeError):
    """TTS worker rejected a request because every decoder slot is busy.

    The C++ ``qwen3_tts_worker`` launched with ``--max_slots N`` returns
    ``{"event":"error","error":"pool_saturated","status":4429,"max_slots":N}``
    when an N+1st concurrent request arrives while all N slots are in flight.
    This is the worker-internal fail-fast that backstops the Python
    ``WorkerIO`` semaphore admission gate (sized to the same N); OVS callers
    map it to a 429/4429 reject-not-queue busy response.

    NB (mirrors the ASR backend's PoolSaturatedError, by design):
    intentionally a *plain* ``RuntimeError`` and NOT routed into the
    destructive worker-restart path. ``_generate_streaming_single`` only
    restarts the worker on an *empty* stateful result (0 chunks emitted);
    a saturation never reaches that branch because it is raised before any
    chunk loop runs. Keeping saturation off the restart path means the
    N-1 healthy in-flight requests on the shared worker are not torn down
    by an N+1st rejection. The ``status`` field is carried so the WS/HTTP
    layer can surface the 4429/429 reject-not-queue response.
    """

    status: int = 4429

    def __init__(self, message: str, max_slots: Optional[int] = None) -> None:
        super().__init__(message)
        self.max_slots = max_slots


def _tts_pool_saturated_error(event: dict) -> Optional[PoolSaturatedError]:
    """Return a PoolSaturatedError if ``event`` is a worker saturation reject.

    Recognized by the structured ``status: 4429`` field OR the error wording
    (``pool_saturated`` / legacy ``too_many_*`` phrasing) so a fast-fail busy
    rejection propagates as PoolSaturatedError instead of a generic
    RuntimeError. Returns ``None`` for any other (or non-error) event so the
    caller's existing error handling stays unchanged.
    """
    if not isinstance(event, dict):
        return None
    # Only inspect explicit failures: error events or ok:false payloads.
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


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


_QWEN3_TTS_MODEL_BASE = _env(
    "OVS_TTS_MODEL_BASE",
    "QWEN3_MODEL_BASE",
    default="/opt/models/qwen3-tts",
)
def _resolve_speaker_encoder() -> str:
    """Speaker encoder onnx path, with fallback to the qwen3-edgellm artifact tree."""
    explicit = os.environ.get("QWEN3_SPEAKER_ENCODER", "")
    if explicit:
        return explicit
    qwen3_root = os.environ.get("QWEN3_ARTIFACT_ROOT", "")
    if qwen3_root:
        candidate = os.path.join(qwen3_root, "tts", "speaker_encoder", "speaker_encoder.onnx")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_QWEN3_TTS_MODEL_BASE, "onnx", "speaker_encoder.onnx")




def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


def _tts_perf_profile() -> str:
    return os.environ.get("EDGE_LLM_TTS_PERF_PROFILE", "quality").strip().lower()


def _tts_fast_profile(profile: str) -> bool:
    return profile in ("fast", "v2v", "low_latency")


def _tts_balanced_or_fast_profile(profile: str) -> bool:
    return profile in ("balanced", "fast", "v2v", "low_latency")


def _tts_stateful_code2wav_enabled() -> bool:
    # Stateful Code2Wav is the validated low-latency streaming path. It keeps
    # subsequent chunks small and continuous instead of optimizing only TTFA.
    return _env_flag("EDGE_LLM_TTS_STATEFUL_CODE2WAV", default=qwen3_highperf_enabled())


def _tts_seed(default: int = 42) -> int:
    return int(_env("OVS_TTS_SEED", default=str(default)))


def _env_float(default: float, *names: str) -> float:
    return float(_env(*names, default=str(default)))


def _env_int(default: int, *names: str) -> int:
    return int(_env(*names, default=str(default)))


_SPEAKER_ENC_SESSION = None  # cached ort session — onnx load is non-trivial


def _qwen3_speaker_embed_inproc(audio_wav_bytes: bytes, encoder_path: str) -> bytes:
    """In-process 1024-d speaker embedding from a reference WAV.

    Drop-in replacement for the previous subprocess + librosa script — uses
    pure numpy mel + ONNX Runtime. Inputs accepted at any mono PCM WAV rate;
    audio is linearly resampled to 24 kHz (matches official qwen3-tts pipeline).
    Returns raw float32 little-endian bytes (4096 bytes for the 1024-d output).
    """
    global _SPEAKER_ENC_SESSION
    import io
    import numpy as np
    import soundfile as sf
    import onnxruntime as ort

    # ── decode WAV → mono float32 @ 24 kHz ────────────────────────
    data, sr_in = sf.read(io.BytesIO(audio_wav_bytes), always_2d=False, dtype="float32")
    if data.ndim == 2:  # stereo → mono mix
        data = data.mean(axis=1).astype(np.float32)
    if sr_in != 24000:
        # FFT-based resample (band-limited, anti-aliased). Equivalent to
        # scipy.signal.resample but pure numpy. Quality matters: linear
        # interpolation aliases high freq into the mel band → garbage
        # embedding → unrecognizable cloned voice.
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

    # ── mel pipeline matching qwen3-tts mel_spectrogram ───────────
    N_FFT, HOP, WIN, N_MEL = 1024, 256, 1024, 128
    FMIN, FMAX = 0.0, 12000.0

    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _slaney_mel_filterbank() -> np.ndarray:
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
        # Slaney normalization: enorm = 2 / (hi - lo)
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
    mel_spec = mag @ mel_basis.T  # [T, n_mels]
    mel_spec = np.log(np.clip(mel_spec, 1e-5, None)).astype(np.float32)

    # ── ONNX inference (CPU) ──────────────────────────────────────
    if _SPEAKER_ENC_SESSION is None or _SPEAKER_ENC_SESSION[0] != encoder_path:
        sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        _SPEAKER_ENC_SESSION = (encoder_path, sess)
    sess = _SPEAKER_ENC_SESSION[1]
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: mel_spec[None, ...]})  # [1, T, 128]
    emb = out[0].squeeze().astype(np.float32)
    if emb.shape != (1024,):
        raise RuntimeError(f"unexpected speaker embedding shape: {emb.shape}")
    return emb.tobytes()


def _code2wav_engine_path(code2wav_dir: str | None = None) -> str:
    """Return the Code2Wav engine path used by current Qwen3 artifact sets.

    Re-reads env via resolve_tts_code2wav_dir() when no dir is supplied so
    callers from module scope (no instance state) still respect hot reload.
    Backend instances should pass their captured ``self._code2wav_dir`` to
    keep the resolution consistent with what __init__ saw.
    """
    if code2wav_dir is None:
        code2wav_dir = resolve_tts_code2wav_dir()
    candidates = [
        os.path.join(code2wav_dir, "code2wav.engine"),
        os.path.join(code2wav_dir, "code2wav_stateful.engine"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _detect_language(text: str) -> str:
    """Simple language detection — returns config-compatible language strings."""
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


def _split_tts_text(text: str, max_chars: Optional[int] = None) -> list[str]:
    """Split long TTS text into independently stable synthesis requests."""
    normalized = " ".join(text.split()) if not _contains_cjk(text) else text.strip()
    if not normalized:
        return []

    if max_chars is None:
        env_name = "EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS" if _contains_cjk(normalized) else "EDGE_LLM_TTS_SEGMENT_MAX_CHARS"
        default_chars = "48" if _contains_cjk(normalized) else "120"
        max_chars = int(os.environ.get(env_name, default_chars))
    max_chars = max(8, max_chars)

    hard_breaks = set("。！？!?；;\n")
    soft_breaks = set("，,、：:")
    segments: list[str] = []
    current: list[str] = []
    is_cjk = _contains_cjk(normalized)
    # CJK stateful Code2Wav is sensitive to long punctuation-free runs; keep
    # the configured limit as a hard cap instead of allowing the Latin-word
    # overrun used to avoid awkward English word breaks.
    max_overrun = 0 if is_cjk else max(2, min(8, max_chars // 2))
    abbreviations = {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
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
                tail = text_so_far[cut + 1 :].lstrip()
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


def _segment_pause_ms(segment: str) -> int:
    """Silence to insert *after* a synthesized segment when concatenating.

    Only inserts when the segment ends in natural punctuation — a comma /
    period earns a real prosodic pause. A segment cut mid-phrase by the
    16-char safety limit gets **zero** padding: there's no linguistic
    reason to break the flow, so the audio should run continuously.
    """
    if not segment:
        return 0
    pause_ms = int(os.environ.get("EDGE_LLM_TTS_SEGMENT_PAUSE_MS", "80"))
    hard_pause_ms = int(os.environ.get("EDGE_LLM_TTS_HARD_SEGMENT_PAUSE_MS", "120"))
    stripped = segment.rstrip()
    if stripped.endswith(("。", "！", "？", "!", "?", ";", "；")):
        return max(0, hard_pause_ms)
    if stripped.endswith(("，", ",", "、", "：", ":")):
        return max(0, pause_ms)
    return 0  # forced cut mid-phrase (no punctuation) — no synthetic silence


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
    """Return the request id of a worker stdout event.

    Phase 1 of the TTS worker concurrency spec (docs/specs/tts-worker-concurrency.md)
    adds a ``request_id`` field to every stdout event while keeping the legacy
    ``id`` field as an alias for back-compat. Callers that need to demux
    events back to a request (or just defensively verify the worker's
    response belongs to the request they issued) should read the id through
    this helper rather than indexing ``event["id"]`` directly. At N=1 with
    ``_worker_lock`` the demux is implicit, so this is informational today;
    it becomes load-bearing once the Phase 2/3 reader thread lands.
    """
    rid = event.get("request_id")
    if rid:
        return rid
    rid = event.get("id")
    return rid if rid else None


def _pcm16_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw mono int16 little-endian PCM in a minimal WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# Default sampling parameters
_DEFAULT_TEMPERATURE = _env_float(
    0.9,
    "OVS_TTS_TALKER_TEMPERATURE",
    "TTS_TALKER_TEMPERATURE",
)
_DEFAULT_TOP_K = _env_int(
    50,
    "OVS_TTS_TALKER_TOP_K",
    "TTS_TALKER_TOP_K",
)
_DEFAULT_TOP_P = _env_float(
    1.0,
    "OVS_TTS_TOP_P",
    "TTS_TOP_P",
)
_DEFAULT_PREDICTOR_TEMPERATURE = _env_float(
    0.9,
    "OVS_TTS_PREDICTOR_TEMPERATURE",
    "TTS_PREDICTOR_TEMPERATURE",
)
_DEFAULT_PREDICTOR_TOP_K = _env_int(
    50,
    "OVS_TTS_PREDICTOR_TOP_K",
    "TTS_PREDICTOR_TOP_K",
)
_DEFAULT_PREDICTOR_TOP_P = _env_float(
    1.0,
    "OVS_TTS_PREDICTOR_TOP_P",
    "TTS_PREDICTOR_TOP_P",
)
_DEFAULT_MAX_AUDIO_LENGTH = int(os.environ.get("TTS_MAX_AUDIO_LENGTH", "1024"))
_DEFAULT_MIN_AUDIO_LENGTH = int(os.environ.get("TTS_MIN_AUDIO_LENGTH", "30"))
_DEFAULT_REPETITION_PENALTY = float(os.environ.get("TTS_REPETITION_PENALTY", "1.05"))
_DEFAULT_CODEC_EOS_LOGIT_OFFSET = float(os.environ.get("TTS_CODEC_EOS_LOGIT_OFFSET", "0"))
_DEFAULT_SEGMENT_TEXT = os.environ.get("EDGE_LLM_TTS_SEGMENT_TEXT", "1").lower() not in ("0", "false", "no")



class TRTEdgeLLMTTSBackend(TTSBackend):
    """TTS via TRT-Edge-LLM qwen3_tts_inference subprocess."""

    @classmethod
    def concurrency_capability(cls, profile=None):
        from app.core.concurrency_capability import ConcurrencyCapability

        # _WorkerIO multiplexes N in-flight requests with a single subprocess
        # (one stdin lock + one stdout reader + per-request queues). See
        # spec Section 1 table row "trt_edge_llm_tts" and
        # app/backends/jetson/trt_edge_llm_tts.py:486 (_WorkerIO).
        env_val = os.environ.get("OVS_TTS_WORKER_CONCURRENCY")
        profile_val = None
        if isinstance(profile, dict):
            # accept both top-level and nested under tts_backend_config
            profile_val = profile.get("tts_worker_concurrency")
            if profile_val is None:
                cfg = profile.get("tts_backend_config")
                if isinstance(cfg, dict):
                    profile_val = cfg.get("worker_concurrency")
        try:
            n = int(env_val) if env_val is not None else (
                int(profile_val) if profile_val is not None else 1
            )
        except (TypeError, ValueError):
            n = 1
        n = max(1, n)
        return ConcurrencyCapability(
            supports_parallel=n > 1,
            max_concurrent=n,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    # PR5b: supports_hot_reload is mode-dependent. The default
    # edgellm_worker / official modes spawn a TRT subprocess that can be
    # terminated to fully release GPU memory (safe to hot-reload). The
    # product_explicit_kv mode embeds an in-process Qwen3TRTBackend
    # (pybind / TRT context held in this process); Qwen3TRTBackend itself
    # declares supports_hot_reload=False, so we must mirror that.
    #
    # Implemented as an instance property so BackendManager's
    # ``getattr(self._current, "supports_hot_reload", False)`` returns the
    # correct per-instance value.
    @property
    def supports_hot_reload(self) -> bool:  # type: ignore[override]
        mode = getattr(self, "_resolved_mode", None) or self._backend_mode()
        if mode in ("product_explicit_kv", "explicit_kv"):
            return False
        return True

    def __init__(self):
        self._ready = False
        self._product_backend = None
        self._resolved_mode: Optional[str] = None
        self._worker: Optional[subprocess.Popen] = None
        # _worker_lock is retained for unload()/restart lifecycle serialization
        # (process spawn + handle swap). Per-request I/O serialization moved
        # to _WorkerIO (see TTS worker concurrency spec §4.2). The fine-grained
        # stdin lock + reader thread live inside ``self._worker_io``.
        self._worker_lock = threading.Lock()
        self._worker_io: Optional[WorkerIO] = None
        self._worker_concurrency: int = max(
            1, int(os.environ.get("OVS_TTS_WORKER_CONCURRENCY", "1"))
        )
        self._worker_ready_meta: dict = {}
        self._worker_stderr_tail = deque(maxlen=80)
        # Capture artifact paths from the *current* env at instance creation
        # time. BackendManager builds a fresh backend after apply_profile()
        # clears the factory cache, so __init__ always sees the latest
        # profile-applied env. Avoid relying on module-level constants in
        # trt_edge_llm_ipc.py which are frozen at first import — see the
        # resolve_*() helpers there.
        self._talker_dir = resolve_tts_talker_dir()
        self._code_predictor_dir = resolve_tts_code_predictor_dir()
        self._tokenizer_dir = resolve_tts_tokenizer_dir()
        self._speaker_encoder = _resolve_speaker_encoder()
        self._code2wav_dir = resolve_tts_code2wav_dir()
        self._worker_binary = resolve_tts_worker_binary()
        self._qwen3_runtime_profile = qwen3_runtime_profile()

    # -- TTSBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

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
        mode = _env("OVS_TTS_BACKEND", "EDGE_LLM_TTS_BACKEND", default="edgellm_worker")
        return mode.strip().lower().replace("-", "_")

    def _load_product_explicit_kv_backend(self):
        model_base = _env("OVS_TTS_MODEL_BASE", default="/home/harvest/voice_test/models/qwen3-tts")
        overlay = _env("OVS_TTS_NATIVE_MODULE_DIR", default="/home/harvest/voice_test/app_overlay")
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
            "Using OpenVoiceStream product_explicit_kv TTS backend (model_base=%s, talker=%s)",
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
                "Unsupported OVS_TTS_BACKEND/EDGE_LLM_TTS_BACKEND value "
                f"{mode!r}; expected edgellm_worker or product_explicit_kv"
            )

        explicit_talker_engine = os.environ.get("EDGE_LLM_TTS_TALKER_ENGINE")
        required = [
            (self._worker_binary if self._use_worker() else TTS_BINARY, "TTS binary"),
            (PLUGIN_PATH, "TRT-Edge-LLM plugin"),
            (os.path.join(self._talker_dir, "config.json"), "talker config"),
            (os.path.join(self._tokenizer_dir, "tokenizer.json"), "tokenizer"),
        ]
        if explicit_talker_engine:
            required.append((explicit_talker_engine, "explicit Talker engine"))
        else:
            required.append((os.path.join(self._talker_dir, "llm.engine"), "talker engine"))
        missing = []
        for path, label in required:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")
        if missing:
            raise FileNotFoundError(
                "TTS preload failed — missing:\n  " + "\n  ".join(missing)
            )

        # Code2Wav is optional (graceful fallback)
        c2w_path = _code2wav_engine_path(self._code2wav_dir)
        if os.path.exists(c2w_path):
            logger.info("Code2Wav engine found at %s", c2w_path)
        else:
            logger.warning(
                "Code2Wav not found at %s — will output RVQ codes only",
                c2w_path,
            )

        logger.info(
            "TTS backend preload OK (profile=%s binary=%s talker=%s)",
            self._qwen3_runtime_profile,
            self._worker_binary if self._use_worker() else TTS_BINARY,
            self._talker_dir,
        )
        if self._use_worker():
            self._ensure_worker()
        self._ready = True

    def unload(self) -> None:
        """Kill the resident worker subprocess so GPU memory is fully released.

        PR5: idempotent + early-return when never preloaded or no worker is
        running. Used by ``BackendManager.reload()`` between profile swaps.

        PR5c FIX_1: only early-return when *all* runtime handles are empty —
        previously a half-finished preload could leave ``_product_backend``
        non-None while ``_ready=False`` and ``_worker=None``, leaking the
        embedded product backend's GPU memory. The product cleanup now also
        runs in a ``finally`` block so a failure in the worker teardown path
        still releases the embedded Qwen3 TRT backend.
        """
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
                # Drop _WorkerIO first so its reader thread sees EOF
                # cleanly when we kill the subprocess below, and wakes any
                # (unexpected) in-flight callers with the exit sentinel.
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
            # product_backend cleanup MUST run even if the worker teardown
            # path above raised, otherwise the embedded Qwen3 TRT backend
            # would leak GPU memory across profile swaps.
            if self._product_backend is not None:
                try:
                    self._product_backend.unload()
                except Exception:
                    logger.exception(
                        "product_backend.unload failed; continuing"
                    )
                self._product_backend = None
            self._resolved_mode = None
            self._ready = False

    def _use_worker(self) -> bool:
        return os.environ.get("EDGE_LLM_TTS_WORKER", "1").lower() not in ("0", "false", "no")

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = PLUGIN_PATH
        # On the current W8A16 Talker path CUDA graph has not shown measurable
        # RTF gain, while it costs extra resident memory during dual ASR+TTS.
        env.setdefault("EDGE_LLM_TTS_CUDA_GRAPH", "0")
        env.setdefault("EDGE_LLM_TTS_LAZY_CODE2WAV", "0")
        env.setdefault("EDGE_LLM_TTS_STATEFUL_CODE2WAV", "1" if qwen3_highperf_enabled() else "0")
        stateful_engine_dir = os.environ.get(
            "EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR",
            "/tmp/qwen3_code2wav_stateful_engine",
        )
        if os.path.isdir(stateful_engine_dir):
            env.setdefault("EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR", stateful_engine_dir)
        if _tts_stateful_code2wav_enabled():
            env.setdefault("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", "0")
            env.setdefault("QWEN3_TTS_CP_DECODE_CUDA_GRAPH", "1")
            env.setdefault("QWEN3_TTS_ACTIVE_CP_GROUPS", "13")
        else:
            # Legacy stateless vocoder100 path: first chunk 50 frames, then 97
            # new frames + 3 context.
            env.setdefault("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", "3")
        vocab_pruned = os.environ.get("EDGE_LLM_TTS_VOCAB_PRUNED")
        if vocab_pruned is not None:
            env.setdefault("QWEN3_TTS_VOCAB_PRUNED", vocab_pruned)
        else:
            env.setdefault("QWEN3_TTS_VOCAB_PRUNED", "0")
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
        # Re-read concurrency from env before the spawn so a fresh backend
        # built after profile env was applied (BackendManager rebuilds on
        # apply_profile) launches the worker with the right slot count. This
        # mirrors the post-ready refresh below; doing it here ensures the
        # ``--max_slots`` arg and the WorkerIO semaphore agree.
        self._worker_concurrency = max(
            1, int(os.environ.get("OVS_TTS_WORKER_CONCURRENCY", str(self._worker_concurrency)))
        )
        cmd = [
            self._worker_binary,
            "--talkerEngineDir",
            self._talker_dir,
            "--codePredictorEngineDir",
            self._code_predictor_dir,
            "--tokenizerDir",
            self._tokenizer_dir,
            "--code2wavEngineDir",
            self._code2wav_dir,
        ]
        # Slot-pool size (TTS D2-3). The worker allocates N decoder slots over
        # the shared engines; the Python WorkerIO semaphore below is bound to
        # the same N. Only emit --max_slots when N>1: at N=1 we omit it for
        # legacy single-slot byte-equivalent behavior AND backward compat with
        # worker binaries built before --max_slots existed (older images reject
        # the unknown flag). The C++ worker also reads OVS_TTS_WORKER_CONCURRENCY
        # from env (still passed through _worker_env).
        if self._worker_concurrency and self._worker_concurrency > 1:
            cmd += ["--max_slots", str(self._worker_concurrency)]
        optional_flags = [
            ("EDGE_LLM_TTS_TALKER_BACKEND", "--qwen3TtsTalkerBackend"),
            ("EDGE_LLM_TTS_TALKER_ENGINE", "--qwen3TtsTalkerEngine"),
            ("EDGE_LLM_TTS_CODE_PREDICTOR_BACKEND", "--codePredictorBackend"),
            ("EDGE_LLM_TTS_TEXT_PROJECTION", "--qwen3TtsTextProjection"),
            ("EDGE_LLM_TTS_PROMPT_KV_CACHE", "--qwen3TtsPromptKvCache"),
        ]
        for env_name, flag in optional_flags:
            value = os.environ.get(env_name)
            if value:
                cmd.extend([flag, value])
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
        # Re-read concurrency in case the env was applied between __init__
        # and the first preload (BackendManager applies profile env before
        # building the backend, but later hot reloads come via __init__ on a
        # fresh instance — so this is mostly defensive).
        self._worker_concurrency = max(
            1, int(os.environ.get("OVS_TTS_WORKER_CONCURRENCY", str(self._worker_concurrency)))
        )
        self._worker_io = WorkerIO(self._worker, self._worker_concurrency)

    def _restart_worker_locked(self, reason: str) -> None:
        """Restart the resident TTS worker.

        Called from inside ``_worker_lock`` (held by the streaming retry-on-
        empty path). Drops the current ``_WorkerIO`` so the daemon reader
        thread exits cleanly via EOF on the killed process's stdout, then
        spawns a fresh process + new ``_WorkerIO`` via ``_ensure_worker``.
        Any other in-flight request on the old ``_WorkerIO`` (none expected
        in the empty-stream retry path, but safe by construction) receives
        the ``_worker_exit`` sentinel and raises ``WorkerExitError``.
        """
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
        if _tts_stateful_code2wav_enabled():
            # The C++ worker rejects oneshot requests with "Stateful Code2Wav
            # currently requires stream=true". Drive the same streaming path
            # that /tts/stream uses, accumulate raw PCM, and wrap as a single
            # WAV so the public non-streaming /tts endpoint stays usable.
            return self._synthesize_worker_via_stream(text, language=language, **kwargs)
        req_id = uuid.uuid4().hex
        with tempfile.NamedTemporaryFile(prefix="trt_edgellm_tts_", suffix=".wav", delete=False) as f:
            output_file = f.name
        request = {
            "id": req_id,
            "text": text,
            "output_file": output_file,
            "language": language or _detect_language(text),
            "talker_temperature": _DEFAULT_TEMPERATURE,
            "talker_top_k": _DEFAULT_TOP_K,
            "talker_top_p": _DEFAULT_TOP_P,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
            "codec_eos_logit_offset": _DEFAULT_CODEC_EOS_LOGIT_OFFSET,
            "predictor_temperature": _DEFAULT_PREDICTOR_TEMPERATURE,
            "predictor_top_k": _DEFAULT_PREDICTOR_TOP_K,
            "predictor_top_p": _DEFAULT_PREDICTOR_TOP_P,
            "max_audio_length": kwargs.get("max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH),
            "min_audio_length": kwargs.get("min_audio_length", _DEFAULT_MIN_AUDIO_LENGTH),
            "seed": int(kwargs.get("seed", _tts_seed())),
        }
        voice_kwargs = self._resolve_voice_kwargs(kwargs)
        speaker_embedding = voice_kwargs.get("speaker_embedding")
        if speaker_embedding:
            request["speaker_embedding_b64"] = base64.b64encode(speaker_embedding).decode("ascii")
        else:
            self._add_speaker_request_fields(request, voice_kwargs)
        # Phase 3b-B-3 (TTS worker concurrency): one-shot path now goes through
        # _WorkerIO.request() — same protocol contract (one event per request
        # in the file-output mode), but the stdin write is serialized only at
        # the byte level and the response is demuxed by request_id, allowing
        # other concurrent requests to interleave at the stdout reader.
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
        # Slot-pool saturation (status 4429): fast-fail busy, NOT a worker
        # fault. Raise PoolSaturatedError (plain RuntimeError) so the WS/HTTP
        # layer surfaces a reject-not-queue 429/4429 without tearing down the
        # worker or the other in-flight requests.
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
        """Aggregate streaming PCM chunks into a single WAV.

        Used by the non-streaming /tts endpoint when stateful Code2Wav is on
        (the C++ worker only accepts stream=true in that mode).
        """
        t0 = time.time()
        done_meta: dict = {}
        # Force segment_text=False so we drive a single streaming request
        # here; outer synthesize() already split into segments if needed.
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

        if _DEFAULT_SEGMENT_TEXT and kwargs.get("segment_text", True):
            segments = _split_tts_text(text, kwargs.get("segment_max_chars"))
            if len(segments) > 1:
                segment_kwargs = dict(kwargs)
                segment_kwargs["segment_text"] = False
                segment_kwargs.setdefault("seed", _tts_seed())
                for segment in segments:
                    yield from self.generate_streaming(segment, **segment_kwargs)
                return

        yield from self._generate_streaming_single(text, **kwargs)

    def _generate_streaming_single(self, text: str, meta_out: Optional[dict] = None, **kwargs):
        """Yield raw PCM int16 chunks for one already-bounded TTS request.

        If ``meta_out`` is a dict, the worker's terminal ``done`` event JSON
        (rtf, total_ms, first_chunk_ms, etc.) is merged into it before the
        generator returns. Used by ``_synthesize_worker`` to assemble a
        single-shot WAV in stateful Code2Wav mode.
        """
        retry_empty = bool(kwargs.pop("_retry_empty", True))
        req_id = uuid.uuid4().hex
        streaming_profile = str(
            kwargs.get("streaming_profile", os.environ.get("EDGE_LLM_TTS_STREAMING_PROFILE", "continuous_playback"))
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
        if _tts_stateful_code2wav_enabled():
            perf_profile = _tts_perf_profile()
            if _tts_fast_profile(perf_profile):
                default_first_chunk_frames = 4
            elif perf_profile == "balanced":
                default_first_chunk_frames = 6
            else:
                # 7 frames is the smallest validated default that still gives
                # the second chunk overlap with stateful Code2Wav on Orin Nano.
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
            "talker_temperature": _DEFAULT_TEMPERATURE,
            "talker_top_k": _DEFAULT_TOP_K,
            "talker_top_p": _DEFAULT_TOP_P,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
            "codec_eos_logit_offset": _DEFAULT_CODEC_EOS_LOGIT_OFFSET,
            "predictor_temperature": _DEFAULT_PREDICTOR_TEMPERATURE,
            "predictor_top_k": _DEFAULT_PREDICTOR_TOP_K,
            "predictor_top_p": _DEFAULT_PREDICTOR_TOP_P,
            "max_audio_length": kwargs.get("max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH),
            "min_audio_length": kwargs.get("min_audio_length", _DEFAULT_MIN_AUDIO_LENGTH),
            "seed": int(kwargs.get("seed", _tts_seed())),
            "stream": True,
            "stream_only": True,
            "first_chunk_frames": kwargs.get(
                "first_chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", str(default_first_chunk_frames))),
            ),
            "chunk_frames": kwargs.get(
                "chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_CHUNK_FRAMES", str(default_chunk_frames))),
            ),
            "adaptive_chunks": kwargs.get(
                "adaptive_chunks",
                os.environ.get("EDGE_LLM_TTS_ADAPTIVE_CHUNKS", "1" if default_adaptive_chunks else "0").lower()
                not in ("0", "false", "no"),
            ),
            "max_chunk_frames": kwargs.get(
                "max_chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_MAX_CHUNK_FRAMES", str(default_max_chunk_frames))),
            ),
            "chunk_growth_frames": kwargs.get(
                "chunk_growth_frames",
                int(os.environ.get("EDGE_LLM_TTS_CHUNK_GROWTH_FRAMES", str(default_chunk_growth_frames))),
            ),
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
        # Phase 3b-B-3 (TTS worker concurrency): streaming path now iterates
        # over _WorkerIO.request() — the daemon reader thread demuxes events
        # to this request's queue by request_id. The old end-to-end
        # _worker_lock is no longer held across the chunk loop, so other
        # in-flight requests can run concurrently up to OVS_TTS_WORKER_CONCURRENCY.
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker_io is not None
            worker_io = self._worker_io
        try:
            for event in worker_io.request(request):
                # Forward-compat: worker emits both "request_id" and "id".
                # _WorkerIO already demuxes by request_id, but keep the
                # mismatch-warn for protocol-drift detection.
                event_rid = _event_request_id(event)
                if event_rid is not None and event_rid != req_id and event_rid != "__worker__":
                    logger.debug(
                        "TTS worker event id mismatch: expected=%s got=%s event=%s",
                        req_id,
                        event_rid,
                        event.get("event"),
                    )
                # Cooperative cancel: terminal "cancelled" event has ok:true
                # per spec §4.1; check it BEFORE the ok-flag gate so it
                # never surfaces as a RuntimeError.
                if event.get("event") == "cancelled":
                    logger.info(
                        "TTS worker acknowledged cancel for %s (reason=%s)",
                        req_id,
                        event.get("reason"),
                    )
                    return
                # Slot-pool saturation (status 4429): fast-fail busy. Raise
                # PoolSaturatedError (a plain RuntimeError) BEFORE the generic
                # ok-flag gate so it never surfaces as an opaque RuntimeError
                # and never reaches the empty-stream restart branch below — a
                # saturation is not a worker fault, so the shared worker (and
                # its other N-1 in-flight requests) must NOT be restarted.
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
                    # Surface the worker's terminal metadata to callers that
                    # passed a `meta_out` dict (used by _synthesize_worker in
                    # stateful Code2Wav mode to assemble a single-shot WAV).
                    if meta_out is not None and isinstance(meta_out, dict):
                        meta_out.update(event)
                    if (
                        retry_empty
                        and _tts_stateful_code2wav_enabled()
                        and emitted_chunks == 0
                    ):
                        retry_after_empty = True
                        # Reacquire the lifecycle lock for the restart —
                        # other concurrent requests on the old worker_io
                        # will receive _worker_exit sentinels and raise
                        # WorkerExitError (caller handles).
                        with self._worker_lock:
                            self._restart_worker_locked(
                                f"stateful stream returned 0 chunks for request {req_id}"
                            )
                    break
        except GeneratorExit:
            # Consumer abandoned us mid-stream (HTTP client disconnect /
            # caller broke out of the for loop). Notify the worker so its
            # CUDA context doesn't get poisoned by partial cleanup — see
            # docs/specs/tts-worker-cancel-protocol.md §1.
            logger.info(
                "generator exit during TTS stream; cancelling worker for %s",
                req_id,
            )
            try:
                worker_io.cancel(req_id)
            except Exception:
                logger.debug("worker_io.cancel() failed during GeneratorExit",
                             exc_info=True)
            raise
        except WorkerExitError as exc:
            self._worker = None
            self._worker_io = None
            raise RuntimeError(
                f"TTS worker exited during stream: {self._worker_stderr_snip()}"
            ) from exc
        if retry_after_empty:
            logger.warning(
                "Retrying TTS stream after empty stateful result "
                "(done=%s stderr_tail=%s)",
                done_event,
                self._worker_stderr_snip(),
            )
            yield from self._generate_streaming_single(
                text,
                meta_out=meta_out,
                _retry_empty=False,
                **kwargs,
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
        """Run TTS inference via subprocess.

        Returns (wav_bytes, meta_dict).  ``wav_bytes`` will be empty if the
        Code2Wav engine is unavailable (the backend produced RVQ codes only).
        """
        if not self._ready:
            raise RuntimeError("TTS backend not preloaded")
        if self._product_backend is not None:
            return self._synthesize_single(
                text,
                speaker_id=speaker_id,
                speed=speed,
                pitch_shift=pitch_shift,
                language=language,
                **kwargs,
            )

        if _DEFAULT_SEGMENT_TEXT and kwargs.get("segment_text", True):
            segments = _split_tts_text(text, kwargs.get("segment_max_chars"))
            if len(segments) > 1:
                segment_kwargs = dict(kwargs)
                segment_kwargs["segment_text"] = False
                segment_kwargs.setdefault("seed", _tts_seed())
                wav_parts: list[bytes] = []
                segment_meta: list[dict] = []
                total_elapsed = 0.0
                total_duration = 0.0
                total_samples = 0
                for segment in segments:
                    wav, meta = self.synthesize(
                        segment,
                        speaker_id=speaker_id,
                        speed=speed,
                        pitch_shift=pitch_shift,
                        language=language,
                        **segment_kwargs,
                    )
                    wav_parts.append(wav)
                    segment_meta.append({"text": segment, **meta})
                    total_elapsed += float(meta.get("inference_time_s", 0.0))
                    wav_duration, wav_samples = _wav_duration_and_samples(wav)
                    total_duration += wav_duration
                    total_samples += wav_samples

                pauses_ms = [_segment_pause_ms(segment) for segment in segments[:-1]]
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
            text,
            speaker_id=speaker_id,
            speed=speed,
            pitch_shift=pitch_shift,
            language=language,
            **kwargs,
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
                text=text,
                speaker_embedding=speaker_embedding,
                language=language,
                speed=speed,
                **kwargs,
            )
        if len(speaker_embedding) % 4 != 0:
            raise ValueError("speaker_embedding must be a float32 byte vector")
        return self.synthesize(
            text,
            speed=speed,
            language=language,
            speaker_embedding=speaker_embedding,
            **kwargs,
        )

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        if self._product_backend is not None:
            return self._product_backend.extract_speaker_embedding(audio_wav_bytes)
        if not os.path.exists(self._speaker_encoder):
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
                text,
                speaker_id=speaker_id,
                speed=speed,
                pitch_shift=pitch_shift,
                language=language,
                **kwargs,
            )
        if self._use_worker():
            return self._synthesize_worker(text, language, speaker_id=speaker_id, **kwargs)

        # Build input JSON
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
            "talker_temperature": _DEFAULT_TEMPERATURE,
            "talker_top_k": _DEFAULT_TOP_K,
            "talker_top_p": _DEFAULT_TOP_P,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
            "codec_eos_logit_offset": _DEFAULT_CODEC_EOS_LOGIT_OFFSET,
            "predictor_temperature": _DEFAULT_PREDICTOR_TEMPERATURE,
            "predictor_top_k": _DEFAULT_PREDICTOR_TOP_K,
            "predictor_top_p": _DEFAULT_PREDICTOR_TOP_P,
            "max_audio_length": kwargs.get(
                "max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH
            ),
            "min_audio_length": kwargs.get(
                "min_audio_length", _DEFAULT_MIN_AUDIO_LENGTH
            ),
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

            # Build CLI args
            cli_args = [
                "--inputFile",
                input_path,
                "--talkerEngineDir",
                self._talker_dir,
                "--codePredictorEngineDir",
                self._code_predictor_dir,
                "--tokenizerDir",
                self._tokenizer_dir,
                "--outputFile",
                output_path,
                "--outputAudioDir",
                audio_dir,
            ]

            # Add code2wav if engine exists
            c2w_path = _code2wav_engine_path(self._code2wav_dir)
            if os.path.exists(c2w_path):
                cli_args += ["--code2wavEngineDir", self._code2wav_dir]

            t0 = time.time()
            result = run_binary(TTS_BINARY, cli_args, timeout=120)
            elapsed = time.time() - t0

            # Parse output — fail loudly on errors
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
        # Pop speaker_id/sid from a copy to avoid passing them twice
        # (resolve_speaker_kwargs also accepts them via **forward).
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
