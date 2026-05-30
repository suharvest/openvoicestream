"""RK ASR adapter — voxedge adapter.

adapted from app/backends/rk/asr.py + app/core/rk_*.py (2026-05-30), dedup
after registry switch.

Wraps ``rkvoice_stream.create_asr()`` output to fit the voxedge ASRBackend
interface. The two ABCs (voxedge's in ``voxedge.backends.base`` and
rkvoice-stream's in ``rkvoice_stream.engine.asr``) are intentionally
near-identical; this module bridges the capability enum and forwards methods.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (not ``app.core.*``);
    ``ConcurrencyCapability`` from ``voxedge.engine.concurrency_capability``.
  * ALL ``os.environ.get(...)`` reads (``RK_PLATFORM``,
    ``ASR_ENERGY_SPLIT_RMS``, ``ASR_ENERGY_MIN_SILENCE_MS``, long-audio
    threshold) replaced by an explicit ``RKASRConfig`` dataclass injected at
    construction time. voxedge has no module-scope or hardcoded env reads.
  * ``import rkvoice_stream`` stays inside ``__init__`` (lazy) so this module
    imports cleanly on a machine without the optional ``voxedge[rk]`` extra /
    rknn runtime (e.g. a Mac dev box).

Long-audio guard (from the production copy): the Qwen3-on-RKLLM pipeline has a
512-token decoder context cap and a sliding-window decoder that snowballs
garbage from one chunk into the next. On audio >~10s the model often bails to
its own instruction suffix ("转录") or hallucinates only the last segment. We
fix this in the adapter layer (no submodule changes) by:
  (a) energy-RMS splitting long audio into <=4.5s segments at silence
  (b) running each segment as an INDEPENDENT inner.transcribe() (fresh
      StreamSession internally → no cross-segment prefix poisoning)
  (c) discarding placeholder echoes (e.g. just "转录" or "转录：")
  (d) joining with a language-aware separator
"""
from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    TranscriptionResult,
)
from voxedge.engine.concurrency_capability import ConcurrencyCapability

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                  → RKASRConfig field
#   RK_PLATFORM                     → platform                 (default "rk3576")
#   ASR_ENERGY_SPLIT_RMS            → energy_split_rms         (default 0.003)
#   ASR_ENERGY_MIN_SILENCE_MS       → energy_min_silence_ms    (default 120)
#   (literal _LONG_AUDIO_THRESHOLD_S) → long_audio_threshold_s (default 15.0)

# Segmentation constants (unchanged from production; not env-derived).
_VAD_MAX_SEG_SEC = 4.5
_VAD_MIN_SEG_SEC = 0.5
_VAD_FRAME_MS = 20

# Outputs that mean "model gave up and echoed its own instruction suffix" —
# drop these from the joined transcript.
_PLACEHOLDER_OUTPUTS = {
    "", "转录", "转录。", "转录：", "转录:",
    "transcription", "transcription.", "transcription:",
}


@dataclass
class RKASRConfig:
    """Explicit construction-time config for :class:`RKASRBackend`.

    Every field default is identical to the production env default; nothing
    here reads ``os.environ``.

    ``platform`` mirrors the old ``RK_PLATFORM`` env (default ``"rk3576"``).
    The energy-split / long-audio-threshold fields were previously read from
    env *inside* the splitter on every call; they are now injected once.
    """

    platform: str = "rk3576"
    energy_split_rms: float = 0.003
    energy_min_silence_ms: int = 120
    long_audio_threshold_s: float = 15.0


def _split_at_silence_energy(
    audio: np.ndarray,
    sr: int = 16000,
    *,
    energy_split_rms: float = 0.003,
    energy_min_silence_ms: int = 120,
) -> list[np.ndarray]:
    max_seg = int(_VAD_MAX_SEG_SEC * sr)
    min_seg = int(_VAD_MIN_SEG_SEC * sr)
    if len(audio) <= max_seg:
        return [audio]

    frame_len = int(_VAD_FRAME_MS * sr / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [audio]

    framed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    threshold = energy_split_rms
    is_silence = rms < threshold
    min_run = max(1, int(energy_min_silence_ms) // _VAD_FRAME_MS)

    cut_candidates: list[int] = []
    run_start: Optional[int] = None
    for i, silent in enumerate(is_silence):
        if silent:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                cut_candidates.append(((run_start + i) // 2) * frame_len)
            run_start = None
    if run_start is not None and n_frames - run_start >= min_run:
        cut_candidates.append(((run_start + n_frames) // 2) * frame_len)
    cand = np.array(cut_candidates, dtype=np.int64)

    cuts = [0]
    while len(audio) - cuts[-1] > max_seg:
        target = cuts[-1] + max_seg
        lo = cuts[-1] + min_seg
        hi = target
        mask = (cand >= lo) & (cand <= hi)
        if mask.any():
            pick = int(cand[mask][np.argmax(cand[mask])])
        else:
            pick = int(target)
        cuts.append(pick)
    cuts.append(len(audio))

    # Merge mid-fragments <1s into the previous segment to avoid model bailout
    min_frag = int(1.0 * sr)
    min_tail = int(1.5 * sr)
    i = 1
    while i < len(cuts) - 1:
        if (cuts[i + 1] - cuts[i]) < min_frag:
            cuts.pop(i)
        else:
            i += 1
    while len(cuts) >= 3 and (cuts[-1] - cuts[-2]) < min_tail:
        cuts.pop(-2)
    return [audio[cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)]


def _float_to_wav_bytes(samples: np.ndarray, sr: int = 16000) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _wav_to_float(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio
    ratio = 16000 / sr
    new_len = int(len(audio) * ratio)
    idx = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def _to_str(value) -> str:
    """rkvoice-stream's inner.finalize() returns a structured dict, not a
    plain string. Unwrap the canonical 'text' field recursively (and tolerate
    plain str / None inputs)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _to_str(value.get("text", ""))
    if value is None:
        return ""
    return str(value)


def _clean_segment_text(text) -> str:
    """Drop model-bailout placeholders. Tolerates dict/str/None inputs."""
    s = _to_str(text)
    if not s:
        return ""
    stripped = s.strip()
    if stripped in _PLACEHOLDER_OUTPUTS:
        return ""
    # Some bailouts pad with whitespace/newlines around the placeholder
    if stripped.rstrip("。：:.\n ") in _PLACEHOLDER_OUTPUTS:
        return ""
    return stripped


def _join_segments(texts: list[str], language: str) -> str:
    texts = [t for t in texts if t]
    if not texts:
        return ""
    if len(texts) > 1:
        # Trim trailing CJK/Latin punctuation off all-but-last segments
        trail = "。，、！？；,.!?;"
        texts = [t.rstrip(trail).rstrip() for t in texts[:-1]] + [texts[-1]]
    cjk = {"Chinese", "Japanese", "Korean", "Cantonese", "zh", "ja", "ko"}
    sep = "" if (language in cjk or any(language.startswith(p) for p in ("zh", "ja", "ko"))) else " "
    return sep.join(texts).strip()


# ---------------------------------------------------------------------------
# Stream adapter
# ---------------------------------------------------------------------------


class _UnloadRaceError(RuntimeError):
    """Raised by the stream adapter when its backing inner has been unloaded
    mid-stream. Distinguishes "known unload race" from native RuntimeError
    surfaced by rkvoice_stream itself."""


class _RKASRStreamAdapter(ASRStream):
    """Forwards accept_waveform to the inner stream for partial emission, but
    intercepts finalize: if the accumulated audio is longer than the long-audio
    threshold, segment + per-segment transcribe instead of trusting the inner's
    sliding-window decoder (which snowballs garbage past ~10s)."""

    def __init__(self, inner, backend: "RKASRBackend", language: str = "auto"):
        self._inner = inner
        self._backend = backend
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._sample_rate = 16000

    # Every inner/backend._inner deref guards against the unload race window —
    # manager hard-closes the WS on profile swap, but a stream task in flight
    # may still touch the adapter one more time. Raise a dedicated
    # _UnloadRaceError (subclass of RuntimeError) so callers can distinguish
    # "known unload race" from a native RuntimeError surfaced by the
    # rkvoice_stream inner.
    def _live_inner(self):
        inner = self._inner
        if inner is None:
            raise _UnloadRaceError(
                "RK ASR stream adapter inner was unloaded; stream is dead"
            )
        return inner

    def _live_backend_inner(self):
        backend = self._backend
        inner = getattr(backend, "_inner", None) if backend is not None else None
        if inner is None:
            raise _UnloadRaceError(
                "RK ASR backend was unloaded; stream is dead"
            )
        return inner

    @property
    def _long_audio_threshold_s(self) -> float:
        backend = self._backend
        if backend is not None:
            return backend._config.long_audio_threshold_s
        return 15.0

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        inner = self._live_inner()
        self._sample_rate = sample_rate
        # Buffer for our own finalize path. Cheap copy; the underlying memory
        # is already a numpy array.
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._chunks.append(samples)
        inner.accept_waveform(sample_rate, samples)

    def finalize(self):
        inner = self._live_inner()
        if not self._chunks:
            return (inner.finalize() or ""), None
        audio = np.concatenate(self._chunks)
        dur_s = len(audio) / max(self._sample_rate, 1)
        if dur_s <= self._long_audio_threshold_s:
            text = inner.finalize() or ""
            return _clean_segment_text(text), None

        # Long path: segment + per-segment offline transcribe via inner.
        audio = _resample_to_16k(audio, self._sample_rate)
        backend = self._backend
        cfg = backend._config if backend is not None else RKASRConfig()
        try:
            segments = _split_at_silence_energy(
                audio,
                16000,
                energy_split_rms=cfg.energy_split_rms,
                energy_min_silence_ms=cfg.energy_min_silence_ms,
            )
        except Exception as e:
            logger.warning("RK ASR splitter failed (%.1fs audio): %s", dur_s, e)
            segments = [audio]

        texts: list[str] = []
        for seg in segments:
            if len(seg) / 16000 < 0.4:
                continue
            wav_bytes = _float_to_wav_bytes(seg, 16000)
            try:
                backend_inner = self._live_backend_inner()
                result = backend_inner.transcribe(
                    wav_bytes, language=self._language
                )
            except _UnloadRaceError:
                # Backend was unloaded mid-finalize — propagate so the
                # caller distinguishes this from a true transcribe error.
                # Native RuntimeError from transcribe() still falls through
                # to the broad except below and is logged-and-skipped.
                raise
            except Exception as e:
                logger.warning(
                    "RK ASR segment failed (%.1fs): %s", len(seg) / 16000, e
                )
                continue
            seg_text = _clean_segment_text(getattr(result, "text", "") or "")
            if seg_text:
                texts.append(seg_text)

        # Discard the inner's sliding-window result entirely — it's the
        # poisoned snowball. Some inners need their state torn down; trust
        # the GC and a fresh stream next call.
        return _join_segments(texts, self._language), None

    def prepare_finalize(self) -> None:
        self._live_inner().prepare_finalize()

    def cancel_and_finalize(self) -> None:
        self._live_inner().cancel_and_finalize()

    def get_partial(self) -> tuple[str, bool]:
        return self._live_inner().get_partial()


# ---------------------------------------------------------------------------
# Capability map + backend
# ---------------------------------------------------------------------------

_CAP_MAP = {
    "offline": ASRCapability.OFFLINE,
    "streaming": ASRCapability.STREAMING,
    "multi_language": ASRCapability.MULTI_LANGUAGE,
}


class RKASRBackend(ASRBackend):
    """Adapter around rkvoice_stream.create_asr().

    Backend selection is delegated to rkvoice-stream itself via the
    ``ASR_BACKEND`` env var (set in the rk3576/rk3588 profile / process env);
    that env is read by rkvoice-stream, not by this adapter.
    """

    @classmethod
    def concurrency_capability(cls, profile=None):
        """Declare concurrency for RK NPU ASR.

        rkvoice-stream owns the NPU lifecycle and runs single-session. NPU is
        an exclusive device.
        """
        return ConcurrencyCapability(
            supports_parallel=False,
            max_concurrent=1,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="external_managed",
        )

    def __init__(self, config: Optional[RKASRConfig] = None):
        from rkvoice_stream import create_asr

        self._config = config or RKASRConfig()
        self._inner = create_asr()
        self._platform = self._config.platform
        # Cache metadata at construction time so post-unload status queries
        # don't crash on ``self._inner is None``.
        try:
            self._cached_name = f"rk:{self._inner.name}"
        except Exception:
            self._cached_name = "rk:unknown"
        try:
            self._cached_sample_rate = int(self._inner.sample_rate)
        except Exception:
            self._cached_sample_rate = 0
        try:
            cached_caps: set[ASRCapability] = set()
            for cap in self._inner.capabilities:
                value = cap.value if hasattr(cap, "value") else str(cap)
                mapped = _CAP_MAP.get(value)
                if mapped is not None:
                    cached_caps.add(mapped)
            self._cached_capabilities = cached_caps
        except Exception:
            self._cached_capabilities = set()

    @property
    def name(self) -> str:
        if self._inner is None:
            return self._cached_name
        return f"rk:{self._inner.name}"

    @property
    def capabilities(self) -> set[ASRCapability]:
        if self._inner is None:
            return set(self._cached_capabilities)
        out: set[ASRCapability] = set()
        for cap in self._inner.capabilities:
            value = cap.value if hasattr(cap, "value") else str(cap)
            mapped = _CAP_MAP.get(value)
            if mapped is not None:
                out.add(mapped)
        return out

    @property
    def sample_rate(self) -> int:
        if self._inner is None:
            return self._cached_sample_rate
        return self._inner.sample_rate

    def is_ready(self) -> bool:
        if self._inner is None:
            return False
        return self._inner.is_ready()

    def preload(self) -> None:
        if self._inner is None:
            raise RuntimeError("RKASRBackend not loaded (was unloaded)")
        self._inner.preload()

    def unload(self) -> None:
        """Drop the rkvoice-stream inner backend handle. Idempotent.

        ``supports_hot_reload`` stays False — see RKTTSBackend.unload().
        """
        if self._inner is None:
            return
        try:
            self._inner = None
            import gc
            gc.collect()
        except Exception:
            logger.exception("RKASRBackend.unload failed; continuing")

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if self._inner is None:
            raise RuntimeError("RKASRBackend not loaded (was unloaded)")
        cfg = self._config
        # Long-audio guard: if WAV is longer than the threshold, split at
        # silence and run each segment through inner.transcribe() independently
        # (fresh session internally), then concatenate. Mirrors the streaming
        # finalize path.
        try:
            audio, sr = _wav_to_float(audio_bytes)
        except Exception:
            audio, sr = np.empty(0, dtype=np.float32), 16000
        dur_s = len(audio) / max(sr, 1)

        if dur_s <= cfg.long_audio_threshold_s:
            result = self._inner.transcribe(audio_bytes, language=language)
            text = _clean_segment_text(getattr(result, "text", "") or "")
            meta = getattr(result, "meta", {}) or {}
            return TranscriptionResult(text=text, language=result.language, meta=meta)

        audio = _resample_to_16k(audio, sr)
        try:
            segments = _split_at_silence_energy(
                audio,
                16000,
                energy_split_rms=cfg.energy_split_rms,
                energy_min_silence_ms=cfg.energy_min_silence_ms,
            )
        except Exception as e:
            logger.warning("RK ASR splitter failed offline (%.1fs): %s", dur_s, e)
            segments = [audio]

        texts: list[str] = []
        last_lang = language
        for seg in segments:
            if len(seg) / 16000 < 0.4:
                continue
            wav_seg = _float_to_wav_bytes(seg, 16000)
            try:
                result = self._inner.transcribe(wav_seg, language=language)
            except Exception as e:
                logger.warning(
                    "RK ASR offline segment failed (%.1fs): %s", len(seg) / 16000, e
                )
                continue
            seg_text = _clean_segment_text(getattr(result, "text", "") or "")
            if seg_text:
                texts.append(seg_text)
            last_lang = getattr(result, "language", last_lang) or last_lang

        return TranscriptionResult(
            text=_join_segments(texts, language),
            language=last_lang,
        )

    def create_stream(self, language: str = "auto") -> ASRStream:
        if self._inner is None:
            raise RuntimeError("RKASRBackend not loaded (was unloaded)")
        return _RKASRStreamAdapter(
            self._inner.create_stream(language=language), self, language=language
        )
