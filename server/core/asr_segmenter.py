"""Automatic segmentation for the offline (file-upload) ASR path.

Why this exists
---------------
Several offline engines have a **fixed** input tensor shape and silently drop
whatever does not fit. SenseVoice TRT/RKNN is the worst offender: its engine
takes ``(1, T_FIXED=344, LFR_DIM=560)``, and the preprocessor simply does
``sp_in[:T_FIXED]`` (``voxedge/backends/jetson/sensevoice_trt.py``). With a
10 ms fbank hop and LFR ``n=6`` (60 ms per stacked frame) minus 4 prompt
frames that is ~20.4 s of audio — everything past it is discarded with **no
error and no log line**, so the caller gets a normal HTTP 200 carrying a
truncated transcript. Measured on hardware: a 26.57 s clip and a 21.21 s clip
produced byte-identical output.

Quality also degrades well before the hard cut (English started swallowing
words around 10.65–12.15 s), so the target segment length is configurable and
defaults to a conservative value rather than the hard engine limit.

Design
------
``maybe_transcribe_segmented()`` is the single entry point. It returns ``None``
whenever segmentation must not apply (feature disabled, clip already short
enough, duration unknown) so the caller can keep its original single-shot
decode path completely untouched.

Segment boundaries are chosen by Silero VAD when possible so cuts land in
silence instead of mid-word. This mirrors the existing pseudo-streaming
implementation in
``third_party/rkvoice-stream/rkvoice_stream/backends/asr/sensevoice_sherpa.py``
(Silero VAD segmentation → per-segment offline decode) and reuses the same
kind of parameters (min silence 0.25 s, min speech 0.1 s).

When VAD is unavailable (no onnxruntime, no model file, non-16 kHz or
non-PCM16 input) the module falls back to fixed-length slices with a small
overlap at each boundary so a word straddling a cut is seen whole by at least
one segment.

Environment
-----------
``OVS_ASR_AUTO_SEGMENT``            0/false → disable, restore legacy behaviour
``OVS_ASR_MAX_SEGMENT_SECONDS``     target segment length (default 10.5)
``OVS_ASR_SEGMENT_OVERLAP_SECONDS`` fixed-slice overlap (default 0.4)
``OVS_ASR_SEGMENT_VAD``             ``silero`` (default) | ``webrtcvad`` | ``none``
"""

from __future__ import annotations

import io
import logging
import os
import wave
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from server.core.env_helpers import env_float, truthy

logger = logging.getLogger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────
# 10.5 s is the measured "safe for English too" length. Chinese stayed clean
# up to 15.45 s and the hard SenseVoice cut is ~20.4 s, so this leaves margin
# on every engine we ship.
DEFAULT_MAX_SEGMENT_SECONDS = 10.5
DEFAULT_OVERLAP_SECONDS = 0.4
# Below this a fixed-slice tail is not worth its own decode call; merge it into
# the previous segment instead.
_MIN_TAIL_SECONDS = 0.5

# VAD parameters copied from the rkvoice-stream reference implementation.
_VAD_MIN_SILENCE_S = 0.25
_VAD_MIN_SPEECH_S = 0.10
_VAD_PAD_S = 0.10

ENV_AUTO_SEGMENT = "OVS_ASR_AUTO_SEGMENT"
ENV_MAX_SEGMENT_SECONDS = "OVS_ASR_MAX_SEGMENT_SECONDS"
ENV_OVERLAP_SECONDS = "OVS_ASR_SEGMENT_OVERLAP_SECONDS"
ENV_SEGMENT_VAD = "OVS_ASR_SEGMENT_VAD"

# Backend-declared safe offline length, checked in order. A backend that knows
# its own fixed-shape ceiling can expose one of these instead of relying on env.
_BACKEND_LIMIT_ATTRS = (
    "max_segment_seconds",
    "max_offline_audio_seconds",
    "max_audio_seconds",
    "max_input_seconds",
)

# STOPGAP — name matching, pending a declared capability on ASRBackend.
#
# Audit of the 5 registered backends (server/core/asr_backend.py:178-184):
#
#   sensevoice_trt  NO internal split — truncates at T_FIXED=344 (~20.4 s)  → segment
#   paraformer_trt  splits internally, 400-frame chunks, and CARRIES CIF state
#                   (carry_weight/carry_embed) + decoder cache across chunks    → keep out
#   trt_edgellm     splits internally (offline_segment_enabled default True,
#                   threshold 6.0 s, webrtcvad→energy cascade)                  → keep out
#   rk:*            splits internally (energy-RMS to <=4.5 s, each segment an
#                   INDEPENDENT transcribe() on purpose — a shared session
#                   poisons the next segment's prefix)                          → keep out
#   sherpa_asr      no internal split, but the ONNX export is dynamic-shape, so
#                   it does not truncate                                        → keep out
#
# Note ``rk.asr`` reports ``rk:<inner>`` (rk/asr.py:507-511), i.e. literally
# "rk:sensevoice_rknn" when wrapping SenseVoice — substring-matching
# "sensevoice" would wrongly segment a backend that already segments itself.
# Hence exact names plus an explicit rk: prefix rule, never a loose substring.
#
# The right fix is a capability declared by each backend; see the module
# docstring. Until then an unknown backend is NOT segmented (behaviour is left
# unchanged) but an over-long clip on it logs a warning naming the risk.
_SEGMENT_REQUIRED_BACKENDS = {
    "sensevoice_trt": DEFAULT_MAX_SEGMENT_SECONDS,
}
_SELF_SEGMENTING_BACKENDS = ("paraformer_trt", "trt_edgellm", "sherpa_asr")
_SELF_SEGMENTING_PREFIXES = ("rk:",)


def _backend_name(backend) -> str:
    return str(getattr(backend, "name", "") or "").strip().lower()


def backend_chunks_internally(backend) -> bool:
    """True when the backend already handles over-long clips itself."""
    name = _backend_name(backend)
    return name in _SELF_SEGMENTING_BACKENDS or name.startswith(
        _SELF_SEGMENTING_PREFIXES
    )


def backend_declared_limit(backend) -> Optional[float]:
    """Known fixed-input ceiling for ``backend``; ``None`` when unclassified."""
    return _SEGMENT_REQUIRED_BACKENDS.get(_backend_name(backend))


# ── Result types ─────────────────────────────────────────────────────

@dataclass
class DecodedSegment:
    """One decoded slice of the original clip."""

    index: int
    start: float
    end: float
    text: str
    language: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SegmentedTranscription:
    """Aggregate of a segmented decode."""

    text: str
    language: Optional[str] = None
    segments: List[DecodedSegment] = field(default_factory=list)
    total_duration: float = 0.0
    strategy: str = "fixed"
    meta: dict = field(default_factory=dict)

    @property
    def segments_decoded(self) -> int:
        return len(self.segments)

    def as_meta(self) -> dict:
        """Extra response fields — additive only, never replaces text/language."""
        out = {
            "segments_decoded": self.segments_decoded,
            "segmentation": {
                "applied": True,
                "strategy": self.strategy,
                "audio_duration": round(self.total_duration, 3),
                "segments": [
                    {
                        "index": s.index,
                        "start": round(s.start, 3),
                        "end": round(s.end, 3),
                        "text": s.text,
                    }
                    for s in self.segments
                ],
            },
        }
        if self.meta:
            out["segmentation"].update(self.meta)
        return out


# ── Config ───────────────────────────────────────────────────────────

def auto_segment_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """True unless ``OVS_ASR_AUTO_SEGMENT`` is explicitly falsy."""
    src = env if env is not None else os.environ
    raw = src.get(ENV_AUTO_SEGMENT)
    if raw is None or str(raw).strip() == "":
        return True
    return truthy(raw)


def resolve_max_segment_seconds(
    backend=None, env: Optional[Mapping[str, str]] = None
) -> float:
    """Safe per-decode audio length for ``backend``.

    Precedence: explicit env override → backend-declared limit →
    0.0 for backends that chunk internally → default.

    Returning ``0.0`` disables segmentation for that backend.
    """
    src = env if env is not None else os.environ
    if src.get(ENV_MAX_SEGMENT_SECONDS) not in (None, ""):
        value = env_float(ENV_MAX_SEGMENT_SECONDS, DEFAULT_MAX_SEGMENT_SECONDS, env=src)
        if value and value > 0:
            return float(value)
    for attr in _BACKEND_LIMIT_ATTRS:
        value = getattr(backend, attr, None)
        try:
            value = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    if backend_chunks_internally(backend):
        return 0.0
    return DEFAULT_MAX_SEGMENT_SECONDS


def resolve_overlap_seconds(env: Optional[Mapping[str, str]] = None) -> float:
    src = env if env is not None else os.environ
    value = env_float(ENV_OVERLAP_SECONDS, DEFAULT_OVERLAP_SECONDS, env=src)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_OVERLAP_SECONDS
    return max(0.0, value)


def resolve_vad_backend(env: Optional[Mapping[str, str]] = None) -> str:
    src = env if env is not None else os.environ
    return (src.get(ENV_SEGMENT_VAD) or "silero").strip().lower()


# ── WAV probing / slicing ────────────────────────────────────────────

@dataclass
class WavClip:
    """Decoded WAV container metadata plus its raw frame payload."""

    sample_rate: int
    channels: int
    sampwidth: int
    n_frames: int
    frames: bytes

    @property
    def duration(self) -> float:
        return (self.n_frames / self.sample_rate) if self.sample_rate else 0.0

    def slice_wav(self, start_s: float, end_s: float) -> bytes:
        """Re-emit ``[start_s, end_s)`` as a standalone WAV byte string."""
        bytes_per_frame = self.channels * self.sampwidth
        i0 = max(0, int(round(start_s * self.sample_rate)))
        i1 = min(self.n_frames, int(round(end_s * self.sample_rate)))
        if i1 <= i0:
            i1 = min(self.n_frames, i0 + 1)
        payload = self.frames[i0 * bytes_per_frame : i1 * bytes_per_frame]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sampwidth)
            wf.setframerate(self.sample_rate)
            wf.writeframes(payload)
        return buf.getvalue()

    def mono_float32(self):
        """Mono float32 [-1, 1] view for the VAD, or ``None`` if unsupported."""
        if self.sampwidth != 2:
            return None
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy ships with the server
            return None
        pcm = np.frombuffer(self.frames, dtype="<i2")
        if self.channels > 1:
            usable = (pcm.size // self.channels) * self.channels
            pcm = pcm[:usable].reshape(-1, self.channels).mean(axis=1)
        return pcm.astype(np.float32) / 32768.0


def probe_wav(audio_bytes: bytes) -> Optional[WavClip]:
    """Parse a WAV payload, or ``None`` when it is not a readable WAV."""
    try:
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            rate = wf.getframerate()
            if not rate:
                return None
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            clip = WavClip(
                sample_rate=rate,
                channels=wf.getnchannels() or 1,
                sampwidth=wf.getsampwidth() or 2,
                n_frames=n_frames,
                frames=frames,
            )
    except Exception:
        return None
    if clip.n_frames <= 0 or not clip.frames:
        return None
    return clip


def wav_duration_seconds(audio_bytes: bytes) -> float:
    """Best-effort clip duration from a WAV header; 0.0 when undecodable."""
    clip = probe_wav(audio_bytes)
    return clip.duration if clip else 0.0


# ── Boundary planning ────────────────────────────────────────────────

def fixed_segment_bounds(
    duration: float,
    max_segment_seconds: float,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> List[Tuple[float, float]]:
    """Fixed-length slices with ``overlap_seconds`` of context at each cut."""
    if duration <= 0 or max_segment_seconds <= 0:
        return []
    if duration <= max_segment_seconds:
        return [(0.0, duration)]
    overlap = min(max(0.0, overlap_seconds), max_segment_seconds / 2)
    step = max_segment_seconds - overlap
    bounds: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration - 1e-9:
        end = min(duration, start + max_segment_seconds)
        bounds.append((start, end))
        if end >= duration - 1e-9:
            break
        start = start + step
    # A sliver of a tail costs a whole decode for almost no audio — fold it in.
    # Extend the previous slice to cover the tail, but pull its start forward so
    # the merged slice never exceeds max_segment_seconds: when that limit is a
    # backend's hard fixed-input ceiling, going over it would silently truncate
    # exactly the tail this fold is meant to preserve. The earlier slice still
    # overlaps the new start (it reaches prev_start + overlap), so no audio is
    # skipped.
    if len(bounds) >= 2 and (bounds[-1][1] - bounds[-1][0]) < _MIN_TAIL_SECONDS:
        prev_start = bounds[-2][0]
        tail_end = bounds[-1][1]
        merged_start = max(prev_start, tail_end - max_segment_seconds)
        bounds[-2] = (merged_start, tail_end)
        bounds.pop()
    return bounds


def _speech_regions_from_vad(
    samples,
    sample_rate: int,
    vad_session,
    window: int,
) -> List[Tuple[float, float]]:
    """Run a ``VADSession`` window-by-window and collect speech spans.

    One window per ``process()`` call: the session API returns at most a single
    transition per call, so feeding bigger chunks would drop events.
    """
    from server.core.vad import VADSession

    regions: List[Tuple[int, int]] = []
    start: Optional[int] = None
    i = 0
    total = len(samples)
    while i + window <= total:
        event = vad_session.process(samples[i : i + window])
        if event == VADSession.SPEECH_START:
            if start is None:
                start = i
        elif event == VADSession.SPEECH_END:
            if start is not None:
                regions.append((start, i + window))
                start = None
        i += window
    if start is not None:
        regions.append((start, total))

    pad = int(_VAD_PAD_S * sample_rate)
    min_speech = _VAD_MIN_SPEECH_S * sample_rate
    out: List[Tuple[float, float]] = []
    for s, e in regions:
        if (e - s) < min_speech:
            continue
        s = max(0, s - pad)
        e = min(total, e + pad)
        out.append((s / sample_rate, e / sample_rate))
    return out


def group_regions(
    regions: Sequence[Tuple[float, float]],
    max_segment_seconds: float,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> List[Tuple[float, float]]:
    """Pack VAD speech regions into segments of at most ``max_segment_seconds``.

    Adjacent regions are merged while they fit; a region longer than the budget
    on its own is split with ``fixed_segment_bounds`` (VAD found no silence
    inside it, so there is nowhere better to cut).
    """
    segments: List[Tuple[float, float]] = []
    cur_start: Optional[float] = None
    cur_end = 0.0
    for start, end in regions:
        span = end - start
        if span > max_segment_seconds:
            if cur_start is not None:
                segments.append((cur_start, cur_end))
                cur_start = None
            for s, e in fixed_segment_bounds(span, max_segment_seconds, overlap_seconds):
                segments.append((start + s, start + e))
            continue
        if cur_start is None:
            cur_start, cur_end = start, end
            continue
        if (end - cur_start) <= max_segment_seconds:
            cur_end = end
        else:
            segments.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        segments.append((cur_start, cur_end))
    return segments


def plan_segments(
    clip: WavClip,
    max_segment_seconds: float,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    vad_backend: str = "silero",
    vad_factory: Optional[Callable[[], object]] = None,
) -> Tuple[List[Tuple[float, float]], str]:
    """Return ``(bounds, strategy)`` for ``clip``.

    Tries VAD first ("vad"), falls back to fixed slices ("fixed" /
    "fixed_vad_unavailable" / "fixed_vad_empty") so a missing VAD model or a
    non-PCM16 container never blocks segmentation.
    """
    duration = clip.duration
    if duration <= max_segment_seconds:
        return [(0.0, duration)], "single"

    if vad_backend in ("none", "off", "disabled", ""):
        return fixed_segment_bounds(duration, max_segment_seconds, overlap_seconds), "fixed"

    samples = clip.mono_float32()
    if samples is None or clip.sample_rate != 16000:
        # SileroVADSession requires 16 kHz PCM16; resampling here would add a
        # dependency for no gain over overlapped fixed slices.
        logger.info(
            "asr_segmenter: VAD skipped (sample_rate=%d sampwidth=%d) — "
            "using fixed slices",
            clip.sample_rate,
            clip.sampwidth,
        )
        return (
            fixed_segment_bounds(duration, max_segment_seconds, overlap_seconds),
            "fixed_vad_unavailable",
        )

    try:
        if vad_factory is not None:
            session = vad_factory()
        else:
            from server.core.vad import create_vad

            session = create_vad(
                backend=vad_backend,
                sample_rate=clip.sample_rate,
                silence_ms=int(_VAD_MIN_SILENCE_S * 1000),
            )
        if session is None:
            raise RuntimeError("VAD factory returned None")
        window = int(getattr(session, "WINDOW_16K", 256) or 256)
        regions = _speech_regions_from_vad(samples, clip.sample_rate, session, window)
    except Exception as exc:
        logger.warning(
            "asr_segmenter: VAD unavailable (%s) — falling back to fixed "
            "%.2fs slices with %.2fs overlap",
            exc,
            max_segment_seconds,
            overlap_seconds,
        )
        return (
            fixed_segment_bounds(duration, max_segment_seconds, overlap_seconds),
            "fixed_vad_unavailable",
        )

    if not regions:
        logger.info(
            "asr_segmenter: VAD found no speech regions in %.2fs of audio — "
            "using fixed slices",
            duration,
        )
        return (
            fixed_segment_bounds(duration, max_segment_seconds, overlap_seconds),
            "fixed_vad_empty",
        )

    bounds = group_regions(regions, max_segment_seconds, overlap_seconds)
    if not bounds:
        return (
            fixed_segment_bounds(duration, max_segment_seconds, overlap_seconds),
            "fixed_vad_empty",
        )
    return bounds, "vad"


# ── Text joining ─────────────────────────────────────────────────────

_CJK_RANGES = (
    (0x3000, 0x303F),    # CJK punctuation
    (0x3040, 0x30FF),    # Hiragana + Katakana
    (0x3400, 0x4DBF),    # CJK ext A
    (0x4E00, 0x9FFF),    # CJK unified
    (0xF900, 0xFAFF),    # CJK compatibility
    (0xFF00, 0xFFEF),    # halfwidth/fullwidth forms
    (0x20000, 0x2FA1F),  # CJK ext B+
    (0xAC00, 0xD7AF),    # Hangul syllables
)


def is_cjk(ch: str) -> bool:
    """True for CJK/Kana/Hangul code points (no inter-word spacing)."""
    if not ch:
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


# Longest repeated run we will strip at a segment boundary. Slices overlap by
# DEFAULT_OVERLAP_SECONDS, so the duplicate is bounded by what fits in that
# window; the cap keeps an unlucky match from eating real text.
_MAX_OVERLAP_DEDUP_CHARS = 60


def strip_overlap_repeat(prev: str, cur: str) -> str:
    """Drop the leading part of ``cur`` that repeats the tail of ``prev``.

    Adjacent slices overlap on purpose so a word is not cut in half, but that
    means the audio in the overlap is decoded twice and the same words come back
    in both segment texts. Concatenating them verbatim duplicates those words.

    Finds the longest suffix of ``prev`` that is also a prefix of ``cur`` and
    removes it from ``cur``. For space-separated text the match must land on a
    word boundary, so "the boy" + "boyish" does not lose "ish".
    """
    if not prev or not cur:
        return cur
    limit = min(len(prev), len(cur), _MAX_OVERLAP_DEDUP_CHARS)
    for n in range(limit, 0, -1):
        head = cur[:n]
        if not prev.endswith(head):
            continue
        # CJK runs have no word delimiters, so a character-level match is the
        # only thing available and is safe. Anything else is word-based: the
        # match must start and end on a boundary, or a shared word fragment gets
        # swallowed ("called for the boy" + "boyish grin" must not lose "ish").
        if not all(is_cjk(ch) for ch in head):
            before = prev[:-n]
            after = cur[n:]
            if before and not before[-1].isspace():
                continue
            if after and not after[0].isspace():
                continue
        return cur[n:].lstrip()
    return cur


def join_segment_texts(parts: Sequence[str]) -> str:
    """Concatenate segment texts, inserting a space only where needed.

    No space when both characters straddling the boundary are CJK; a single
    space otherwise. Existing whitespace at either side is respected. Text that
    repeats across a slice overlap is dropped — see ``strip_overlap_repeat``.
    """
    out = ""
    for raw in parts:
        piece = (raw or "").strip()
        if not piece:
            continue
        if not out:
            out = piece
            continue
        piece = strip_overlap_repeat(out, piece)
        if not piece:
            continue
        left, right = out[-1], piece[0]
        if left.isspace() or right.isspace():
            out += piece
        elif is_cjk(left) and is_cjk(right):
            out += piece
        else:
            out += " " + piece
    return out


# ── Entry point ──────────────────────────────────────────────────────

def maybe_transcribe_segmented(
    backend,
    audio_bytes: bytes,
    language: str = "auto",
    env: Optional[Mapping[str, str]] = None,
    vad_factory: Optional[Callable[[], object]] = None,
) -> Optional[SegmentedTranscription]:
    """Segment-and-decode ``audio_bytes`` when it is too long for one pass.

    Returns ``None`` when the caller should keep its original single-shot
    ``backend.transcribe()`` path: segmentation disabled, clip within the safe
    length, or duration undecodable (a WARNING is logged in that last case —
    the engine may still truncate, which is exactly the failure this module
    exists to make visible).
    """
    src = env if env is not None else os.environ
    if not auto_segment_enabled(src):
        return None

    max_seconds = resolve_max_segment_seconds(backend, src)
    if max_seconds <= 0:
        return None

    clip = probe_wav(audio_bytes)
    if clip is None:
        logger.warning(
            "asr_segmenter: cannot read audio duration (not a decodable WAV, "
            "%d bytes) — auto-segmentation skipped. A fixed-shape backend may "
            "silently truncate this clip past its input limit; send WAV to get "
            "automatic segmentation.",
            len(audio_bytes),
        )
        return None

    duration = clip.duration
    if duration <= max_seconds:
        return None

    overlap = resolve_overlap_seconds(src)
    vad_backend = resolve_vad_backend(src)
    bounds, strategy = plan_segments(
        clip,
        max_segment_seconds=max_seconds,
        overlap_seconds=overlap,
        vad_backend=vad_backend,
        vad_factory=vad_factory,
    )
    if len(bounds) <= 1:
        return None

    logger.info(
        "asr_segmenter: audio %.2fs exceeds safe %.2fs for backend %r — "
        "decoding %d segments (strategy=%s, overlap=%.2fs): %s",
        duration,
        max_seconds,
        getattr(backend, "name", "?"),
        len(bounds),
        strategy,
        overlap,
        ", ".join(f"{s:.2f}-{e:.2f}s ({e - s:.2f}s)" for s, e in bounds),
    )

    decoded: List[DecodedSegment] = []
    languages: List[str] = []
    for idx, (start, end) in enumerate(bounds):
        chunk = clip.slice_wav(start, end)
        result = backend.transcribe(chunk, language=language)
        text = getattr(result, "text", "") or ""
        lang = getattr(result, "language", None)
        if lang:
            languages.append(lang)
        decoded.append(
            DecodedSegment(index=idx, start=start, end=end, text=text, language=lang)
        )
        logger.debug(
            "asr_segmenter: segment %d/%d %.2f-%.2fs -> %r",
            idx + 1,
            len(bounds),
            start,
            end,
            text,
        )

    joined = join_segment_texts([s.text for s in decoded])
    # Most frequent detected language wins; ties break toward the first seen.
    resolved_lang = None
    if languages:
        resolved_lang = max(dict.fromkeys(languages), key=languages.count)

    logger.info(
        "asr_segmenter: decoded %d segments, %d chars total (language=%s)",
        len(decoded),
        len(joined),
        resolved_lang,
    )
    return SegmentedTranscription(
        text=joined,
        language=resolved_lang,
        segments=decoded,
        total_duration=duration,
        strategy=strategy,
        meta={"max_segment_seconds": max_seconds, "overlap_seconds": overlap},
    )
