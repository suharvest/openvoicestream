"""Speaker diarization orchestration — product-layer shim over voxedge.

The clustering kernel (``OnlineDiarizer`` / ``OfflineDiarizer`` /
``SpeakerSegment``) is pure numpy and env-free, living in
``voxedge.capabilities.diarization``. This product-layer module keeps the
deployment concerns, mirroring ``speaker_embedding.py``:

  * the ``OVS_DIARIZE`` feature flag (default off, query/config overridable),
  * session-scoped online-diarizer construction with tuning params sourced
    from env (leaf params are injected as env by the composition boot path),
  * offline audio → segments → embeddings → clustering for ``POST /diarize``.

Embeddings are always taken from the existing ``speaker_embedding`` shim
(``compute_embedding``) — this module never re-extracts vectors and never
touches the inference engine. Opt-in, default-OFF, never-raise. Enabling
diarization implicitly requires speaker embeddings (clustering needs vectors).
"""

from __future__ import annotations

import logging
import os
from typing import List, Mapping, Optional

from server.core.concurrency_capability import ConcurrencyCapability
from server.core.env_helpers import env_float, env_int, truthy

logger = logging.getLogger(__name__)

# Clustering kernel comes straight from voxedge (single source, env-free).
try:
    from voxedge.capabilities.diarization import (  # noqa: F401
        OfflineDiarizer,
        OnlineDiarizer,
        SpeakerSegment,
    )
    _KERNEL_OK = True
except Exception:  # voxedge optional at import time
    OfflineDiarizer = OnlineDiarizer = SpeakerSegment = None  # type: ignore
    _KERNEL_OK = False

# Embedding model metadata for the offline response envelope.
try:
    from voxedge.capabilities.speaker_embedding import SPEAKER_MODEL_NAME
except Exception:
    SPEAKER_MODEL_NAME = "campplus_sv_zh_en_3dspeaker"


def diarize_enabled() -> bool:
    """Global default, from ``OVS_DIARIZE`` (default off). Overridable per
    connection via ``?diarize=`` (/asr/stream) or the v2v ``config`` field.
    """
    return truthy(os.environ.get("OVS_DIARIZE", ""))


# ── tuning params (env, with leaf-friendly defaults from spec §8) ────────────

def _online_threshold() -> float:
    return env_float("OVS_DIARIZE_THRESHOLD", 0.55)


def _ema() -> float:
    return env_float("OVS_DIARIZE_EMA", 0.7)


def _max_speakers() -> int:
    return env_int("OVS_DIARIZE_MAX_SPEAKERS", 10)


def _offline_min_sim() -> float:
    return env_float("OVS_DIARIZE_MIN_SIM", 0.50)


def _min_segment_ms() -> int:
    # 600 (not 400): real-CAM++ validation showed energy-segmenter fragments
    # shorter than ~0.6s yield unreliable embeddings that inflate the blind
    # num_speakers estimate (e.g. a 2-speaker clip over-counted to 6). Spans
    # >=0.63s score >=0.73 cosine to their speaker centroid; everything that
    # broke was <=0.54s. A 600ms floor drops those fragments and restores
    # correct blind speaker counts (2/3/1) without touching the cosine
    # threshold (which is well-centered, sep gap 0.545). The segmenter now
    # prefers the production silero VAD (utterance-level spans, P5 done); this
    # floor still applies as a safety net under both the silero and energy
    # paths.
    return env_int("OVS_DIARIZE_MIN_SEGMENT_MS", 600)


# ── concurrency capability (folds into the session ceiling) ──────────────────

def _max_concurrent(env_map: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """Read ``OVS_DIARIZE_MAX_CONCURRENT``; **default None** (no fixed cap).

    Deliberately *not* using ``_env_int`` (whose default is an int): for the
    diarization concurrency ceiling an unset — or non-positive / unparseable —
    value means "do not constrain", so it is dropped from the ``min()``
    aggregate (treated as ``+inf``) rather than pinned to a number.
    """
    m = env_map if env_map is not None else os.environ
    raw = m.get("OVS_DIARIZE_MAX_CONCURRENT")
    if raw is None:
        return None
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def diarization_concurrency_capability(
    profile: Optional[Mapping[str, object]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[ConcurrencyCapability]:
    """Diarization's runtime concurrency descriptor, or ``None`` when it does
    not participate in the session ceiling.

    Returns ``None`` (does **not** constrain the ceiling) unless diarization is
    enabled — either globally via ``OVS_DIARIZE`` (``diarize_enabled()``) or by
    a profile that opts in (``profile["diarize"]`` truthy). Default-off, so the
    common path returns ``None`` and the ceiling is byte-identical to a deploy
    that never declared diarization at all.

    Design / cost model: the blind clustering kernel is pure numpy and its CPU
    cost is negligible. The real per-stream constraint is the *one CAM++ forward
    pass per speech segment*, which competes for CPU/accelerator. That ceiling
    is device-specific (real-CAM++ validation: Pi4 / RK3576 ≈ 3 concurrent
    streams, Pi5 ≈ 6), so we do **not** hard-limit by default
    (``max_concurrent=_max_concurrent()`` → ``None``). Operators pin the
    measured per-device limit via ``OVS_DIARIZE_MAX_CONCURRENT``.

    Never raises — any failure returns ``None`` so a malformed env/profile can
    never break (or silently tighten) the aggregate ceiling.
    """
    try:
        env_map = env if env is not None else os.environ
        enabled = truthy(env_map.get("OVS_DIARIZE", ""))
        if not enabled and isinstance(profile, Mapping):
            enabled = bool(profile.get("diarize"))
        if not enabled:
            return None

        trt = (env_map.get("DIAR_CAMPPLUS_ENGINE_FILE", "") or "").strip()
        vram = 90 if (trt and os.path.exists(trt)) else 120
        return ConcurrencyCapability(
            supports_parallel=True,
            max_concurrent=_max_concurrent(env_map),
            is_stateful=True,
            requires_exclusive_device=False,
            scaling_mode="per_call_isolated",
            vram_mb_per_slot=vram,
        )
    except Exception:
        return None


def make_session_diarizer():
    """Construct a fresh per-connection ``OnlineDiarizer``, or None.

    One instance per streaming session — it holds the running cluster
    centroids for that conversation. Never raises.
    """
    if not _KERNEL_OK:
        logger.warning("diarization kernel unavailable (voxedge missing); diarize is a no-op")
        return None
    try:
        return OnlineDiarizer(
            threshold=_online_threshold(),
            ema=_ema(),
            max_speakers=_max_speakers(),
        )
    except Exception:
        logger.exception("Failed to build OnlineDiarizer; diarize disabled for this session")
        return None


# ── serialization helpers (cross-service envelope) ──────────────────────────

def segment_to_dict(seg, include_embedding: bool = False) -> dict:
    """One ``SpeakerSegment`` → JSON-safe dict (spec §4.1/§4.2)."""
    d = {
        "start": round(float(seg.start), 3),
        "end": round(float(seg.end), 3),
        "speaker": seg.speaker,
        "confidence": round(float(seg.confidence), 3),
    }
    if include_embedding and getattr(seg, "embedding", None) is not None:
        try:
            from voxedge.capabilities.speaker_embedding import encode_embedding
            d["embedding_b64"] = encode_embedding(seg.embedding)
        except Exception:
            logger.exception("encode_embedding failed; omitting embedding from segment")
    return d


def _num_speakers(segments: List) -> int:
    return len({s.speaker for s in segments})


def summary_payload(diarizer) -> Optional[dict]:
    """Build a ``diarization_summary`` event from a session diarizer.

    Runs the kernel's offline ``relabel()`` to reconcile greedy online splits
    into globally consistent labels (spec §4.1). Returns None when there is
    nothing to summarize. Never raises.
    """
    if diarizer is None:
        return None
    try:
        segs = diarizer.relabel()
        if not segs:
            return None
        return {
            "type": "diarization_summary",
            "segments": [segment_to_dict(s) for s in segs],
            "num_speakers": _num_speakers(segs),
        }
    except Exception:
        logger.exception("diarization summary failed; skipping")
        return None


def diarize_response(segments: List, return_embeddings: bool = False) -> dict:
    """Build the ``POST /diarize`` JSON envelope (spec §4.2)."""
    dim = 0
    for s in segments:
        emb = getattr(s, "embedding", None)
        if emb is not None:
            try:
                dim = int(len(emb))
            except Exception:
                dim = 0
            break
    return {
        "num_speakers": _num_speakers(segments),
        "segments": [segment_to_dict(s, include_embedding=return_embeddings) for s in segments],
        "embedding_model": SPEAKER_MODEL_NAME,
        "dim": dim,
    }


# ── offline: audio → segments → embeddings → clustering ──────────────────────

def _segment_audio_silero(samples, sr: int, min_segment_ms: int):
    """Utterance-level speech spans via the production silero VAD.

    Same model the streaming path uses (``server.core.vad.SileroVADSession``),
    so ``POST /diarize`` and ``?diarize`` endpoint identically. Feeds the clip
    through the VAD one 16 ms window at a time, tracking sample position, and
    turns its ``SPEECH_START`` / ``SPEECH_END`` transitions into closed
    ``(start, end)`` spans. The trailing ``silence_ms`` that triggers an
    endpoint is trimmed back off each span end so boundaries hug the actual
    speech.

    Returns:
      * ``list[(start, end)]`` (possibly empty) on success — the caller treats
        this as authoritative and does NOT fall back, or
      * ``None`` when silero is unavailable (no onnxruntime / no model file) or
        the sample rate is unsupported, signalling the caller to fall back to
        the energy splitter.

    Never raises — any unexpected failure also returns ``None`` (fall back).
    """
    # silero (the bundled v5 ONNX) only runs at 16 kHz; defer anything else.
    if sr != 16000:
        return None

    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    n = samples.shape[0]
    if n == 0:
        return []

    try:
        from server.core.vad import SileroVADSession
    except Exception:
        return None

    # Tighter endpoint than the streaming default (400 ms): offline we want
    # crisp utterance boundaries, not conversational turn-taking latency.
    silence_ms = 300
    try:
        vad = SileroVADSession(sample_rate=sr, silence_ms=silence_ms)
    except Exception:
        # onnxruntime missing or model file absent (CI / CPU smoke) → fall back.
        return None

    try:
        window = vad.WINDOW_16K  # 256 samples = 16 ms at 16 kHz
        silence_s = silence_ms / 1000.0
        spans = []
        cur_start = None
        i = 0
        while i + window <= n:
            ev = vad.process(samples[i : i + window])
            if ev == SileroVADSession.SPEECH_START:
                cur_start = i / float(sr)
            elif ev == SileroVADSession.SPEECH_END:
                if cur_start is not None:
                    win_end_s = (i + window) / float(sr)
                    end_s = max(cur_start, win_end_s - silence_s)
                    spans.append((cur_start, end_s))
                    cur_start = None
            i += window
        # Speech still open at clip end → close it at the final sample.
        if cur_start is not None:
            spans.append((cur_start, n / float(sr)))
    except Exception:
        logger.exception("silero VAD segmentation failed; falling back to energy")
        return None

    # Same safety floor the energy path uses: drop sub-min_segment_ms scraps
    # whose embeddings are unreliable and inflate the blind speaker count.
    spans = [
        (s, e) for (s, e) in spans if (e - s) * 1000.0 >= min_segment_ms
    ]
    return spans


def _segment_audio_energy(samples, sr: int, min_segment_ms: int):
    """Dependency-free energy/silence splitter — the silero fallback.

    Used when the silero model is unavailable (CI / CPU smoke runs with no
    onnxruntime or no bundled ONNX). 30 ms frames, an adaptive energy gate,
    runs of speech frames grouped into segments and short fragments
    (< ``min_segment_ms``) dropped. Coarser endpoints than silero, but good
    enough to lay out the segment time-axis for blind clustering.
    """
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    n = samples.shape[0]
    if n == 0:
        return []

    frame = max(1, int(sr * 0.03))  # 30 ms frames
    n_frames = (n + frame - 1) // frame
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = samples[i * frame : (i + 1) * frame]
        energies[i] = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0

    peak = float(energies.max())
    if peak <= 0.0:
        return []
    # Adaptive gate: a fraction of peak energy, floored to ignore DC/noise.
    gate = max(0.05 * peak, 1e-4)
    speech = energies > gate

    spans = []
    i = 0
    while i < n_frames:
        if not speech[i]:
            i += 1
            continue
        j = i
        while j < n_frames and speech[j]:
            j += 1
        start_s = (i * frame) / float(sr)
        end_s = min(j * frame, n) / float(sr)
        if (end_s - start_s) * 1000.0 >= min_segment_ms:
            spans.append((start_s, end_s))
        i = j

    # If the whole clip was one continuous utterance under the min length,
    # still emit it so single-speaker short clips are not dropped.
    if not spans:
        spans.append((0.0, n / float(sr)))
    return spans


def _segment_audio(samples, sr: int, min_segment_ms: int):
    """Split a mono float32 waveform into speech spans → list of (start, end).

    Prefers the production silero VAD (``_segment_audio_silero``) — the same
    model the streaming ``?diarize`` path uses — for utterance-level spans that
    fix the over-segmentation root cause (energy fragments < ~0.6 s yield
    unreliable embeddings that inflate the blind ``num_speakers`` estimate).
    Falls back to the dependency-free energy splitter (``_segment_audio_energy``)
    when silero is unavailable (CI / CPU smoke: no onnxruntime or no model
    file). Never raises.
    """
    spans = _segment_audio_silero(samples, sr, min_segment_ms)
    if spans:
        return spans
    # silero unavailable (None) or found no qualifying speech ([]) → energy
    # splitter. It handles empty audio (→ []) and emits the whole clip as a
    # last resort for borderline single-speaker short audio.
    return _segment_audio_energy(samples, sr, min_segment_ms)


def diarize_audio(samples, sr: int, num_speakers: Optional[int] = None):
    """Offline blind diarization of a long, possibly multi-speaker clip.

    VAD/energy-segment → CAM++ embedding per segment (via the existing
    ``speaker_embedding`` shim) → ``OfflineDiarizer.cluster()``. Returns a
    time-ordered ``list[SpeakerSegment]`` (embeddings attached so callers may
    opt to return them). Never raises — returns ``[]`` on any failure or when
    the embedding model / kernel is unavailable.
    """
    if not _KERNEL_OK:
        logger.warning("diarization kernel unavailable; /diarize is a no-op")
        return []
    try:
        from server.core import speaker_embedding as _spk

        spans = _segment_audio(samples, sr, _min_segment_ms())
        if not spans:
            return []

        import numpy as np

        samples = np.asarray(samples, dtype=np.float32)
        items = []  # (embedding, start, end)
        for (start_s, end_s) in spans:
            a = int(start_s * sr)
            b = int(end_s * sr)
            slice_ = samples[a:b]
            if slice_.size == 0:
                continue
            emb = _spk.compute_embedding(slice_, sr)
            if emb is None:
                continue
            items.append((np.asarray(emb, dtype=np.float32), start_s, end_s))

        if not items:
            return []

        # cluster() attaches each segment's embedding, so ?return_embeddings=true
        # is honoured without re-matching segments back to their vectors.
        return OfflineDiarizer(
            min_sim=_offline_min_sim(),
            max_speakers=_max_speakers(),
        ).cluster(items, num_speakers=num_speakers)
    except Exception:
        logger.exception("diarize_audio failed; returning empty result")
        return []


# TODO(P3): identification (?identify=true). spec §4.3 keeps name-mapping a
# default-off consumer responsibility — OVS only does blind clustering here.
# A future hook would compare each cluster centroid against an injected voice
# registry and override ``speaker`` with a known name above a threshold,
# leaving unmatched clusters as anonymous ``spk_N``. Not implemented.
