"""Decoupled helpers for the voxedge TRT-Edge-LLM adapters.

adapted from app/backends/jetson/qwen3_asr.py (deleted; recovered from git
8ef061f~1) + app/core/tts_speakers.py (2026-05-30), dedup after registry
switch.

These are env-free reproductions of the helper functions the production
TRT-Edge-LLM backends imported from ``app.*``. voxedge must not import
``app.*`` (open-core split), so the necessary logic is reproduced here with
ZERO module-scope env reads and ZERO file I/O. The two env-tunable constants
the original energy splitter read (``ASR_ENERGY_SPLIT_RMS`` /
``ASR_ENERGY_MIN_SILENCE_MS``) become explicit parameters with identical
defaults.

The production offline-splitter import path
(``from app.backends.jetson.qwen3_asr import _split_at_silence_vad ...``)
already pointed at a DELETED module — the functions are reproduced here so the
voxedge ASR adapter's long-audio finalize path is self-contained.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Long-audio VAD split parameters (Qwen3-ASR emits premature '。'+EOS after
# ~6.5s of continuous speech; split longer audio at silence to stay under the
# safe boundary and avoid deterministic truncation.)
VAD_MAX_SEG_SEC = 4.5        # conservative — leaves margin below Bug A boundary
VAD_MIN_SEG_SEC = 0.5        # allow finer splits when silence is available
VAD_FRAME_MS = 20            # webrtcvad frame size (10/20/30 supported)
VAD_AGGRESSIVENESS = 2       # 0-3; 2 = balanced for mixed noise conditions
VAD_MIN_SILENCE_MS = 150     # minimum silence run to count as a cut candidate

# Energy-splitter defaults (were env reads ASR_ENERGY_SPLIT_RMS /
# ASR_ENERGY_MIN_SILENCE_MS in the deleted production module).
DEFAULT_ENERGY_SPLIT_RMS = 0.003
DEFAULT_ENERGY_MIN_SILENCE_MS = 80


def _split_at_silence_vad(audio: np.ndarray, sr: int = 16000) -> list[np.ndarray]:
    """Split a long audio into segments at natural silence points via webrtcvad.

    Greedy: walk forward, try to cut at the silence point closest to
    VAD_MAX_SEG_SEC from the last cut, within [MIN_SEG, MAX_SEG] window. Falls
    back to a hard cut at MAX_SEG if no silence is found.

    Imports ``webrtcvad`` lazily; callers must handle ``ImportError`` and fall
    back to :func:`_split_at_silence_energy`.
    """
    import webrtcvad

    max_seg = int(VAD_MAX_SEG_SEC * sr)
    min_seg = int(VAD_MIN_SEG_SEC * sr)

    if len(audio) <= max_seg:
        return [audio]

    frame_len = int(VAD_FRAME_MS * sr / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [audio]

    pcm16 = (np.clip(audio[:n_frames * frame_len], -1.0, 1.0) * 32767).astype(np.int16)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    is_speech = np.zeros(n_frames, dtype=bool)
    frame_bytes = frame_len * 2
    raw = pcm16.tobytes()
    for i in range(n_frames):
        is_speech[i] = vad.is_speech(raw[i * frame_bytes:(i + 1) * frame_bytes], sr)

    min_run = max(1, VAD_MIN_SILENCE_MS // VAD_FRAME_MS)
    cut_candidates = []
    run_start = None
    for i in range(n_frames):
        if not is_speech[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                mid = (run_start + i) // 2
                cut_candidates.append(mid * frame_len)
            run_start = None
    if run_start is not None and n_frames - run_start >= min_run:
        mid = (run_start + n_frames) // 2
        cut_candidates.append(mid * frame_len)
    cut_candidates = np.array(cut_candidates, dtype=np.int64)

    cuts = [0]
    while len(audio) - cuts[-1] > max_seg:
        target = cuts[-1] + max_seg
        lo = cuts[-1] + min_seg
        hi = target
        mask = (cut_candidates >= lo) & (cut_candidates <= hi)
        if mask.any():
            pick = int(cut_candidates[mask][np.argmax(cut_candidates[mask])])
        else:
            pick = int(target)
        cuts.append(pick)
    cuts.append(len(audio))

    min_frag = int(1.0 * sr)
    min_tail = int(2.0 * sr)
    i = 1
    while i < len(cuts) - 1:
        if (cuts[i + 1] - cuts[i]) < min_frag:
            cuts.pop(i)
        else:
            i += 1
    while len(cuts) >= 3 and (cuts[-1] - cuts[-2]) < min_tail:
        cuts.pop(-2)
    return [audio[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]


def _split_at_silence_energy(
    audio: np.ndarray,
    sr: int = 16000,
    *,
    split_rms: float = DEFAULT_ENERGY_SPLIT_RMS,
    min_silence_ms: int = DEFAULT_ENERGY_MIN_SILENCE_MS,
) -> list[np.ndarray]:
    """Dependency-free fallback splitter for generated TTS audio.

    Uses frame RMS to find silence gaps. The two tuning knobs (``split_rms`` /
    ``min_silence_ms``) were env reads in the production module; here they are
    explicit parameters with identical defaults.
    """
    max_seg = int(VAD_MAX_SEG_SEC * sr)
    min_seg = int(VAD_MIN_SEG_SEC * sr)
    if len(audio) <= max_seg:
        return [audio]

    frame_len = int(VAD_FRAME_MS * sr / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [audio]

    framed = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(framed * framed, axis=1))
    is_silence = rms < split_rms
    min_run = max(1, int(min_silence_ms) // VAD_FRAME_MS)

    cut_candidates = []
    run_start = None
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
    cut_candidates = np.array(cut_candidates, dtype=np.int64)

    cuts = [0]
    while len(audio) - cuts[-1] > max_seg:
        target = cuts[-1] + max_seg
        lo = cuts[-1] + min_seg
        hi = target
        mask = (cut_candidates >= lo) & (cut_candidates <= hi)
        if mask.any():
            pick = int(cut_candidates[mask][np.argmax(cut_candidates[mask])])
        else:
            pick = int(target)
        cuts.append(pick)
    cuts.append(len(audio))

    min_frag = int(1.0 * sr)
    min_tail = int(2.0 * sr)
    i = 1
    while i < len(cuts) - 1:
        if (cuts[i + 1] - cuts[i]) < min_frag:
            cuts.pop(i)
        else:
            i += 1
    while len(cuts) >= 3 and (cuts[-1] - cuts[-2]) < min_tail:
        cuts.pop(-2)
    return [audio[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]


# ── trimmed from app/core/tts_speakers.py:555-590 ────────────────────────────
# voxedge has no speaker registry / file / env, so the model-scoped preset
# resolution collapses to a passthrough of the numeric speaker_id. The
# downstream ``_add_speaker_request_fields`` handles both ``speaker_id`` and
# ``speaker`` keys, so a numeric id passes straight to the C++ worker.


def resolve_speaker_kwargs(
    model_id: str,
    *,
    allow_embedding: bool = True,
    **kwargs: object,
) -> dict[str, object]:
    """Env-free, registry-free speaker kwargs resolver.

    Input priority (first wins), mirroring app/core/tts_speakers.py:
    1. ``speaker_embedding`` — raw float32 bytes (direct voice clone).
    2. ``speaker_id`` — numeric id passed straight through.
    3. ``sid`` — deprecated alias for speaker_id.

    Returns ``{"speaker_embedding": bytes}``, ``{"speaker_id": int}``, or
    ``{}``. ``model_id`` is accepted for signature-compat with the production
    helper but is unused (no registry in voxedge).
    """
    emb = kwargs.get("speaker_embedding")
    if emb is not None:
        if not allow_embedding:
            raise ValueError(
                f"Model {model_id!r} does not support voice clone embeddings"
            )
        return {"speaker_embedding": emb}

    sid = kwargs.get("speaker_id", kwargs.get("sid"))
    if sid is not None:
        return {"speaker_id": int(sid)}

    return {}
