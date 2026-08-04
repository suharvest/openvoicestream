"""TTS runtime overrides — operator-controlled defaults at the service level.

These overrides sit between the per-request payload and the backend's intrinsic
defaults. They let an admin change the default speaker / speed / pitch without
rebuilding the backend or restarting the service.

Priority (first wins) when synthesising a request::

    request value > runtime override > backend / speaker-table default

The store is in-process only (not persisted). For persisted defaults use the
profile system. PR4 is responsible for wiring ``merge_tts_request_kwargs`` into
the actual ``/tts`` and ``/v2v`` paths — this module just owns the state and
the merge logic.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from server.core.tts_speakers import (
    default_speaker_id,
    speaker_spec_for_id,
    resolve_speaker_selector,
)

# Sentinel used in update_overrides to distinguish "argument omitted" from
# "argument explicitly set to None" (the latter clears the override).
_UNSET: Any = object()


@dataclass
class TTSRuntimeOverrides:
    default_speaker_id: int | None = None
    default_speed: float | None = None
    default_pitch_shift: float | None = None
    updated_at: float = 0.0


_overrides: TTSRuntimeOverrides = TTSRuntimeOverrides()
_lock = threading.RLock()


# Validation bounds
_SPEED_MIN = 0.25
_SPEED_MAX = 4.0
_PITCH_MIN = -24.0
_PITCH_MAX = 24.0


def get_overrides() -> TTSRuntimeOverrides:
    """Return a snapshot of the current runtime overrides."""
    with _lock:
        return TTSRuntimeOverrides(
            default_speaker_id=_overrides.default_speaker_id,
            default_speed=_overrides.default_speed,
            default_pitch_shift=_overrides.default_pitch_shift,
            updated_at=_overrides.updated_at,
        )


def reset_overrides() -> None:
    """Clear every runtime override."""
    global _overrides
    with _lock:
        _overrides = TTSRuntimeOverrides()


def update_overrides(
    *,
    speaker_id: Any = _UNSET,
    speed: Any = _UNSET,
    pitch_shift: Any = _UNSET,
    model_id: str | None = None,
) -> TTSRuntimeOverrides:
    """Patch one or more overrides.

    Each argument has three-valued semantics:

    * not passed → keep the existing value
    * ``None``   → clear this override (request falls back to backend default)
    * a value    → set this override (validated first)

    ``speaker_id`` is validated against the speaker table for ``model_id`` (if
    provided). Out-of-range ``speed`` / ``pitch_shift`` raise :class:`ValueError`.
    """
    with _lock:
        new_speaker = _overrides.default_speaker_id
        new_speed = _overrides.default_speed
        new_pitch = _overrides.default_pitch_shift

        if speaker_id is not _UNSET:
            if speaker_id is not None and model_id is not None:
                # Runtime/admin writes are strict and normalize aliases (e.g.
                # CustomVoice legacy 0 -> canonical 3065) before storing.
                new_speaker = validate_speaker_id(int(speaker_id), model_id=model_id)
            else:
                new_speaker = None if speaker_id is None else int(speaker_id)

        if speed is not _UNSET:
            if speed is not None:
                _validate_speed(float(speed))
                new_speed = float(speed)
            else:
                new_speed = None

        if pitch_shift is not _UNSET:
            if pitch_shift is not None:
                _validate_pitch_shift(float(pitch_shift))
                new_pitch = float(pitch_shift)
            else:
                new_pitch = None

        globals()["_overrides"] = TTSRuntimeOverrides(
            default_speaker_id=new_speaker,
            default_speed=new_speed,
            default_pitch_shift=new_pitch,
            updated_at=time.time(),
        )
        return get_overrides()


def validate_speaker_id(speaker_id: int | None, *, model_id: str) -> int | None:
    """Raise :class:`ValueError` if ``speaker_id`` is unknown for ``model_id``.

    Honest lookup that doesn't rely on the permissive
    ``OVS_TTS_ALLOW_UNMAPPED_SPEAKER_ID`` fallback — admin overrides should
    only accept ids that genuinely exist in the table.
    """
    if speaker_id is None:
        return None
    try:
        spec = resolve_speaker_selector(int(speaker_id), model_id)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if spec is None:
        raise ValueError(
            f"Unknown TTS speaker_id {speaker_id} for model {model_id!r}"
        )
    return spec.id


def _validate_speed(value: float) -> None:
    if not (_SPEED_MIN <= value <= _SPEED_MAX):
        raise ValueError(
            f"speed must be in [{_SPEED_MIN}, {_SPEED_MAX}]; got {value}"
        )


def _validate_pitch_shift(value: float) -> None:
    if not (_PITCH_MIN <= value <= _PITCH_MAX):
        raise ValueError(
            f"pitch_shift must be in [{_PITCH_MIN}, {_PITCH_MAX}]; got {value}"
        )


def merge_tts_request_kwargs(
    *,
    request_speaker_id: int | None,
    request_speed: float | None,
    request_pitch_shift: float | None,
    model_id: str,
) -> dict[str, object]:
    """Resolve final speaker_id / speed / pitch_shift for a TTS call.

    Returns a dict with exactly those three keys. The caller is responsible for
    spreading them into ``synthesize()`` (and translating speaker_id to backend
    kwargs via :func:`speaker_kwargs_for_id` if needed).
    """
    snap = get_overrides()

    if request_speaker_id is not None:
        request_spec = speaker_spec_for_id(int(request_speaker_id), model_id)
        speaker_id: int | None = request_spec.id if request_spec is not None else None
    elif snap.default_speaker_id is not None:
        override_spec = speaker_spec_for_id(snap.default_speaker_id, model_id)
        speaker_id = override_spec.id if override_spec is not None else None
    else:
        speaker_id = default_speaker_id(model_id)

    if request_speed is not None:
        speed: float | None = float(request_speed)
    else:
        speed = snap.default_speed  # may be None — backend uses its own default

    if request_pitch_shift is not None:
        pitch_shift: float | None = float(request_pitch_shift)
    else:
        pitch_shift = snap.default_pitch_shift

    return {
        "speaker_id": speaker_id,
        "speed": speed,
        "pitch_shift": pitch_shift,
    }
