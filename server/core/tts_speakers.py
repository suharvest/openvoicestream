"""TTS speaker registry — model-scoped.

Each ``model_id`` owns an independent speaker table. A ``SpeakerSpec`` stores
canonical data (label, payload); backends translate it to engine-specific
kwargs through ``speaker_kwargs_for_id()``.

Persistence
-----------
``OVS_TTS_SPEAKERS_FILE`` (default ``/opt/seeed-local-voice/data/speakers.json``)
holds runtime-registered embedding-type speakers. Preset speakers are hardcoded
in ``_PRESETS``. Load order: presets → file → ``OVS_TTS_SPEAKERS_JSON`` env var.
Later layers override earlier ones on id collision.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

SpeakerType = Literal["preset", "embedding"]


@dataclass(frozen=True)
class SpeakerSpec:
    """Canonical speaker record — backend-agnostic.

    *Preset* speakers carry a backend-native identifier string in ``payload``
    (e.g. ``"2301"`` for Qwen3, ``"52"`` for Kokoro).

    *Embedding* speakers carry a base64-encoded float32 speaker embedding in
    ``payload``, plus optional metadata about the extractor model.
    """

    id: int
    type: SpeakerType
    label: str = ""
    payload: str = ""  # preset → speaker string; embedding → b64 embedding
    meta: dict[str, object] | None = None

    @property
    def speaker_embedding_bytes(self) -> bytes:
        if self.type != "embedding":
            raise TypeError(f"SpeakerSpec id={self.id} is not an embedding")
        if not self.payload:
            raise ValueError(f"SpeakerSpec id={self.id} has empty embedding payload")
        return base64.b64decode(self.payload)


# ---------------------------------------------------------------------------
# Preset tables (one per model_id)
# ---------------------------------------------------------------------------

_QWEN3_PRESETS: dict[int, SpeakerSpec] = {
    0: SpeakerSpec(id=0, type="preset", label="Default", payload=""),
    2301: SpeakerSpec(id=2301, type="preset", label="Female 1", payload="2301"),
    2302: SpeakerSpec(id=2302, type="preset", label="Female 2", payload="2302"),
}


# Qwen3-CustomVoice ships 9 built-in speakers identified by numeric speaker_id.
# IDs come from the engines-nx/talker/config.json `speaker_id` field on the
# tensorrt-edge-llm CustomVoice spike (orin-nx). CustomVoice does NOT support
# voice cloning (no speaker_encoder, no embedding extractor) — registry holds
# preset-type entries only.
_QWEN3_CUSTOMVOICE_PRESETS: dict[int, SpeakerSpec] = {
    sid: SpeakerSpec(id=sid, type="preset", label=label, payload=str(sid))
    for sid, label in [
        (3065, "vivian"),
        (3061, "ryan"),
        (2861, "aiden"),
        (3066, "serena"),
        (2878, "dylan"),
        (2875, "eric"),
        (3010, "uncle_fu"),
        (2873, "ono_anna"),
        (2864, "sohee"),
    ]
}

# Authoritative speaker labels for 'kokoro-multi-lang-v1_0' (53 speakers, 0-52).
_KOKORO_LABELS = [
    "af_heart",         # 0  — default female (American)
    "af_bella",         # 1
    "af_nicole",        # 2
    "af_aoede",         # 3
    "af_kore",          # 4
    "af_sarah",         # 5
    "af_nova",          # 6
    "af_sky",           # 7
    "af_alloy",         # 8
    "af_jessica",       # 9
    "af_river",         # 10
    "af_miley",         # 11
    "af_seraph",        # 12
    "af_eve",           # 13
    "am_adam",          # 14 — first male (American)
    "am_echo",          # 15
    "am_eric",          # 16
    "am_fenrir",        # 17
    "am_liam",          # 18
    "am_michael",       # 19
    "am_onyx",          # 20
    "am_puck",          # 21
    "am_santa",         # 22
    "bf_alice",         # 23 — first female (British)
    "bf_emma",          # 24
    "bf_isabella",      # 25
    "bf_lily",          # 26
    "bm_daniel",        # 27 — first male (British)
    "bm_fable",         # 28
    "bm_george",        # 29
    "bm_lewis",         # 30
    "ef_dora",          # 31 — female (Spanish)
    "em_alex",          # 32 — male (Spanish)
    "ff_siwis",         # 33 — female (French)
    "hf_alpha",         # 34 — female (Hindi)
    "hf_beta",          # 35
    "hm_omega",         # 36 — male (Hindi)
    "hm_psi",           # 37
    "if_irina",         # 38 — female (Italian)
    "im_nikolai",       # 39 — male (Italian)
    "jf_alpha",         # 40 — female (Japanese)
    "jf_nezumi",        # 41
    "jf_gorudo",        # 42
    "jm_fLEMING",       # 43 — male (Japanese)
    "pf_dora",          # 44 — female (Portuguese)
    "pm_alex",          # 45 — male (Portuguese)
    "pm_santa",         # 46
    "zf_xiaobei",       # 47 — female (Mandarin)
    "zf_xiaoni",        # 48
    "zf_xiaoxiao",      # 49
    "zf_xiaoyi",        # 50
    "zm_yunyang",       # 51 — male (Mandarin)
    "zm_yunxi",         # 52 — male (Mandarin); patched to af_cute on Seeed images
]


def _kokoro_presets() -> dict[int, SpeakerSpec]:
    return {
        i: SpeakerSpec(id=i, type="preset", label=_KOKORO_LABELS[i], payload=str(i))
        for i in range(len(_KOKORO_LABELS))
    }


_SINGLE_SPEAKER: dict[int, SpeakerSpec] = {
    0: SpeakerSpec(id=0, type="preset", label="Default", payload="0"),
}

# SparkTTS interprets the generic ``speaker`` string as
# ``<gender>_<pitch>_<speed>``. Without a model-scoped entry, the registry's
# permissive unknown-id fallback turns the API default id=0 into speaker="0",
# which Spark parses as pitch=very_low. That prompt can terminate before all
# 32 required global tokens are emitted. Keep id=0 for API compatibility while
# mapping it to the backend's actual intrinsic defaults.
_SPARKTTS_PRESETS: dict[int, SpeakerSpec] = {
    0: SpeakerSpec(
        id=0,
        type="preset",
        label="Female · moderate pitch · moderate speed",
        payload="female_moderate_moderate",
    ),
}

_PRESETS: dict[str, dict[int, SpeakerSpec]] = {
    "qwen3-tts": _QWEN3_PRESETS,
    "qwen3-tts-customvoice": _QWEN3_CUSTOMVOICE_PRESETS,
    "kokoro-multi-lang-v1_0": _kokoro_presets(),
    "matcha-icefall-zh-en": _SINGLE_SPEAKER,
    "matcha-icefall-zh-en.rknn": _SINGLE_SPEAKER,
    "sherpa": _SINGLE_SPEAKER,
    "sparktts-0p5b": _SPARKTTS_PRESETS,
}

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------


def _speakers_file() -> str:
    return os.environ.get(
        "OVS_TTS_SPEAKERS_FILE",
        "/opt/seeed-local-voice/data/speakers.json",
    )


# ---------------------------------------------------------------------------
# In-memory cache + lock
# ---------------------------------------------------------------------------

_cache: dict[str, dict[int, SpeakerSpec]] = {}
_cache_lock = threading.Lock()
_file_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Loading / cache management
# ---------------------------------------------------------------------------


def _env_overrides() -> dict[int, SpeakerSpec]:
    """Parse OVS_TTS_SPEAKERS_JSON once and cache.

    Accepts a JSON object ``{"id": spec}`` or a list ``[{"id": ..., ...}]``.
    Overrides are global (applied to every model_id) — per-model scoping is
    handled by the layers above when a model-scoped format is needed.
    """
    raw = os.environ.get("OVS_TTS_SPEAKERS_JSON")
    if not raw:
        return {}
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("OVS_TTS_SPEAKERS_JSON is invalid JSON; ignoring")
        return {}

    entries: dict[int, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            entries[int(key)] = value
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "id" in item:
                entries[int(item["id"])] = item
    else:
        logger.warning("OVS_TTS_SPEAKERS_JSON must be an object or list; ignoring")
        return {}

    result: dict[int, SpeakerSpec] = {}
    for sid, value in entries.items():
        result[sid] = _parse_speaker_entry(sid, value)
    return result


def _parse_speaker_entry(sid: int, value: Any) -> SpeakerSpec:
    if isinstance(value, dict):
        typ = str(value.get("type", "preset")).strip().lower()
        label = str(value.get("label", value.get("name", "")))
        if typ == "embedding":
            payload = str(value.get("speaker_embedding_b64") or value.get("payload") or value.get("embedding_b64") or "")
            if not payload:
                raise ValueError(f"speaker_id {sid} embedding entry missing speaker_embedding_b64")
            meta = value.get("meta")
            return SpeakerSpec(id=sid, type="embedding", label=label, payload=payload, meta=meta)
        if typ != "preset":
            raise ValueError(f"speaker_id {sid} has unsupported type {typ!r}")
        payload = str(value.get("speaker", value.get("name", "")))
        return SpeakerSpec(id=sid, type="preset", label=label, payload=payload)
    return SpeakerSpec(id=sid, type="preset", label=str(value), payload=str(value))


# Parsed once at module load — env var is immutable after startup. Must come
# after _parse_speaker_entry's definition because _env_overrides() calls it.
_ENV_OVERRIDES: dict[int, SpeakerSpec] = _env_overrides()


def _load_speaker_map(model_id: str) -> dict[int, SpeakerSpec]:
    """Build the effective speaker table for *model_id*.

    Layers (later wins on id collision):
    1. Hardcoded presets
    2. speakers.json file
    3. OVS_TTS_SPEAKERS_JSON env var
    """
    with _cache_lock:
        if model_id in _cache:
            return _cache[model_id]

    mapping: dict[int, SpeakerSpec] = dict(_PRESETS.get(model_id, {}))

    # Layer 2: file persistence
    _load_file_speakers(mapping, model_id)

    # Layer 3: env overrides (global, parsed once at module load)
    for sid, spec in _ENV_OVERRIDES.items():
        mapping[sid] = spec

    with _cache_lock:
        _cache[model_id] = mapping
    return mapping


def _load_file_speakers(mapping: dict[int, SpeakerSpec], model_id: str) -> None:
    file_path = _speakers_file()
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read speakers file %s: %s", file_path, exc)
        return
    model_entries = data.get(model_id, {}) if isinstance(data, dict) else {}
    for key_str, entry in model_entries.items():
        sid = int(key_str)
        spec = _parse_speaker_entry(sid, entry)
        mapping[sid] = spec


def _invalidate_cache(model_id: str | None = None) -> None:
    with _cache_lock:
        if model_id is None:
            _cache.clear()
        else:
            _cache.pop(model_id, None)


def reload_speakers() -> None:
    """Force-reload all speaker tables (useful after external file changes)."""
    _invalidate_cache()


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------


def speaker_spec_for_id(speaker_id: int | None, model_id: str) -> SpeakerSpec | None:
    """Look up a single speaker by id. Returns ``None`` if not found."""
    if speaker_id is None:
        return None
    sid = int(speaker_id)
    mapping = _load_speaker_map(model_id)
    if sid in mapping:
        return mapping[sid]
    if os.environ.get("OVS_TTS_ALLOW_UNMAPPED_SPEAKER_ID", "1").lower() in ("0", "false", "no", "off"):
        raise ValueError(f"Unknown TTS speaker_id {sid} for model {model_id!r}")
    return SpeakerSpec(id=sid, type="preset", label=str(sid), payload=str(sid))


def speaker_kwargs_for_id(speaker_id: int | None, model_id: str) -> dict[str, object]:
    """Translate *speaker_id* into backend-ready kwargs.

    Returns:
        For presets:  ``{"speaker_id": int, "speaker": str}``
        For embeddings: ``{"speaker_id": int, "speaker_embedding": bytes}``
        For unknown / None: ``{}``
    """
    spec = speaker_spec_for_id(speaker_id, model_id)
    if spec is None:
        return {}
    if spec.type == "embedding":
        return {
            "speaker_id": spec.id,
            "speaker_embedding": spec.speaker_embedding_bytes,
        }
    return {"speaker_id": spec.id, "speaker": spec.payload or ""}


def default_speaker_id(model_id: str) -> int:
    """Return the default speaker id for *model_id*.

    Resolution order:
    1. ``OVS_TTS_DEFAULT_SPEAKER_ID`` env var (or deprecated ``TTS_DEFAULT_SID``)
    2. Fallback: 0
    """
    raw = os.environ.get("OVS_TTS_DEFAULT_SPEAKER_ID") or os.environ.get("TTS_DEFAULT_SID")
    if raw is not None:
        try:
            sid = int(raw)
            if speaker_spec_for_id(sid, model_id) is not None:
                return sid
            logger.warning(
                "OVS_TTS_DEFAULT_SPEAKER_ID=%d not found in model %r; falling back to 0",
                sid, model_id,
            )
        except (ValueError, TypeError):
            logger.warning("Invalid OVS_TTS_DEFAULT_SPEAKER_ID=%r", raw)
    return 0


def available_speakers(model_id: str) -> list[dict[str, object]]:
    """List all speakers registered for *model_id*."""
    speakers: list[dict[str, object]] = []
    for sid, spec in sorted(_load_speaker_map(model_id).items()):
        item: dict[str, object] = {"id": sid, "type": spec.type, "label": spec.label}
        if spec.type == "preset":
            item["payload"] = spec.payload
        else:
            item["speaker_embedding_b64"] = spec.payload[:40] + "..." if len(spec.payload) > 40 else spec.payload
            if spec.meta:
                item["meta"] = spec.meta
        speakers.append(item)
    return speakers


# ---------------------------------------------------------------------------
# Registration / unregistration (embedding-type only)
# ---------------------------------------------------------------------------


def register_speaker(
    model_id: str,
    payload: str,
    label: str = "",
    meta: dict[str, object] | None = None,
    speaker_id: int | None = None,
) -> SpeakerSpec:
    """Register a new embedding-type speaker, persist to file.

    Args:
        model_id: Model scope to register under.
        payload: Base64-encoded float32 speaker embedding.
        label: Human-readable name.
        meta: Optional metadata (extractor model, dims, dtype, created, etc.).
        speaker_id: Explicit id to use; auto-assigned (max existing + 1, ≥ 10000) if None.

    Returns:
        The newly registered SpeakerSpec.
    """
    # Validate embedding payload early
    _validate_embedding_payload(payload)

    meta = dict(meta or {})
    meta.setdefault("model_id", model_id)
    meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    with _file_lock:
        # Re-read the file to get the ground-truth set of IDs under lock.
        file_path = _speakers_file()
        file_data: dict[str, dict[str, Any]] = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    file_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                file_data = {}
        if not isinstance(file_data, dict):
            file_data = {}

        existing_ids: set[int] = set(_PRESETS.get(model_id, {}).keys())
        for sid_str in (file_data.get(model_id) or {}):
            existing_ids.add(int(sid_str))

        if speaker_id is None:
            speaker_id = max(existing_ids, default=0) + 1
            speaker_id = max(speaker_id, 10000)
        else:
            sid = int(speaker_id)
            if sid in existing_ids:
                existing = speaker_spec_for_id(sid, model_id)
                if existing is not None and existing.type == "preset":
                    raise ValueError(
                        f"Cannot overwrite preset speaker {sid} for model {model_id!r}"
                    )
                logger.info("Overwriting existing embedding speaker %d for model %r", sid, model_id)
            speaker_id = sid

        spec = SpeakerSpec(
            id=speaker_id,
            type="embedding",
            label=label or f"voice-{speaker_id}",
            payload=payload,
            meta=meta,
        )

        file_data.setdefault(model_id, {})[str(speaker_id)] = {
            "type": "embedding",
            "label": spec.label,
            "speaker_embedding_b64": spec.payload,
            "meta": spec.meta,
        }
        _atomic_write_json(file_path, file_data)

    # Invalidate the cache — a concurrent _load_speaker_map that read
    # the stale file between our write and this invalidation could still
    # cache stale data, but the _invalidate here makes the window small
    # enough that the next read will almost certainly hit the fresh file.
    with _cache_lock:
        _cache.pop(model_id, None)

    logger.info(
        "Registered embedding speaker %d (%r) for model %r", speaker_id, spec.label, model_id
    )
    return spec


def _validate_embedding_payload(payload: str) -> None:
    """Validate a base64 speaker embedding before persistence.

    Raises ValueError on malformed or unreasonably large payloads.
    """
    max_bytes = 4 * 1024  # 4 KB of float32 = 1024-dim, well above typical 256-dim
    if len(payload) > max_bytes * 2:  # base64 expands ~4/3
        raise ValueError(f"speaker_embedding payload too large ({len(payload)} chars)")
    try:
        emb = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 speaker_embedding") from exc
    if len(emb) % 4 != 0:
        raise ValueError(
            f"speaker_embedding must be a multiple of 4 bytes (float32), got {len(emb)}"
        )
    if len(emb) == 0:
        raise ValueError("speaker_embedding must not be empty")


def register_speaker_from_embedding(
    model_id: str,
    embedding_bytes: bytes,
    label: str = "",
    meta: dict[str, object] | None = None,
    speaker_id: int | None = None,
) -> SpeakerSpec:
    """Convenience: register with raw embedding bytes instead of b64 payload."""
    return register_speaker(
        model_id=model_id,
        payload=base64.b64encode(embedding_bytes).decode("ascii"),
        label=label,
        meta=meta,
        speaker_id=speaker_id,
    )


def unregister_speaker(model_id: str, speaker_id: int) -> bool:
    """Delete an embedding-type speaker. Preset speakers cannot be deleted.

    Returns False if the speaker does not exist.
    """
    mapping = _load_speaker_map(model_id)
    if speaker_id not in mapping:
        return False
    spec = mapping[speaker_id]
    if spec.type != "embedding":
        raise ValueError(
            f"Cannot unregister preset speaker {speaker_id} for model {model_id!r}"
        )

    with _file_lock:
        file_path = _speakers_file()
        data: dict[str, Any] = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        if model_id in data and str(speaker_id) in data[model_id]:
            del data[model_id][str(speaker_id)]
            if not data[model_id]:
                del data[model_id]
            _atomic_write_json(file_path, data)

    with _cache_lock:
        if model_id in _cache:
            _cache[model_id].pop(speaker_id, None)

    logger.info(
        "Unregistered embedding speaker %d for model %r", speaker_id, model_id
    )
    return True


# ---------------------------------------------------------------------------
# Backend helper
# ---------------------------------------------------------------------------


def resolve_speaker_kwargs(model_id: str, *, allow_embedding: bool = True, **kwargs: object) -> dict[str, object]:
    """Unified entry point for backends.

    Extracts voice parameters from the keyword dict, resolves through the
    speaker registry for *model_id*, and returns backend-ready kwargs.

    Input priority (first wins):
    1. ``speaker_embedding`` — raw float32 bytes (direct voice clone)
    2. ``speaker_id`` — numeric id resolved through registry
    3. ``sid`` — deprecated alias for speaker_id

    When *allow_embedding* is False, a ``ValueError`` is raised if the
    resolved speaker is an embedding type. Backends that do not support
    voice cloning should pass ``allow_embedding=False``.

    Returns:
        Empty dict, or dict with one or more of:
        ``speaker_id``, ``speaker`` (for presets),
        ``speaker_embedding`` (for embeddings / direct clone).
    """
    # 1. Direct embedding bytes
    emb = kwargs.get("speaker_embedding")
    if emb is not None:
        if not allow_embedding:
            raise ValueError(f"Model {model_id!r} does not support voice clone embeddings")
        return {"speaker_embedding": emb}

    # 2. speaker_id (preferred) or sid (deprecated)
    sid = kwargs.get("speaker_id", kwargs.get("sid"))
    if sid is not None:
        result = speaker_kwargs_for_id(int(sid), model_id)
        if not allow_embedding and result.get("speaker_embedding"):
            raise ValueError(f"Model {model_id!r} does not support voice clone embeddings")
        return result

    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(file_path: str, data: object) -> None:
    """Write JSON atomically via temp file + rename."""
    from pathlib import Path

    parent = Path(file_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(parent), suffix=".json", prefix=".speakers-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_path, file_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
