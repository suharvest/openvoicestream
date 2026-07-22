"""SparkTTS clone voice enrollment + registry — OVS server layer (spec §4.4).

A SparkTTS clone voice is a *VoiceProfile* (spec §10): ``<voice_id>.json`` (routing /
metadata) + ``<voice_id>.npz`` (``global_ids`` int32[32], ``ref_semantic_ids`` int32[Tr],
``d_vector`` f32[1024]). Profiles live in ``SPARKTTS_VOICES_DIR`` — the SAME directory the
voxedge ``SparkTTSBackend`` voice registry scans. Registering a voice writes a pair there
and asks the live backend to ``reload()`` its registry so the next synth sees it.

Enrollment runs the reference-audio analysis chain (wav2vec2-XLSR-53 + BiCodec
semantic/global tokenizers). That chain is PyTorch + ~300M params and runs on a GPU
**host** (spec §3.2) — it is deliberately NOT on the Jetson hot path (device-side
self-enrollment is P4, out of scope). This module therefore supports two registration
inputs:

  1. ``register_from_profile_files`` — the caller already ran ``enroll_voice.py`` on a host
     and uploads the resulting ``.json`` + ``.npz``. Always available (no torch needed).
  2. ``enroll_from_audio`` — run the analysis chain in-process. Only works where the
     SparkTTS PyTorch stack + pretrained models are importable (host deployment). On a
     Jetson it raises ``EnrollmentUnavailable`` with a clear message pointing at (1).

No torch / numpy import at module load — both are imported lazily so importing this
module on a torch-less device is free.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class EnrollmentUnavailable(RuntimeError):
    """Raised when in-process audio→profile enrollment cannot run on this host
    (no SparkTTS PyTorch stack / pretrained models). Use profile-file upload."""


def voices_dir() -> str:
    """Resolve the clone VoiceProfile directory (shared with the backend registry)."""
    return os.environ.get(
        "SPARKTTS_VOICES_DIR",
        "/opt/seeed-local-voice/data/sparktts_voices",
    )


def _safe_id(voice_id: str) -> str:
    return voice_id.replace(":", "_").replace("/", "_")


_write_lock = threading.Lock()


# --------------------------------------------------------------------------- listing
def list_voices() -> list[dict]:
    """List clone voices from the active backend registry if available, else from disk."""
    reg = _live_registry()
    if reg is not None:
        return reg.list_voices()
    return _list_from_disk()


def _list_from_disk() -> list[dict]:
    d = voices_dir()
    out: list[dict] = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name), "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        out.append({
            "voice_id": j.get("voice_id") or os.path.splitext(name)[0],
            "type": "clone",
            "sample_rate": j.get("sample_rate", 16000),
            "has_ref_semantic": bool(j.get("ref_semantic_len")),
            "ref_semantic_len": j.get("ref_semantic_len", 0),
            "ref_text": j.get("ref_text"),
            "source_meta": j.get("source_meta"),
        })
    return out


# --------------------------------------------------------------------------- registry hook
def _live_registry():
    """Return the live SparkTTSBackend's VoiceRegistry, or None if not active."""
    try:
        from server.core import tts_service
        if not tts_service.is_ready():
            return None
        backend = tts_service.get_backend()
    except Exception:
        return None
    return getattr(backend, "voices", None)


def _reload_live_registry() -> Optional[int]:
    reg = _live_registry()
    if reg is None:
        return None
    try:
        return reg.reload()
    except Exception:
        logger.warning("VoiceRegistry reload failed", exc_info=True)
        return None


# --------------------------------------------------------------------------- register
def register_from_profile_files(
    json_bytes: bytes,
    npz_bytes: bytes,
    voice_id: Optional[str] = None,
) -> dict:
    """Persist a host-enrolled VoiceProfile pair into the voices dir + reload registry.

    The json is validated (must parse, carry 32 global_ids inline OR a matching npz) and
    rewritten with the canonical ``npz_file`` name so the on-disk pair is self-consistent.
    """
    import numpy as np  # lazy

    try:
        j = json.loads(json_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"voice profile json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(j, dict):
        raise ValueError("voice profile json must be an object")

    vid = voice_id or j.get("voice_id")
    if not vid:
        raise ValueError("voice_id missing (not in request and not in profile json)")
    j["voice_id"] = vid

    # Validate the npz carries 32 global_ids.
    import io
    with np.load(io.BytesIO(npz_bytes)) as npz:
        if "global_ids" not in npz:
            raise ValueError("npz missing 'global_ids'")
        g = npz["global_ids"].reshape(-1)
        if g.shape[0] != 32:
            raise ValueError(f"expected 32 global_ids, got {g.shape[0]}")

    d = voices_dir()
    os.makedirs(d, exist_ok=True)
    safe = _safe_id(vid)
    jpath = os.path.join(d, safe + ".json")
    npath = os.path.join(d, safe + ".npz")
    j["npz_file"] = os.path.basename(npath)

    with _write_lock:
        with open(npath, "wb") as f:
            f.write(npz_bytes)
        # write json last (it is the index the registry scans on)
        tmp = jpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        os.replace(tmp, jpath)

    n = _reload_live_registry()
    logger.info("Registered clone voice %r (reloaded registry → %s voices)", vid, n)
    return {"voice_id": vid, "json": jpath, "npz": npath, "registry_count": n}


def register_embedding_voice(
    voice_id: str,
    embedding_bytes: bytes,
    sample_rate: int = 24000,
    ref_text: Optional[str] = None,
    source_meta: Optional[dict] = None,
) -> dict:
    """Persist an *embedding-profile* clone voice (float32[1024] speaker vector).

    Unlike :func:`register_from_profile_files` (SparkTTS ``global_ids`` profiles,
    consumed by the voxedge ``VoiceRegistry``), this writes a lightweight profile
    the *server* resolves at synth time: ``<id>.json`` carries
    ``profile_type: "speaker_embedding"`` and ``<id>.npz`` stores the raw
    embedding under the key ``speaker_embedding``. The Qwen3 BASE backend has no
    voice registry — the server loads the npz on demand and forwards the raw
    ``speaker_embedding`` bytes to the backend (see ``load_embedding_voice`` and
    ``_request_voice_kwargs`` in server/main.py).

    Do NOT route these through ``register_from_profile_files``: it requires 32
    ``global_ids`` and would reject an embedding-only profile.
    """
    import io
    import numpy as np  # lazy

    if not voice_id:
        raise ValueError("voice_id is required")
    emb = np.frombuffer(embedding_bytes, dtype=np.float32)
    if emb.size == 0 or emb.nbytes % 4 != 0:
        raise ValueError("embedding must be a non-empty float32 byte vector")
    embedding_dim = int(emb.size)

    d = voices_dir()
    os.makedirs(d, exist_ok=True)
    safe = _safe_id(voice_id)
    jpath = os.path.join(d, safe + ".json")
    npath = os.path.join(d, safe + ".npz")

    buf = io.BytesIO()
    np.savez(buf, speaker_embedding=emb.astype(np.float32, copy=False))
    npz_bytes = buf.getvalue()

    j = {
        "voice_id": voice_id,
        "npz_file": os.path.basename(npath),
        "profile_type": "speaker_embedding",
        "embedding_dim": embedding_dim,
        "embedding_dtype": "float32",
        "sample_rate": sample_rate,
        "ref_text": ref_text,
        "source_meta": source_meta,
    }

    with _write_lock:
        with open(npath, "wb") as f:
            f.write(npz_bytes)
        tmp = jpath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        os.replace(tmp, jpath)

    # Best-effort registry reload — the SparkTTS VoiceRegistry ignores embedding
    # profiles (no global_ids), so this is a no-op there, but keeps other
    # registries in sync when present.
    n = _reload_live_registry()
    logger.info(
        "Registered embedding clone voice %r (dim=%d, registry→%s)",
        voice_id, embedding_dim, n,
    )
    return {
        "voice_id": voice_id,
        "json": jpath,
        "npz": npath,
        "profile_type": "speaker_embedding",
        "embedding_dim": embedding_dim,
        "registry_count": n,
    }


def load_embedding_voice(voice_id: str) -> Optional[bytes]:
    """Return raw float32 speaker-embedding bytes for an embedding-profile voice.

    Returns ``None`` when the id is unknown or the on-disk profile is not an
    embedding-profile (e.g. a SparkTTS ``global_ids`` clone) — callers then treat
    ``voice_id`` as an opaque backend-routed selector instead.
    """
    import io
    import numpy as np  # lazy

    if not voice_id:
        return None
    d = voices_dir()
    safe = _safe_id(voice_id)
    jpath = os.path.join(d, safe + ".json")
    if not os.path.isfile(jpath):
        return None
    try:
        with open(jpath, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return None
    if j.get("profile_type") != "speaker_embedding":
        return None
    npz_name = j.get("npz_file") or (safe + ".npz")
    npath = os.path.join(d, npz_name)
    if not os.path.isfile(npath):
        return None
    try:
        with open(npath, "rb") as f:
            npz_bytes = f.read()
        with np.load(io.BytesIO(npz_bytes)) as npz:
            key = "speaker_embedding" if "speaker_embedding" in npz else (
                "d_vector" if "d_vector" in npz else None
            )
            if key is None:
                return None
            emb = npz[key].reshape(-1).astype(np.float32, copy=False)
    except Exception:
        logger.warning("failed loading embedding voice %r", voice_id, exc_info=True)
        return None
    return emb.tobytes()


def enroll_from_audio(
    wav_bytes: bytes,
    voice_id: str,
    ref_text: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> dict:
    """Run the host analysis chain on ``wav_bytes`` → VoiceProfile, persist, reload.

    Imports the SparkTTS enrollment logic lazily. Raises :class:`EnrollmentUnavailable`
    when the PyTorch stack / pretrained models are not importable on this host (Jetson) —
    callers should fall back to ``register_from_profile_files`` with a host-generated pair.
    """
    import tempfile

    enroller = _load_enroller(model_dir)
    if enroller is None:
        raise EnrollmentUnavailable(
            "In-process SparkTTS enrollment is unavailable on this host (no torch / "
            "Spark-TTS stack). Run enroll_voice.py on a GPU host and POST the resulting "
            ".json + .npz to /tts/voices/profile instead."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(wav_bytes)
        wav_path = tf.name
    try:
        profile = enroller.enroll(wav_path, voice_id, ref_text)
        npz_path, json_path = enroller.write(profile, voices_dir())
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    n = _reload_live_registry()
    return {"voice_id": voice_id, "json": json_path, "npz": npz_path, "registry_count": n}


def _load_enroller(model_dir: Optional[str]):
    """Best-effort import of the host enrollment chain. Returns an object with
    ``.enroll(wav, voice_id, ref_text)`` and ``.write(profile, out_dir)`` or None."""
    import sys
    spark_repo = os.environ.get("SPARKTTS_SPIKE_DIR")
    if spark_repo and spark_repo not in sys.path:
        sys.path.insert(0, spark_repo)
    model_dir = model_dir or os.environ.get(
        "SPARKTTS_PRETRAINED_DIR", "pretrained_models/Spark-TTS-0.5B"
    )
    try:
        import enroll_voice  # the host tool (spec §10)
    except Exception:
        return None

    class _Adapter:
        def __init__(self):
            self._en = enroll_voice.Enroller(model_dir)

        def enroll(self, wav, voice_id, ref_text):
            return self._en.enroll(wav, voice_id, ref_text)

        def write(self, profile, out_dir):
            return enroll_voice.write_profile(profile, out_dir)

    try:
        return _Adapter()
    except Exception:
        logger.warning("SparkTTS enroller init failed", exc_info=True)
        return None


# --------------------------------------------------------------------------- delete
def delete_voice(voice_id: str) -> bool:
    """Delete a clone voice's json+npz pair from the voices dir + reload. False if absent."""
    d = voices_dir()
    safe = _safe_id(voice_id)
    jpath = os.path.join(d, safe + ".json")
    npath = os.path.join(d, safe + ".npz")
    existed = False
    with _write_lock:
        for p in (jpath, npath):
            if os.path.exists(p):
                existed = True
                try:
                    os.unlink(p)
                except OSError:
                    pass
    if existed:
        _reload_live_registry()
    return existed
