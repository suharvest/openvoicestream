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
import secrets
import shutil
import threading
import tempfile
from contextlib import contextmanager
from pathlib import Path
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
    """Return the legacy on-disk id while rejecting unsafe selectors.

    The public id is intentionally not made reversible: existing SparkTTS
    deployments use ``:``/``/`` → ``_``.  Because that mapping is lossy,
    callers must use :func:`_assert_safe_id_available` before replacing a
    profile; otherwise ``clone:a/b`` could silently overwrite
    ``clone:a:b``.
    """
    if not isinstance(voice_id, str):
        raise ValueError("voice_id must be a string")
    voice_id = voice_id.strip()
    if not voice_id:
        raise ValueError("voice_id must not be empty")
    if "\x00" in voice_id:
        raise ValueError("voice_id must not contain NUL")
    safe = voice_id.replace(":", "_").replace("/", "_")
    if safe in {"", ".", ".."} or safe.startswith("/"):
        raise ValueError(f"voice_id {voice_id!r} does not have a safe file id")
    return safe


_write_lock = threading.Lock()

# The two profile formats live in one directory for historical reasons, but
# they are not interchangeable.  Keep their model scope explicit on disk so a
# Qwen speaker embedding can never be picked up by Spark's VoiceRegistry (or
# vice versa) merely because both happen to use ``.json + .npz`` files.
SPARK_MODEL_ID = "sparktts-0p5b"
QWEN_BASE_MODEL_ID = "qwen3-tts-0.6b-base"
SPARK_PROFILE_TYPE = "voice_profile"
EMBEDDING_PROFILE_TYPE = "speaker_embedding"
_PROFILE_TYPES = {SPARK_PROFILE_TYPE, EMBEDDING_PROFILE_TYPE}
_TXN_PREFIX = ".sparktts-txn-"


@contextmanager
def _profile_lock():
    """Serialize profile mutations across threads *and* worker processes.

    The file lock is deliberately kept beside the profiles and is ignored by
    the JSON scanner.  ``fcntl`` is available on all supported Linux targets;
    the thread lock remains a safe fallback for platforms without it.
    """
    with _write_lock:
        lock_fd = None
        try:
            import fcntl

            lock_path = os.path.join(voices_dir(), ".sparktts-voices.lock")
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o660)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)


def _pair_paths(voice_id: str) -> tuple[str, str, str]:
    safe = _safe_id(voice_id)
    d = voices_dir()
    return safe, os.path.join(d, safe + ".json"), os.path.join(d, safe + ".npz")


def _canonical_model_id(model_id: object, default: str | None = None) -> str | None:
    """Canonicalise a persisted model id without inventing unknown aliases."""
    if model_id is None or (isinstance(model_id, str) and not model_id.strip()):
        return default
    value = str(model_id).strip()
    try:
        from server.core.tts_speakers import canonical_model_id

        value = canonical_model_id(value)
    except Exception:
        # This module is deliberately importable in a torch-less host tool; a
        # missing speaker registry must not make profile validation permissive.
        pass
    return value or default


def _normalise_compatible_models(
    model_id: str,
    compatible_models: object = None,
) -> list[str]:
    """Return a stable, non-empty canonical compatibility list."""
    values: list[object]
    if compatible_models is None:
        values = []
    elif isinstance(compatible_models, (list, tuple, set)):
        values = list(compatible_models)
    else:
        raise ValueError("compatible_models must be a list of model ids")
    out: list[str] = []
    for value in [model_id, *values]:
        canonical = _canonical_model_id(value)
        if not canonical:
            raise ValueError("compatible_models must contain non-empty model ids")
        if canonical not in out:
            out.append(canonical)
    return out


def _fsync_parent(path: str | os.PathLike[str]) -> None:
    """Durably persist a directory-entry change on POSIX filesystems."""
    parent = os.path.dirname(os.path.abspath(os.fspath(path))) or "."
    try:
        fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        logger.warning("failed fsync profile directory %s", parent, exc_info=True)
    finally:
        os.close(fd)


def _write_bytes_fsync(path: str | os.PathLike[str], payload: bytes) -> None:
    """Write a private transaction file and fsync its contents."""
    fd = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_fsync(path: str | os.PathLike[str], value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _write_bytes_fsync(path, payload)


def _transaction_paths(parent: Path, token: str) -> tuple[Path, Path]:
    txdir = parent / f"{_TXN_PREFIX}{token}"
    return txdir, parent / f"{_TXN_PREFIX}{token}.json"


def _json_pair_token(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    token = data.get("__pair_token")
    return str(token) if token is not None else None


def _npz_pair_token(npz) -> str | None:
    """Read the private generation marker from an open ``numpy.load`` file."""
    if "__pair_token" not in npz:
        return None
    try:
        value = npz["__pair_token"]
        if getattr(value, "ndim", 0) == 0:
            value = value.item()
        elif getattr(value, "size", 0) == 1:
            value = value.reshape(-1)[0].item()
        return str(value)
    except Exception:
        return None


def _pair_token_matches(json_data: object, npz_path: str) -> bool:
    """Return whether a pair is complete and from one generation.

    Legacy profiles predate generation markers and remain readable when both
    files are present.  New writes include a marker in both files; replacing
    one file and crashing before the second then makes the pair unreadable
    instead of exposing mixed generations.
    """
    if not os.path.isfile(npz_path):
        return False
    json_token = _json_pair_token(json_data)
    try:
        import numpy as np  # lazy

        with np.load(npz_path, allow_pickle=False) as npz:
            npz_token = _npz_pair_token(npz)
    except Exception:
        return False
    if json_token is None and npz_token is None:
        return True
    return json_token is not None and json_token == npz_token


def _load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _assert_safe_id_available(voice_id: str, safe: str, jpath: str, npath: str) -> None:
    """Reject lossy safe-id collisions and half-written existing profiles."""
    j_exists = os.path.lexists(jpath)
    n_exists = os.path.lexists(npath)
    if not j_exists and not n_exists:
        return
    if j_exists != n_exists:
        raise ValueError(
            f"cannot register {voice_id!r}: incomplete voice profile for safe id "
            f"{safe!r} already exists"
        )
    data = _load_json(jpath)
    existing = data.get("voice_id") if data else None
    if existing != voice_id:
        raise ValueError(
            f"safe-id collision: {voice_id!r} and {existing!r} map to {safe!r}"
        )
    if not data or data.get("npz_file", safe + ".npz") != os.path.basename(npath):
        raise ValueError(
            f"cannot replace {voice_id!r}: profile pair points outside canonical safe id"
        )
    if not _pair_token_matches(data, npath):
        raise ValueError(
            f"cannot replace {voice_id!r}: existing profile pair is half-written or invalid"
        )
    metadata = _profile_metadata(data, npath)
    if metadata is None:
        raise ValueError(
            f"cannot replace {voice_id!r}: profile metadata is missing or invalid; "
            "delete and re-register this legacy profile explicitly"
        )
    if metadata[3]:
        raise ValueError(
            f"cannot replace {voice_id!r}: legacy profile lacks canonical metadata; "
            "delete and re-register it to migrate"
        )


def _cleanup_temp(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("failed cleaning profile temp %s", path, exc_info=True)


def _files_equal(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as lf, right.open("rb") as rf:
            while True:
                lb = lf.read(1024 * 1024)
                rb = rf.read(1024 * 1024)
                if lb != rb:
                    return False
                if not lb:
                    return True
    except (OSError, ValueError):
        return False


def _pair_has_token(jpath: Path, npath: Path, token: str) -> bool:
    """Check a new-generation pair without treating a legacy pair as new."""
    data = _load_json(str(jpath))
    if _json_pair_token(data) != token:
        return False
    try:
        import numpy as np  # lazy

        with np.load(npath, allow_pickle=False) as npz:
            return _npz_pair_token(npz) == token
    except Exception:
        return False


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        pass


def _cleanup_transaction(txdir: Path, journal: Path) -> None:
    """Remove transaction material and persist the directory update."""
    try:
        _remove_path(txdir)
        _remove_path(journal)
        _fsync_parent(journal)
    except OSError:
        # A leftover journal is safe: the next reader retries recovery.  Do
        # not turn an already committed profile into a failed registration.
        logger.warning("failed cleaning SparkTTS transaction %s", journal, exc_info=True)


def _restore_from_file(backup: Path, destination: Path) -> None:
    """Restore a backup using an atomic same-filesystem rename.

    ``os.rename`` is intentionally used for recovery rather than ``replace``:
    tests (and some operators' fault-injection wrappers) monkeypatch
    ``os.replace`` to fail the second commit step.  POSIX rename has the same
    atomic overwrite semantics here and is followed by a parent fsync.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(destination.parent), prefix=".rollback-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f, backup.open("rb") as src:
            shutil.copyfileobj(src, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, destination)
        _fsync_parent(destination)
    finally:
        _cleanup_temp(str(tmp))


def _restore_or_remove_transaction(data: dict, journal: Path, txdir: Path) -> None:
    """Roll back an interrupted transaction, then remove its journal."""
    jpath = Path(str(data.get("json_path", "")))
    npath = Path(str(data.get("npz_path", "")))
    if not jpath.is_absolute() or not npath.is_absolute():
        raise ValueError("transaction paths must be absolute")
    old_present = bool(data.get("old_present"))
    old_json = txdir / "old.json"
    old_npz = txdir / "old.npz"
    if old_present:
        if not old_json.is_file() or not old_npz.is_file():
            raise ValueError("transaction is missing its old profile backup")
        _restore_from_file(old_json, jpath)
        _restore_from_file(old_npz, npath)
    else:
        for path in (jpath, npath):
            try:
                path.unlink()
                _fsync_parent(path)
            except FileNotFoundError:
                pass
    _cleanup_transaction(txdir, journal)


def _recover_transaction(journal: Path) -> None:
    """Complete or roll back one journal after a process/power interruption."""
    data = _load_json(str(journal))
    if data is None or data.get("schema_version") != 1:
        logger.warning("ignoring malformed SparkTTS transaction journal %s", journal)
        return
    token = str(data.get("token") or "")
    if not token or not journal.name.startswith(_TXN_PREFIX):
        logger.warning("ignoring invalid SparkTTS transaction journal %s", journal)
        return
    root = journal.parent.resolve()
    tx_name = str(data.get("tx_dir") or f"{_TXN_PREFIX}{token}")
    if Path(tx_name).name != tx_name or not tx_name.startswith(_TXN_PREFIX):
        logger.warning("ignoring unsafe SparkTTS transaction path %s", journal)
        return
    txdir = root / tx_name
    jpath = Path(str(data.get("json_path", "")))
    npath = Path(str(data.get("npz_path", "")))
    if (
        not txdir.is_dir()
        or not jpath.is_absolute()
        or not npath.is_absolute()
        or jpath.resolve().parent != root
        or npath.resolve().parent != root
    ):
        logger.warning("ignoring incomplete SparkTTS transaction %s", journal)
        return

    op = data.get("operation")
    if op == "write":
        new_complete = _pair_has_token(jpath, npath, token)
        old_present = bool(data.get("old_present"))
        old_json = txdir / "old.json"
        old_npz = txdir / "old.npz"
        old_complete = (
            _files_equal(jpath, old_json) and _files_equal(npath, old_npz)
            if old_present
            else not jpath.exists() and not npath.exists()
        )
        if new_complete:
            _cleanup_transaction(txdir, journal)
            return
        if old_complete:
            _cleanup_transaction(txdir, journal)
            return
        _restore_or_remove_transaction(data, journal, txdir)
        return

    if op == "delete":
        # Delete commits only when both canonical entries are gone.  Any mixed
        # state is restored from the durable old pair backup.
        if not jpath.exists() and not npath.exists():
            _cleanup_transaction(txdir, journal)
            return
        _restore_or_remove_transaction(data, journal, txdir)
        return

    logger.warning("unknown SparkTTS transaction operation %r in %s", op, journal)


def _recover_transactions_unlocked(root: Path) -> None:
    journals = sorted(root.glob(f"{_TXN_PREFIX}*.json"))
    for journal in journals:
        try:
            _recover_transaction(journal)
        except Exception:
            # Leave the journal in place.  Failing closed is preferable to
            # deleting a profile that may still be recoverable on retry.
            logger.error("failed recovering SparkTTS transaction %s", journal, exc_info=True)
    # A hard exit before the journal was made durable can leave a private
    # generation directory.  It cannot be visible to the registry, so remove
    # only orphaned transaction directories with no matching journal.
    for txdir in root.glob(f"{_TXN_PREFIX}*"):
        if not txdir.is_dir() or txdir.name.endswith(".json"):
            continue
        token = txdir.name[len(_TXN_PREFIX):]
        if not (root / f"{_TXN_PREFIX}{token}.json").exists():
            _remove_path(txdir)
            _fsync_parent(txdir)


def _recover_pending_transactions() -> None:
    root = Path(voices_dir())
    if not root.is_dir():
        return
    with _profile_lock():
        _recover_transactions_unlocked(root)


def _atomic_write_pair(
    jpath: str,
    npath: str,
    json_data: dict,
    npz_arrays: dict[str, object],
) -> None:
    """Durably publish a JSON/NPZ pair with journaled crash recovery.

    The registry still consumes the historical sibling files in the voices
    root, so a single directory-entry rename cannot replace both at once.  A
    private generation directory stores new and old bytes, and a fsynced
    journal records the transaction before either canonical entry changes.
    Readers recover the journal before scanning; a hard exit between the two
    replacements therefore leaves either the complete new pair or the exact
    old pair, never a mixed generation.
    """
    import numpy as np  # lazy

    parent = Path(jpath).parent
    parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    json_data = dict(json_data)
    json_data["__pair_token"] = token
    npz_arrays = dict(npz_arrays)
    npz_arrays.pop("__pair_token", None)
    npz_arrays["__pair_token"] = np.asarray(token)

    txdir, journal = _transaction_paths(parent, token)
    txdir.mkdir(mode=0o770)
    old_present = os.path.isfile(jpath) and os.path.isfile(npath)
    if old_present:
        _write_bytes_fsync(txdir / "old.json", Path(jpath).read_bytes())
        _write_bytes_fsync(txdir / "old.npz", Path(npath).read_bytes())
    # Build new payloads in the generation directory, not in the live scan
    # root.  np.savez needs a file object to avoid appending its own suffix.
    new_json = json.dumps(json_data, ensure_ascii=False, indent=2).encode("utf-8")
    _write_bytes_fsync(txdir / "new.json", new_json)
    with (txdir / "new.npz").open("wb") as f:
        np.savez(f, **npz_arrays)
        f.flush()
        os.fsync(f.fileno())
    _fsync_parent(txdir)
    _write_json_fsync(
        journal,
        {
            "schema_version": 1,
            "operation": "write",
            "token": token,
            "tx_dir": txdir.name,
            "json_path": os.path.abspath(jpath),
            "npz_path": os.path.abspath(npath),
            "old_present": old_present,
        },
    )
    _fsync_parent(journal)
    try:
        # Replace payload first and metadata second.  The journal makes the
        # ordering recoverable even if the process dies between these calls.
        os.replace(txdir / "new.npz", npath)
        _fsync_parent(npath)
        os.replace(txdir / "new.json", jpath)
        _fsync_parent(jpath)
    except BaseException:
        try:
            _recover_transaction(journal)
        except Exception:
            logger.error("failed immediate SparkTTS transaction rollback", exc_info=True)
        raise
    _cleanup_transaction(txdir, journal)


# --------------------------------------------------------------------------- listing
def list_voices(
    model_id: str | None = None,
    profile_type: str | None = None,
    compatible_model: str | None = None,
) -> list[dict]:
    """List complete, metadata-scoped profiles, optionally filtered by model.

    The live backend registry is intentionally not the source of truth here:
    it exposes only decoded Spark fields and cannot distinguish a Qwen
    embedding profile in the shared persistent directory.  The disk scanner
    validates the pair and its canonical metadata before returning it.
    """
    _recover_pending_transactions()
    return _list_from_disk(
        model_id=model_id,
        profile_type=profile_type,
        compatible_model=compatible_model,
    )


def _profile_metadata(j: dict, npath: str) -> tuple[str, str, list[str], bool] | None:
    """Read canonical profile metadata, with explicit legacy Spark migration.

    Old Spark files had no model/profile fields.  They are accepted only when
    their NPZ contains the unmistakable 32-element ``global_ids`` payload and
    are marked ``legacy_metadata=true``.  An old embedding-shaped file has no
    safe model inference and is therefore hidden until re-registered.
    """
    model_present = "model_id" in j
    type_present = "profile_type" in j
    compatible_present = "compatible_models" in j
    model = _canonical_model_id(j.get("model_id"))
    ptype = j.get("profile_type")
    compatible = j.get("compatible_models")
    if model_present and type_present and compatible_present:
        if not model or ptype not in _PROFILE_TYPES:
            return None
        if not _profile_payload_matches(str(ptype), npath):
            return None
        try:
            models = _normalise_compatible_models(model, compatible)
        except ValueError:
            return None
        if model not in models:
            return None
        return model, str(ptype), models, False
    # Partially written/newly malformed metadata fails closed; only a fully
    # legacy object gets the shape-based Spark migration path.
    if model_present or type_present or compatible_present:
        return None
    try:
        import numpy as np  # lazy

        with np.load(npath, allow_pickle=False) as npz:
            if "global_ids" not in npz or npz["global_ids"].reshape(-1).shape[0] != 32:
                return None
    except Exception:
        return None
    return SPARK_MODEL_ID, SPARK_PROFILE_TYPE, [SPARK_MODEL_ID], True


def _profile_payload_matches(profile_type: str, npath: str) -> bool:
    """Reject metadata that does not match the numeric payload shape."""
    try:
        import numpy as np  # lazy

        with np.load(npath, allow_pickle=False) as npz:
            if profile_type == SPARK_PROFILE_TYPE:
                return "global_ids" in npz and npz["global_ids"].reshape(-1).shape[0] == 32
            if profile_type == EMBEDDING_PROFILE_TYPE:
                return "speaker_embedding" in npz or "d_vector" in npz
    except Exception:
        return False
    return False


def _list_from_disk(
    *,
    model_id: str | None = None,
    profile_type: str | None = None,
    compatible_model: str | None = None,
) -> list[dict]:
    # Keep the private helper safe when called directly by capability builders
    # or tests; the public wrapper performs the same idempotent pass.
    _recover_pending_transactions()
    d = voices_dir()
    out: list[dict] = []
    if not os.path.isdir(d):
        return out
    requested_model = _canonical_model_id(model_id)
    requested_compatible = _canonical_model_id(compatible_model)
    for name in sorted(os.listdir(d)):
        if name.startswith(".") or not name.endswith(".json"):
            continue
        jpath = os.path.join(d, name)
        j = _load_json(jpath)
        if j is None:
            continue
        safe = os.path.splitext(name)[0]
        npz_name = j.get("npz_file") or (safe + ".npz")
        # Registry discovery must never advertise a JSON-only or mixed-
        # generation profile left by an interrupted replacement.
        if not isinstance(npz_name, str) or os.path.basename(npz_name) != npz_name:
            continue
        npath = os.path.join(d, npz_name)
        if not _pair_token_matches(j, npath):
            continue
        metadata = _profile_metadata(j, npath)
        if metadata is None:
            continue
        profile_model, profile_type_value, compatible_models, legacy = metadata
        if requested_model and requested_model != profile_model:
            continue
        if profile_type and profile_type != profile_type_value:
            continue
        if requested_compatible and requested_compatible not in compatible_models:
            continue
        out.append({
            "voice_id": j.get("voice_id") or safe,
            "type": profile_type_value,
            "profile_type": profile_type_value,
            "model_id": profile_model,
            "compatible_models": compatible_models,
            "legacy_metadata": legacy,
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
    model_id: str | None = None,
    compatible_models: list[str] | None = None,
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

    # This endpoint consumes SparkTTS VoiceProfiles.  Keep the metadata in the
    # uploaded JSON but override a missing/legacy value with the canonical
    # active Spark model.  A profile explicitly belonging to another model is
    # rejected instead of becoming a cross-backend voice selector.
    requested_model = _canonical_model_id(model_id or j.get("model_id"), SPARK_MODEL_ID)
    if requested_model != SPARK_MODEL_ID:
        raise ValueError(
            f"SparkTTS profile model_id must be {SPARK_MODEL_ID!r}, got {requested_model!r}"
        )
    incoming_type = j.get("profile_type")
    if incoming_type not in (None, SPARK_PROFILE_TYPE):
        raise ValueError(
            f"SparkTTS profile_type must be {SPARK_PROFILE_TYPE!r}, got {incoming_type!r}"
        )
    incoming_compatible = compatible_models if compatible_models is not None else j.get("compatible_models")
    j["model_id"] = requested_model
    j["profile_type"] = SPARK_PROFILE_TYPE
    j["compatible_models"] = _normalise_compatible_models(requested_model, incoming_compatible)

    # Validate the npz carries 32 global_ids and materialise its arrays.  The
    # arrays are written back with a generation marker so JSON/NPZ replacement
    # cannot expose a mixed pair after a crash.
    import io
    with np.load(io.BytesIO(npz_bytes)) as npz:
        if "global_ids" not in npz:
            raise ValueError("npz missing 'global_ids'")
        g = npz["global_ids"].reshape(-1)
        if g.shape[0] != 32:
            raise ValueError(f"expected 32 global_ids, got {g.shape[0]}")
        arrays = {key: npz[key] for key in npz.files if key != "__pair_token"}

    safe, jpath, npath = _pair_paths(vid)
    j["npz_file"] = os.path.basename(npath)

    with _profile_lock():
        _recover_transactions_unlocked(Path(voices_dir()))
        _assert_safe_id_available(vid, safe, jpath, npath)
        _atomic_write_pair(jpath, npath, j, arrays)

    n = _reload_live_registry()
    logger.info("Registered clone voice %r (reloaded registry → %s voices)", vid, n)
    return {"voice_id": vid, "json": jpath, "npz": npath, "registry_count": n}


def register_embedding_voice(
    voice_id: str,
    embedding_bytes: bytes,
    sample_rate: int = 24000,
    ref_text: Optional[str] = None,
    source_meta: Optional[dict] = None,
    model_id: str | None = None,
    compatible_models: list[str] | None = None,
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
    import numpy as np  # lazy

    if not voice_id:
        raise ValueError("voice_id is required")
    emb = np.frombuffer(embedding_bytes, dtype=np.float32)
    if emb.size == 0 or emb.nbytes % 4 != 0:
        raise ValueError("embedding must be a non-empty float32 byte vector")
    embedding_dim = int(emb.size)

    safe, jpath, npath = _pair_paths(voice_id)

    canonical_model = _canonical_model_id(model_id, QWEN_BASE_MODEL_ID)
    if not canonical_model:
        raise ValueError("model_id must not be empty")
    j = {
        "voice_id": voice_id,
        "npz_file": os.path.basename(npath),
        "profile_type": EMBEDDING_PROFILE_TYPE,
        "model_id": canonical_model,
        "compatible_models": _normalise_compatible_models(canonical_model, compatible_models),
        "embedding_dim": embedding_dim,
        "embedding_dtype": "float32",
        "sample_rate": sample_rate,
        "ref_text": ref_text,
        "source_meta": source_meta,
    }

    with _profile_lock():
        _recover_transactions_unlocked(Path(voices_dir()))
        _assert_safe_id_available(voice_id, safe, jpath, npath)
        _atomic_write_pair(
            jpath,
            npath,
            j,
            {"speaker_embedding": emb.astype(np.float32, copy=False)},
        )

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
        "profile_type": EMBEDDING_PROFILE_TYPE,
        "model_id": canonical_model,
        "compatible_models": j["compatible_models"],
        "embedding_dim": embedding_dim,
        "registry_count": n,
    }


def load_embedding_voice(
    voice_id: str,
    model_id: str | None = QWEN_BASE_MODEL_ID,
) -> Optional[bytes]:
    """Return raw float32 speaker-embedding bytes for an embedding-profile voice.

    Returns ``None`` when the id is unknown or the on-disk profile is not an
    embedding-profile (e.g. a SparkTTS ``global_ids`` clone) — callers then treat
    ``voice_id`` as an opaque backend-routed selector instead.
    """
    import io
    import numpy as np  # lazy

    if not voice_id:
        return None
    _recover_pending_transactions()
    try:
        safe, jpath, npath = _pair_paths(voice_id)
    except ValueError:
        return None
    if not os.path.isfile(jpath):
        return None
    j = _load_json(jpath)
    if j is None or j.get("voice_id") != voice_id:
        return None
    metadata = _profile_metadata(j, npath)
    if metadata is None:
        return None
    profile_model, profile_type_value, compatible_models, _legacy = metadata
    requested_model = _canonical_model_id(model_id)
    if profile_type_value != EMBEDDING_PROFILE_TYPE:
        return None
    if requested_model and requested_model not in compatible_models:
        return None
    if profile_model != requested_model and requested_model:
        return None
    npz_name = j.get("npz_file") or (safe + ".npz")
    if not isinstance(npz_name, str) or os.path.basename(npz_name) != npz_name:
        return None
    canonical_npath = os.path.join(voices_dir(), npz_name)
    if canonical_npath != npath or not _pair_token_matches(j, npath):
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
        # The host enroller is allowed to choose its own serialization shape,
        # but it must never write directly into the live registry.  Stage the
        # pair in a private temporary directory, read it back, then route the
        # bytes through register_from_profile_files so collision checks,
        # generation tokens, the cross-process lock, and atomic commit are
        # identical for upload and in-process enrollment.
        with tempfile.TemporaryDirectory(prefix="sparktts-enroll-") as stage_dir:
            result = enroller.write(profile, stage_dir)
            npz_path, json_path = _staged_profile_paths(result, stage_dir)
            with open(json_path, "rb") as f:
                json_bytes = f.read()
            with open(npz_path, "rb") as f:
                npz_bytes = f.read()
        return register_from_profile_files(json_bytes, npz_bytes, voice_id=voice_id)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _staged_profile_paths(result, stage_dir: str) -> tuple[str, str]:
    """Normalize an enroller ``write`` result and keep it inside ``stage_dir``."""
    root = Path(stage_dir).resolve()
    paths: list[Path] = []
    if isinstance(result, (tuple, list)) and len(result) == 2:
        # The host helper historically returns (npz_path, json_path).
        paths = [Path(str(result[0])), Path(str(result[1]))]
    if not paths or any(not p.is_absolute() for p in paths):
        paths = sorted(root.iterdir())
    resolved = [p.resolve() for p in paths]
    if any(root not in p.parents for p in resolved):
        raise ValueError("SparkTTS enroller wrote a profile outside its staging directory")
    npz = next((p for p in resolved if p.suffix == ".npz"), None)
    js = next((p for p in resolved if p.suffix == ".json"), None)
    if npz is None or js is None:
        raise ValueError("SparkTTS enroller did not produce one .json and one .npz profile")
    return str(npz), str(js)


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
    """Delete a clone voice's JSON/NPZ pair and reload the live registry.

    ``False`` means neither canonical file existed.  Any unlink failure is
    propagated (and the files already removed in that attempt are restored),
    so an API caller can never receive a successful deletion for a partial
    operation.
    """
    safe, jpath, npath = _pair_paths(voice_id)
    with _profile_lock():
        _recover_transactions_unlocked(Path(voices_dir()))
        j_exists = os.path.lexists(jpath)
        n_exists = os.path.lexists(npath)
        if not j_exists and not n_exists:
            return False
        if j_exists != n_exists:
            raise ValueError(
                f"cannot delete {voice_id!r}: incomplete voice profile for safe id "
                f"{safe!r}"
            )

        if j_exists:
            data = _load_json(jpath)
            existing = data.get("voice_id") if data else None
            if existing != voice_id:
                raise ValueError(
                    f"safe-id collision: refusing to delete {voice_id!r}; "
                    f"profile {existing!r} occupies {safe!r}"
                )

        _atomic_delete_pair(jpath, npath)

    _reload_live_registry()
    return True


def _atomic_delete_pair(jpath: str, npath: str) -> None:
    """Journal and durably delete both canonical profile files."""
    parent = Path(jpath).parent
    token = secrets.token_hex(16)
    txdir, journal = _transaction_paths(parent, token)
    txdir.mkdir(mode=0o770)
    _write_bytes_fsync(txdir / "old.json", Path(jpath).read_bytes())
    _write_bytes_fsync(txdir / "old.npz", Path(npath).read_bytes())
    _fsync_parent(txdir)
    _write_json_fsync(
        journal,
        {
            "schema_version": 1,
            "operation": "delete",
            "token": token,
            "tx_dir": txdir.name,
            "json_path": os.path.abspath(jpath),
            "npz_path": os.path.abspath(npath),
            "old_present": True,
        },
    )
    _fsync_parent(journal)
    try:
        os.unlink(jpath)
        _fsync_parent(jpath)
        os.unlink(npath)
        _fsync_parent(npath)
    except BaseException:
        try:
            _recover_transaction(journal)
        except Exception:
            logger.error("failed immediate SparkTTS deletion rollback", exc_info=True)
        raise
    _cleanup_transaction(txdir, journal)
