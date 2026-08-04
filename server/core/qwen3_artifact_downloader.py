"""Auto-download Qwen3 ASR/TTS engine artifacts from HuggingFace.

When a Jetson backend's ``preload()`` detects missing engine / config /
tokenizer files, this module fetches them via ``snapshot_download`` from
the repo declared in
``/opt/qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json``.

Behavior
--------
* Gated by ``OVS_AUTO_DOWNLOAD_ARTIFACTS`` env (default ``"1"`` = on).
  Set to ``"0"`` for air-gapped deployments where artifacts MUST be
  pre-staged.
* Picks the latest published HF artifact set whose name family matches
  the device implied by ``OVS_PROFILE`` (``nx`` ⇒ ``orin-nx``,
  ``nano`` ⇒ ``orin-nano``).
* Serialized with a module-level lock so concurrent backend preloads
  (ASR + TTS in the same process) do not race on the download.
* Idempotent — if all files are present nothing is downloaded.
* Fail-open: any error during the auto-download is logged and ``False``
  is returned; the caller is expected to re-check existence and raise
  its own ``FileNotFoundError`` if files are still missing.

Why this is opt-out (default-on)
--------------------------------
The legacy ``jetson-zh-en`` profile already auto-downloads Paraformer +
Matcha on first boot. Making the Qwen3 profiles silently fail unless the
user knew to pre-run ``deploy_qwen3_artifacts.py`` was a footgun:
switching to ``jetson-qwen3asr-matcha-nx`` would bring the ASR backend
up with ``FileNotFoundError`` and TTS would respond happily, masking the
problem. Default-on matches the existing first-boot UX. Set
``OVS_AUTO_DOWNLOAD_ARTIFACTS=0`` to opt out for production / air-gap.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

# Manifest baked into the qwen3-edgellm-jetson project shipped with the
# Jetson image at /opt/qwen3-edgellm-jetson/.
_MANIFEST_PATH = "/opt/qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json"

# A backend may call ensure_artifacts() multiple times in quick
# succession (ASR preload + TTS preload). One download in flight at a
# time avoids two snapshot_download passes hammering HF.
_LOCK = threading.Lock()


def _artifact_roots(set_spec: dict) -> tuple[Path, Path, str]:
    """Return (stable runtime root, HF local_dir, repository prefix)."""

    declared = Path(set_spec.get("root", "/opt/models/qwen3-edgellm"))
    artifact_root = Path(os.environ.get("QWEN3_ARTIFACT_ROOT") or declared)
    hf_prefix = str(set_spec.get("hf_prefix") or "").strip("/")
    if hf_prefix:
        repo_root = Path(
            os.environ.get("QWEN3_REPO_CACHE_ROOT") or artifact_root.parent
        )
    else:
        repo_root = artifact_root
    return artifact_root, repo_root, hf_prefix


def _download_patterns(
    missing_paths: Iterable[str], artifact_root: Path, hf_prefix: str
) -> tuple[list[str], set[str]]:
    """Select complete artifact directories needed by the active profile.

    An engine is never downloaded alone: its config, tokenizer, weights and
    ``.meta.json`` sidecars live in the same directory and are runtime inputs.
    Directory globs keep the pull profile-scoped without silently producing an
    incomplete engine directory.
    """

    scopes: set[str] = set()
    for raw in missing_paths:
        try:
            rel = Path(raw).resolve(strict=False).relative_to(
                artifact_root.resolve(strict=False)
            )
        except (OSError, ValueError):
            continue
        if len(rel.parts) >= 2 and rel.parts[0] in {"engines", "models"}:
            scopes.add(Path(*rel.parts[:2]).as_posix())
        elif len(rel.parts) >= 2:
            scopes.add(rel.parent.as_posix())
        else:
            scopes.add(rel.as_posix())

    prefix = f"{hf_prefix}/" if hf_prefix else ""
    patterns = {
        f"{prefix}SHA256SUMS",
        f"{prefix}manifest.json",
        f"{prefix}PROVENANCE.md",
    }
    for scope in scopes:
        patterns.add(f"{prefix}{scope}/**")
    return sorted(patterns), scopes


def _link_artifact_root(artifact_root: Path, repo_root: Path, hf_prefix: str) -> None:
    if not hf_prefix:
        return
    downloaded_root = repo_root / hf_prefix
    if artifact_root.resolve(strict=False) == downloaded_root.resolve(strict=False):
        return
    if artifact_root.is_symlink():
        if artifact_root.resolve(strict=False) != downloaded_root.resolve(strict=False):
            artifact_root.unlink()
        else:
            return
    elif artifact_root.exists():
        # Never replace operator data. A non-empty legacy directory must be
        # migrated explicitly instead of being hidden behind a symlink.
        if any(artifact_root.iterdir()):
            raise RuntimeError(
                f"artifact root {artifact_root} exists and is not the HF set root"
            )
        artifact_root.rmdir()
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.symlink_to(downloaded_root, target_is_directory=True)


def _verify_scopes(artifact_root: Path, scopes: set[str]) -> None:
    checksum_file = artifact_root / "SHA256SUMS"
    if not checksum_file.is_file():
        raise RuntimeError(f"artifact checksum manifest is missing: {checksum_file}")

    expected: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        digest, rel = fields
        rel = rel.lstrip("* ")
        if len(digest) == 64:
            expected[rel] = digest.casefold()

    selected = {
        rel: digest
        for rel, digest in expected.items()
        if any(rel == scope or rel.startswith(f"{scope}/") for scope in scopes)
    }
    if scopes and not selected:
        raise RuntimeError("downloaded artifact scopes are absent from SHA256SUMS")
    for rel, digest in sorted(selected.items()):
        path = artifact_root / rel
        if not path.is_file():
            raise RuntimeError(f"artifact file listed by SHA256SUMS is missing: {rel}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        actual = hasher.hexdigest()
        if actual != digest:
            raise RuntimeError(
                f"artifact SHA256 mismatch for {rel}: expected {digest}, got {actual}"
            )


def _is_enabled() -> bool:
    return os.environ.get("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1") == "1"


def _detect_artifact_set(profile: str, manifest: dict) -> str | None:
    """Pick the freshest published HF set matching the device family.

    ``profile`` is the ``OVS_PROFILE`` name (e.g. ``jetson-qwen3asr-matcha-nx``).
    Returns ``None`` if no set matches (caller should log + skip download).
    """
    sets = manifest.get("artifact_sets", {})

    # Explicit override wins. Profiles set ``QWEN3_ARTIFACT_SET`` to the exact
    # set (model_downloader uses the same env, deploy_paths defaults to it), so
    # honor it before the profile-name family heuristic. The heuristic cannot
    # disambiguate device-agnostic profile names like ``jetson-multilang-highperf``
    # (no "nx"/"nano") and would otherwise return None → "auto-download skipped",
    # leaving the talker engine unfetched on a slim first boot.
    explicit = os.environ.get("QWEN3_ARTIFACT_SET")
    if explicit and explicit in sets:
        return explicit

    name = profile.lower()
    if "nx" in name:
        family = "orin-nx"
    elif "nano" in name:
        family = "orin-nano"
    else:
        return None

    candidates = [
        s_name
        for s_name, spec in manifest.get("artifact_sets", {}).items()
        if spec.get("published_to_hf") and family in s_name
    ]
    if not candidates:
        return None
    # Set names are date-suffixed (e.g. orin-nx-highperf-2026-05-14);
    # lexicographic sort picks the latest.
    return sorted(candidates)[-1]


def ensure_artifacts(missing_paths: Iterable[str]) -> bool:
    """Try to fetch any HF artifacts needed to cover ``missing_paths``.

    Returns ``True`` if a download was attempted and completed without
    raising. Returns ``False`` if disabled, manifest unavailable, profile
    can't be mapped to a set, or any other recoverable issue. Download and
    integrity failures are re-raised after we commit to a concrete set.

    Caller MUST re-check file existence after this returns and raise its
    own ``FileNotFoundError`` if the download didn't actually cover what
    was missing (e.g. manifest schema drift, partial repo). A checksum mismatch
    is always fatal: a corrupt engine must never be handed to TensorRT.
    """
    if not _is_enabled():
        logger.info(
            "OVS_AUTO_DOWNLOAD_ARTIFACTS=0 → skipping Qwen3 artifact auto-download"
        )
        return False

    manifest_path = Path(_MANIFEST_PATH)
    if not manifest_path.exists():
        logger.warning(
            "Qwen3 manifest not found at %s — cannot auto-download artifacts",
            _MANIFEST_PATH,
        )
        return False

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as exc:
        logger.warning(
            "Failed to parse Qwen3 manifest %s (%s) — cannot auto-download",
            _MANIFEST_PATH, exc,
        )
        return False

    profile = os.environ.get("OVS_PROFILE", "")
    set_name = _detect_artifact_set(profile, manifest)
    if set_name is None:
        logger.warning(
            "Cannot pick HF artifact set for profile=%r — auto-download skipped. "
            "Use a jetson-qwen3asr-* profile or set OVS_PROFILE explicitly.",
            profile,
        )
        return False

    set_spec = manifest["artifact_sets"][set_name]
    artifact_root, repo_root, hf_prefix = _artifact_roots(set_spec)
    repo_id = os.environ.get("QWEN3_HF_REPO_ID") or manifest.get("hf_repo_id")
    revision = os.environ.get("QWEN3_HF_REVISION") or manifest.get(
        "revision", "main"
    )

    if not repo_id:
        logger.warning("Qwen3 manifest missing 'hf_repo_id' — cannot auto-download")
        return False

    with _LOCK:
        still_missing = [p for p in missing_paths if not Path(p).exists()]
        if not still_missing:
            return True
        logger.warning(
            "Auto-downloading Qwen3 artifact set %r (%d missing files; root=%s repo=%s rev=%s)",
            set_name, len(still_missing), artifact_root, repo_id, revision,
        )
        from huggingface_hub import snapshot_download
        allow, scopes = _download_patterns(still_missing, artifact_root, hf_prefix)
        if not scopes:
            raise RuntimeError("missing artifact paths are outside QWEN3_ARTIFACT_ROOT=" f"{artifact_root}")
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(repo_root), allow_patterns=allow, max_workers=4)
        _link_artifact_root(artifact_root, repo_root, hf_prefix)
        _verify_scopes(artifact_root, scopes)
        return True


# ---------------------------------------------------------------------------
# Model-level repositories (schema v2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelArtifactRequest:
    model_id: str
    repo: str
    required_files: tuple[str, ...] = ()
    revision: str = "main"
    root: str = ""
    manifest: str = ""
    cache_root: str = ""
    canonical_model_id: str = ""


def canonical_model_id(model_id: str) -> str:
    from server.core.hf_artifacts import canonical_model_id as _canonical
    return _canonical(model_id)


def model_cache_dir(model_id: str, cache_root: str | Path | None = None, canonical_id: str | None = None) -> Path:
    from server.core.hf_artifacts import model_cache_dir as _cache
    root = cache_root or os.environ.get("QWEN3_MODEL_CACHE_ROOT") or os.environ.get("HF_MODEL_CACHE_ROOT") or "/opt/models"
    return _cache(root, canonical_id or model_id)


def _model_rel(root: Path, raw: str) -> str:
    p = Path(str(raw))
    if p.is_absolute():
        try:
            p = p.resolve(strict=False).relative_to(root.resolve(strict=False))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"required model artifact is outside model root: {raw}") from exc
    rel = p.as_posix().lstrip("./")
    if not rel or rel == "." or rel.startswith("/") or ".." in Path(rel).parts:
        raise RuntimeError(f"invalid model artifact path: {raw!r}")
    return rel


def _manifest_rel(path: str, model_id: str, canonical_id: str | None = None) -> str:
    rel = str(path or "").replace("\\", "/").lstrip("./")
    names = [model_id.strip("/")]
    if canonical_id:
        names.append(canonical_id.strip("/"))
    for name in names:
        for prefix in (f"models/{name}/", f"{name}/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise RuntimeError(f"invalid manifest file path: {path!r}")
    return rel


def _lock(metadata: object) -> tuple[str | None, int | None]:
    if not isinstance(metadata, Mapping):
        return None, None
    digest = metadata.get("sha256") or metadata.get("sha") or metadata.get("hash")
    size = metadata.get("size", metadata.get("bytes"))
    digest = str(digest).strip().lower() if digest is not None else None
    try:
        size = int(size) if size is not None else None
    except (TypeError, ValueError):
        size = None
    return digest, size


def _manifest_files(manifest: Mapping[str, object], model_id: str, canonical_id: str | None = None) -> dict[str, tuple[str | None, int | None]]:
    raw = manifest.get("files")
    if raw is None:
        raw = manifest.get("artifacts")
    out: dict[str, tuple[str | None, int | None]] = {}
    if isinstance(raw, Mapping):
        for path, metadata in raw.items():
            out[_manifest_rel(str(path), model_id, canonical_id)] = _lock(metadata)
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                out[_manifest_rel(item, model_id, canonical_id)] = (None, None)
            elif isinstance(item, Mapping):
                name = item.get("path") or item.get("name") or item.get("file")
                if name:
                    out[_manifest_rel(str(name), model_id, canonical_id)] = _lock(item)
    return out


def _archive_spec(manifest: Mapping[str, object], model_id: str, canonical_id: str | None = None) -> dict[str, object] | None:
    raw = manifest.get("payload") or manifest.get("archive") or manifest.get("payload_archive")
    if raw is None:
        return None
    metadata: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
    path = raw if isinstance(raw, str) else (
        metadata.get("path") or metadata.get("rel_path") or metadata.get("file") or metadata.get("name") or metadata.get("archive") or ""
    )
    digest, size = _lock(metadata)
    if not path or digest is None or len(digest) != 64 or size is None or size <= 0:
        raise RuntimeError("model manifest payload/archive requires path, sha256 and size")
    return {"path": _manifest_rel(str(path), model_id, canonical_id), "sha256": digest, "size": size}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sums(root: Path) -> dict[str, tuple[str | None, int | None]]:
    path = root / "SHA256SUMS"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and len(fields[0]) == 64:
            out[fields[1].lstrip("* ").replace("\\", "/")] = (fields[0].lower(), None)
    return out


def _verify_model_files(root: Path, model_id: str, required_files: Iterable[str], manifest: Mapping[str, object], canonical_id: str | None = None) -> None:
    declared = manifest.get("model_id")
    expected_ids = {
        canonical_model_id(model_id),
        canonical_model_id(canonical_id or model_id),
    }
    if declared and canonical_model_id(str(declared)) not in expected_ids:
        expected_id = canonical_model_id(canonical_id or model_id)
        raise RuntimeError(f"model manifest id mismatch: expected {expected_id!r}, got {declared!r}")
    files = _manifest_files(manifest, model_id, canonical_id)
    sums = _read_sums(root)
    if not files:
        files = sums
    elif sums:
        for rel, lock in sums.items():
            if rel not in files or files[rel][0] is None:
                files[rel] = lock
    if not files:
        raise RuntimeError(f"model {model_id!r} has no manifest.files or SHA256SUMS entries")
    root_resolved = root.resolve(strict=False)
    for rel, (digest, size) in sorted(files.items()):
        path = root / rel
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"model artifact missing or escapes cache: {rel}") from exc
        if not path.is_file() or digest is None or len(digest) != 64:
            raise RuntimeError(f"model artifact has no valid file lock: {rel}")
        if size is not None and path.stat().st_size != size:
            raise RuntimeError(f"model artifact size mismatch: {rel}")
        if _sha256(path) != digest.lower():
            raise RuntimeError(f"model artifact SHA256 mismatch: {rel}")
    for raw in required_files:
        rel = _model_rel(root, str(raw))
        path = root / rel
        if path.is_file():
            if rel not in files:
                raise RuntimeError(f"required model artifact absent from manifest: {rel}")
        elif path.is_dir():
            if not any(item == rel or item.startswith(rel + "/") for item in files):
                raise RuntimeError(f"required model artifact directory absent from manifest: {rel}")
        else:
            raise RuntimeError(f"required model artifact missing: {rel}")


def _safe_extract(archive: Path, dest: Path) -> None:
    source = archive
    temp: Path | None = None
    if archive.name.endswith(".zst"):
        try:
            import zstandard as zstd
            fd, name = tempfile.mkstemp(suffix=".tar")
            os.close(fd); temp = Path(name)
            with archive.open("rb") as src, temp.open("wb") as out:
                zstd.ZstdDecompressor().copy_stream(src, out)
            source = temp
        except Exception:
            if temp:
                temp.unlink(missing_ok=True)
            raise RuntimeError(
                "zstd model payload requires the Python zstandard package"
            )
    try:
        mode = "r:gz" if source.name.endswith(".tar.gz") else "r:bz2" if source.name.endswith(".tar.bz2") else "r:"
        with tarfile.open(source, mode) as tf:
            base = dest.resolve(strict=False)
            for member in tf.getmembers():
                if (
                    member.name.startswith("/")
                    or ".." in Path(member.name).parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                ):
                    raise RuntimeError(f"unsafe model payload member: {member.name}")
                target = dest / member.name
                target.resolve(strict=False).relative_to(base)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tf.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot read model payload member: {member.name}")
                with stream, target.open("wb") as out:
                    shutil.copyfileobj(stream, out, length=8 * 1024 * 1024)
                target.chmod(member.mode & 0o777)
    finally:
        if temp:
            temp.unlink(missing_ok=True)


def _flatten(root: Path, model_id: str, required: Iterable[str], canonical_id: str | None = None) -> None:
    rels = [str(item).lstrip("./") for item in required if not Path(str(item)).is_absolute()]
    if all((root / rel).exists() for rel in rels):
        return
    names = [Path(model_id).name, canonical_model_id(model_id)]
    if canonical_id:
        names.extend([canonical_model_id(canonical_id), "models/" + Path(model_id).name])
    child = next((root / name for name in names if (root / name).is_dir()), None)
    if child is None:
        # Some publishers wrap a payload in a neutral directory (for example
        # ``payload/`` or ``release/``) rather than the model id. Only flatten
        # a single unambiguous directory so unrelated payloads cannot be
        # silently merged.
        children = [item for item in root.iterdir() if item.is_dir()]
        files = [item for item in root.iterdir() if item.is_file()]
        if len(children) == 1 and not files:
            child = children[0]
    if child is None:
        return
    for item in child.iterdir():
        target = root / item.name
        if target.exists():
            raise RuntimeError(f"duplicate model payload path: {target.name}")
        shutil.move(str(item), str(target))
    child.rmdir()


def _install(staging: Path, cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    old = cache.with_name(f".{cache.name}.old-{uuid.uuid4().hex}") if cache.exists() or cache.is_symlink() else None
    if old:
        os.replace(cache, old)
    try:
        os.replace(staging, cache)
    except Exception:
        if old:
            os.replace(old, cache)
        raise
    if old:
        shutil.rmtree(old, ignore_errors=True)


def _materialize(cache: Path, runtime: Path) -> None:
    if cache.resolve(strict=False) == runtime.resolve(strict=False):
        return
    runtime.parent.mkdir(parents=True, exist_ok=True)
    if runtime.is_symlink():
        runtime.unlink()
    elif runtime.exists():
        if any(runtime.iterdir()):
            raise RuntimeError(f"model runtime root exists and is non-empty: {runtime}")
        runtime.rmdir()
    runtime.symlink_to(cache, target_is_directory=True)


def _fetch_model_manifest(model_id: str, repo: str, revision: str, path: str) -> dict:
    from server.core import hf_artifacts
    try:
        return hf_artifacts.fetch_manifest(model_id, repo=repo, revision=revision, manifest_path=path or "manifest.json")
    except hf_artifacts.ArtifactError:
        if path:
            raise
        return hf_artifacts.fetch_manifest(
            model_id,
            repo=repo,
            revision=revision,
            manifest_path=f"models/{model_id}/manifest.json",
        )


def ensure_model_artifacts(
    model_id: str, repo: str, required_files: Iterable[str] = (), *, revision: str = "main",
    canonical_id: str | None = None, canonical_model_id: str | None = None,
    root: str | Path | None = None, manifest: str | None = None,
    cache_root: str | Path | None = None,
) -> bool:
    model_id = str(model_id or "").strip(); repo = str(repo or "").strip("/")
    if not model_id or not repo:
        raise RuntimeError("model-level artifact request requires model_id and repo")
    required = tuple(str(path) for path in required_files)
    revision = str(revision or "main")
    canonical = canonical_model_id or canonical_id or None
    cache = model_cache_dir(model_id, cache_root, canonical)
    runtime = Path(root) if root else cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        local_path = cache / "manifest.json"
        if local_path.is_file():
            try:
                local = json.loads(local_path.read_text(encoding="utf-8"))
                source = local.get("_source") if isinstance(local, dict) else None
                if isinstance(source, Mapping) and (str(source.get("repo", repo)).strip("/") != repo or str(source.get("revision", revision)) != revision):
                    raise RuntimeError("model cache source/revision changed")
                _verify_model_files(cache, model_id, required, local, canonical)
                _materialize(cache, runtime)
                return True
            except (OSError, json.JSONDecodeError, RuntimeError) as exc:
                logger.warning("model cache invalid for %s: %s", model_id, exc)
        if not _is_enabled():
            raise RuntimeError(f"model {model_id!r} is missing or hash-invalid while auto-download is disabled")
        model_manifest = _fetch_model_manifest(model_id, repo, revision, str(manifest or ""))
        archive = _archive_spec(model_manifest, model_id, canonical)
        staging: Path | None = Path(
            tempfile.mkdtemp(prefix=f".{cache.name}-", dir=str(cache.parent))
        )
        archive_tmp: Path | None = None
        try:
            if archive:
                from server.core import hf_artifacts
                archive_tmp = cache.parent / f".{staging.name}.{Path(str(archive['path'])).name}"
                hf_artifacts.download_file(str(archive["path"]), archive_tmp, expected_sha256=str(archive["sha256"]), expected_size=int(archive["size"]), repo=repo, revision=revision)
                if archive_tmp.stat().st_size != int(archive["size"]) or _sha256(archive_tmp) != str(archive["sha256"]).lower():
                    raise RuntimeError("model payload archive integrity mismatch")
                _safe_extract(archive_tmp, staging)
                _flatten(staging, model_id, required or _manifest_files(model_manifest, model_id, canonical), canonical)
            else:
                from huggingface_hub import snapshot_download
                files = _manifest_files(model_manifest, model_id, canonical)
                rels = set(files) | {"manifest.json", "SHA256SUMS", "PROVENANCE.md"} | set(required)
                snapshot_download(repo_id=repo, revision=revision, local_dir=str(staging), allow_patterns=sorted(rels), max_workers=4)
            out = staging / "manifest.json"
            persisted = {k: v for k, v in model_manifest.items() if k != "_source"}
            persisted["_source"] = {"model_id": model_id, "repo": repo, "revision": revision}
            if out.is_file():
                try:
                    payload_manifest = json.loads(out.read_text(encoding="utf-8"))
                    if isinstance(payload_manifest, dict):
                        payload_manifest["_source"] = persisted["_source"]
                        out.write_text(json.dumps(payload_manifest, indent=2), encoding="utf-8")
                    else:
                        out.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
                except (OSError, json.JSONDecodeError):
                    out.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
            else:
                out.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
            _verify_model_files(staging, model_id, required, model_manifest, canonical)
            _install(staging, cache)
            staging = None
            _materialize(cache, runtime)
            return True
        except Exception:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            if archive_tmp:
                archive_tmp.unlink(missing_ok=True)


def ensure_model_requests(requests: Iterable[ModelArtifactRequest | Mapping[str, object]]) -> bool:
    merged: dict[str, ModelArtifactRequest] = {}
    for raw in requests:
        if isinstance(raw, ModelArtifactRequest):
            req = raw
        elif isinstance(raw, Mapping):
            req = ModelArtifactRequest(
                model_id=str(raw.get("model_id") or raw.get("model") or ""),
                repo=str(raw.get("repo") or raw.get("hf_repo") or ""),
                required_files=tuple(str(v) for v in (raw.get("required_files") or raw.get("files") or ())),
                revision=str(raw.get("revision") or "main"),
                root=str(raw.get("root") or ""), manifest=str(raw.get("manifest") or ""),
                cache_root=str(raw.get("cache_root") or ""),
                canonical_model_id=str(raw.get("canonical_model_id") or raw.get("canonical_id") or ""),
            )
        else:
            raise TypeError(f"unsupported model artifact request: {type(raw)!r}")
        key = req.model_id.strip()
        if not key:
            raise RuntimeError("model-level artifact request requires model_id")
        old = merged.get(key)
        if old is None:
            merged[key] = req
            continue
        old_source = (old.repo.strip("/"), old.revision or "main", old.canonical_model_id, old.root, old.cache_root)
        new_source = (req.repo.strip("/"), req.revision or "main", req.canonical_model_id, req.root, req.cache_root)
        if old_source != new_source:
            raise RuntimeError(f"model {key!r} is declared by multiple repositories/roots")
        merged[key] = ModelArtifactRequest(key, old.repo, tuple(dict.fromkeys((*old.required_files, *req.required_files))), old.revision, old.root or req.root, old.manifest or req.manifest, old.cache_root or req.cache_root, old.canonical_model_id or req.canonical_model_id)
    for req in merged.values():
        ensure_model_artifacts(req.model_id, req.repo, req.required_files, revision=req.revision, canonical_model_id=req.canonical_model_id or None, root=req.root or None, manifest=req.manifest or None, cache_root=req.cache_root or None)
    return bool(merged)


ensure_models = ensure_model_requests


__all__ = [
    "ModelArtifactRequest",
    "canonical_model_id",
    "ensure_artifacts",
    "ensure_model_artifacts",
    "ensure_model_requests",
    "ensure_models",
    "model_cache_dir",
]
