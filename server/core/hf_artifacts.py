"""HuggingFace artifact downloader for OpenVoiceStream.

Downloads ONNX models and pre-built TRT engine bundles from HuggingFace,
with optional China mirror support via HF_ENDPOINT env. Designed to be
called from engine_resolver.

Layout convention on the artifact repo:
    <HF_REPO>/
        models/<model_id>/
            manifest.json              # files + SHA-256 + sizes
            <model-relative ONNX>      # raw / graph-surgery inputs for fallback rebuilds
            engines/<host_sig>.tar.gz  # pre-built engines for a specific host

Where host_sig is "sm<NN>-trt<X.Y>-jp<X.Y>-cuda<X.Y>" — see engine_resolver.

manifest.json schema (top level):
    {
      "model_id": "matcha-icefall-zh-en",
      "files": {
        "onnx/matcha_encoder_s64_trt.onnx": {"sha256": "...", "size": 12345},
        "model-steps-3.onnx": {"sha256": "...", "size": 12345},
        "engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz": {"sha256": "...", "size": 67890}
      }
    }
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
import re

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_REPO = "harvestsu/seeed-local-voice-artifacts"
DEFAULT_REVISION = "main"

# hf-mirror.com rejects Python-urllib/x.y default User-Agent with 403.
# Use a hf_hub-style UA that mirrors what huggingface_hub sends.
_UA = "openvoicestream/1.0; hf_hub-emulating"


def _open(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout)


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be fetched, verified, or extracted."""


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def _repo() -> str:
    return os.environ.get("HF_ARTIFACT_REPO", DEFAULT_REPO).strip("/")


def _revision(revision: Optional[str] = None) -> str:
    return str(revision or os.environ.get("HF_ARTIFACT_REVISION") or DEFAULT_REVISION).strip("/")


def canonical_model_id(model_id: str) -> str:
    value = str(model_id or "").strip().replace("/", "--")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip(".-") or "model"


def model_cache_dir(cache_root: str | Path, model_id: str) -> Path:
    return Path(cache_root) / canonical_model_id(model_id)


def file_url(rel_path: str, *, repo: Optional[str] = None, revision: Optional[str] = None) -> str:
    """Build the HF resolve URL for a file inside the artifact repo."""
    return f"{_endpoint()}/{str(repo or _repo()).strip('/')}/resolve/{_revision(revision)}/{rel_path.lstrip('/')}"


def _sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fetch_manifest(
    model_id: str, *, repo: Optional[str] = None, revision: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> dict:
    """Download and parse a model's manifest.json. Raises ArtifactError on failure."""
    rel = manifest_path or f"models/{model_id}/manifest.json"
    url = file_url(rel, repo=repo, revision=revision)
    try:
        with _open(url, timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ArtifactError(f"manifest not found at {url}") from exc
        raise ArtifactError(f"HTTP {exc.code} fetching {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ArtifactError(f"network error fetching {url}: {exc}") from exc
    try:
        manifest = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid manifest JSON at {url}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError(f"manifest JSON at {url} must be an object")
    manifest.setdefault("_source", {
        "model_id": str(model_id), "repo": str(repo or _repo()).strip("/"),
        "revision": _revision(revision), "path": rel,
    })
    return manifest


def download_file(
    rel_path: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """Stream a file from HF into ``dest`` via a ``.tmp`` sibling then atomic rename.

    If ``expected_sha256`` is given, verifies after download and aborts on mismatch.
    Returns the final dest path.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    url = file_url(rel_path, repo=repo, revision=revision)

    logger.info("downloading %s → %s", url, dest)
    try:
        with _open(url, timeout=60) as resp:
            with tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        if exc.code == 404:
            raise ArtifactError(f"file not found: {url}") from exc
        raise ArtifactError(f"HTTP {exc.code} on {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise ArtifactError(f"network error on {url}: {exc}") from exc

    if expected_size is not None:
        got_size = tmp.stat().st_size
        if got_size != expected_size:
            tmp.unlink(missing_ok=True)
            raise ArtifactError(
                f"size mismatch for {rel_path}: expected {expected_size}, got {got_size}"
            )
    if expected_sha256:
        got = _sha256_file(tmp)
        if got != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ArtifactError(
                f"sha256 mismatch for {rel_path}: expected {expected_sha256}, got {got}"
            )
    os.replace(tmp, dest)
    return dest


def download_and_extract_tarball(
    rel_path: str,
    dest_dir: Path,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
) -> Path:
    """Download a .tar.gz from HF, verify SHA-256, extract into ``dest_dir``.

    Extraction is done into a temp directory first; on success the contents
    are moved atomically into ``dest_dir`` to avoid leaving a partial state.
    Returns the dest_dir path.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hf_extract_", dir=str(dest_dir.parent)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        tarball = tmpdir_path / Path(rel_path).name
        download_file(
            rel_path,
            tarball,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            repo=repo,
            revision=revision,
        )

        extract_dir = tmpdir_path / "extracted"
        extract_dir.mkdir()
        with tarfile.open(tarball, "r:gz") as tf:
            # Reject absolute paths and ".." traversal.
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ArtifactError(f"unsafe tar member: {member.name}")
            tf.extractall(extract_dir)

        # Move extracted contents into dest_dir, overwriting per-file.
        for item in extract_dir.iterdir():
            target = dest_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))

    return dest_dir
