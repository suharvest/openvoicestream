"""Optional Rockchip model artifact downloader — voxedge adapter.

adapted from app/core/rk_artifacts.py (2026-05-30), dedup after registry switch.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ALL ``os.environ.get(...)`` reads replaced by an explicit
    ``RKArtifactConfig`` dataclass injected at call time. voxedge has no
    module-scope or hardcoded env reads.
  * No ``app.*`` import.

RK userspace runtime libraries are baked into the image. Model artifacts
(.rknn/.rkllm/tokenizer/config/lexicon) are larger and SoC/profile-specific,
so they are described by an external manifest when automatic download is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_REVISION = "main"
_UA = "openvoicestream-rk/1.0"


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                  → RKArtifactConfig field
#   RK_ARTIFACT_AUTO_DOWNLOAD       → auto_download        (default True)
#   RK_ARTIFACT_MANIFEST            → manifest_path        (default "")
#   RK_ARTIFACT_REPO_ID             → repo_id              (default "")
#   RK_ARTIFACT_REVISION            → revision             (default None → manifest/main)
#   RK_ARTIFACT_SET                 → set_name             (default None → manifest default_set)
#   RK_ARTIFACT_ROOT                → root                 (default None → spec.root / "/")
#   RK_ARTIFACT_CONTRACT_STRICT     → contract_strict      (default True)
#   HF_ENDPOINT                     → endpoint             (default DEFAULT_ENDPOINT)
# Note: ``_validate_runtime_contract`` compares the artifact set's declared
# runtime contract against the *live* process environment. The production code
# read ``os.environ`` for this; voxedge keeps the rk package fully env-free, so
# the live env must be INJECTED via ``RKArtifactConfig.runtime_env``. When it is
# ``None`` the contract validation is skipped (no implicit env read here) —
# callers that want the check pass ``runtime_env=dict(os.environ)`` from the
# app/profile layer that owns env access.


@dataclass
class RKArtifactConfig:
    """Explicit construction-time config for :func:`ensure_rk_artifacts`.

    Every field default is identical to the production env default; nothing in
    this package reads ``os.environ`` at any scope.

    ``runtime_env`` is the live runtime environment used *only* to validate the
    artifact set's ``runtime_contract`` (the production code compared declared
    contract values against ``os.environ``). It MUST be injected by the caller;
    when ``None`` the live-env contract validation is skipped (the rk package
    never reads ``os.environ`` itself — that belongs to the app/profile layer).
    """

    auto_download: bool = True
    manifest_path: str = ""
    repo_id: str = ""
    revision: Optional[str] = None
    set_name: Optional[str] = None
    root: Optional[str] = None
    contract_strict: bool = True
    endpoint: str = DEFAULT_ENDPOINT
    runtime_env: Optional[dict] = None


class RKArtifactError(RuntimeError):
    """Raised when RK artifacts cannot be downloaded or verified."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RKArtifactError(f"download failed: {url}: {exc}") from exc
    os.replace(tmp, dest)


def _load_manifest(cfg: RKArtifactConfig) -> dict | None:
    manifest_path = cfg.manifest_path.strip()
    repo_id = cfg.repo_id.strip()
    if manifest_path:
        path = Path(manifest_path)
        if not path.exists():
            raise RKArtifactError(f"manifest_path not found: {path}")
        return json.loads(path.read_text())
    if repo_id:
        endpoint = cfg.endpoint.rstrip("/")
        revision = cfg.revision or DEFAULT_REVISION
        url = f"{endpoint}/{repo_id}/resolve/{revision}/rk_manifest.json"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise RKArtifactError(f"failed to fetch RK manifest: {url}: {exc}") from exc
    return None


def _validate_runtime_contract(spec: dict, cfg: RKArtifactConfig) -> None:
    """Validate runtime env against the selected RK artifact set contract.

    RKNN artifacts are static-shape model binaries. A wrong runtime shape env
    can still produce audio bytes, but the audio may fail closed-loop ASR.
    Keep the expected env next to the artifact set so profiles and compose
    files cannot silently drift away from the validated model contract.

    The comparison target is the *live runtime* environment, which the caller
    injects via ``cfg.runtime_env``. When it is ``None`` the rk package does
    NOT read ``os.environ`` itself — validation is skipped (the env-owning
    app/profile layer is expected to pass ``runtime_env=dict(os.environ)``).
    """
    if cfg.runtime_env is None:
        logger.info(
            "RK artifact runtime_contract validation skipped "
            "(no runtime_env injected)."
        )
        return
    runtime_env = cfg.runtime_env
    contract = spec.get("runtime_contract") or {}
    expected_env = contract.get("env") or {}
    errors: list[str] = []
    for key, expected in expected_env.items():
        got = runtime_env.get(key)
        expected_s = str(expected)
        if got != expected_s:
            errors.append(f"{key}: got {got!r}, expected {expected_s!r}")

    if not errors:
        return

    message = (
        "RK artifact runtime contract mismatch for selected artifact set. "
        "Use the profile/compose env that was validated with these artifacts, "
        "or publish a new artifact set with matching runtime_contract. "
        "Details: "
        + "; ".join(errors)
    )
    if cfg.contract_strict:
        raise RKArtifactError(message)
    logger.warning("%s", message)


def ensure_rk_artifacts(config: Optional[RKArtifactConfig] = None) -> None:
    """Download RK artifacts if an RK manifest/repo is configured.

    No-op by default so existing host-mounted deployments continue to work.
    Set ``config.manifest_path`` or ``config.repo_id`` to enable.
    """
    cfg = config or RKArtifactConfig()
    if not cfg.auto_download:
        logger.info("RK artifact auto-download disabled.")
        return

    manifest = _load_manifest(cfg)
    if not manifest:
        logger.info("No RK artifact manifest configured; using mounted/model-volume artifacts.")
        return

    set_name = cfg.set_name or manifest.get("default_set")
    sets = manifest.get("artifact_sets") or {}
    spec = sets.get(set_name)
    if not set_name or not spec:
        raise RKArtifactError(
            f"RK artifact set {set_name!r} not found; available={sorted(sets)}"
        )

    root = Path(cfg.root or spec.get("root") or "/")
    repo_id = cfg.repo_id or manifest.get("hf_repo_id")
    endpoint = cfg.endpoint.rstrip("/")
    revision = cfg.revision or manifest.get("revision") or DEFAULT_REVISION
    if not repo_id:
        raise RKArtifactError("RK artifact manifest must declare hf_repo_id or set repo_id")

    logger.info("Ensuring RK artifact set %s under %s", set_name, root)
    for item in spec.get("files") or []:
        rel = item["path"].lstrip("/")
        source_rel = item.get("source_path", rel).lstrip("/")
        dest = root / rel
        expected_sha = item.get("sha256")
        if dest.exists() and (not expected_sha or _sha256(dest) == expected_sha):
            logger.info("RK artifact OK: %s", dest)
            continue
        url = f"{endpoint}/{repo_id}/resolve/{revision}/{source_rel}"
        logger.info("Downloading RK artifact %s -> %s", source_rel, rel)
        _download(url, dest)
        if expected_sha:
            got = _sha256(dest)
            if got != expected_sha:
                dest.unlink(missing_ok=True)
                raise RKArtifactError(
                    f"sha256 mismatch for {rel}: got {got}, expected {expected_sha}"
                )
    _validate_runtime_contract(spec, cfg)
