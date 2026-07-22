"""edgellm v0.9.0 ASR runtime artifact provisioner (slim image support).

The v090 prefix image bakes ~2.6GB of TensorRT ASR artifacts (thinker engine +
audio encoder + worker binary + plugin .so). This module provisions exactly the
subset a given profile needs at runtime from HuggingFace instead, so the image
carries none of it.

Why this is NOT routed through ``qwen3_artifact_downloader``: that downloader
picks its artifact set by sniffing ``nx``/``nano`` out of the profile NAME, only
serves the legacy ``/opt/models/qwen3-edgellm`` layout from a different HF repo,
and its file set is the full 26-file bundle (including the TTS talker engines
that a MOSS-TTS profile never loads). For v090 it matches nothing and bails with
"Cannot pick HF artifact set". So v090 gets a flat, explicit manifest instead —
same shape as the MOSS one, sharing all mechanics via ``artifact_provision``.

Target layout under ``/opt/edgellm-v090`` (mirrors the baked image so the
profile env is unchanged — the profile MUST NOT override these paths)::

    /opt/edgellm-v090/
        libNvInfer_edgellm_plugin.so          # EDGELLM_PLUGIN_PATH
        bin/qwen3_asr_worker                  # EDGE_LLM_ASR_WORKER_BIN (+x)
        engines/asr_thinker_full_int4_b2/     # EDGE_LLM_ASR_ENGINE_DIR
        engines/asr_audio_encoder/audio/      # EDGE_LLM_ASR_AUDIO_ENC_DIR is the
                                              # PARENT (asr_audio_encoder), the
                                              # runtime appends audio/ itself

Deliberately no engine_resolver ``.meta`` sidecars here (unlike MOSS): the v090
engines are loaded directly by the C++ ``qwen3_asr_worker`` from
``EDGE_LLM_ASR_ENGINE_DIR``, engine_resolver is not in this path (the profile's
``required_engines`` lists only MOSS plans), so sidecars would be dead files
inside a directory a foreign binary scans.

China-mirror friendly: honours ``HF_ENDPOINT`` like the sibling downloaders.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from server.core import artifact_provision as _ap

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = _ap.DEFAULT_ENDPOINT
DEFAULT_REPO = _ap.DEFAULT_REPO
DEFAULT_REVISION = _ap.DEFAULT_REVISION
DEFAULT_HF_PREFIX = "models/edgellm-v090-asr"

# Single on-device target (overridable via the manifest ``targets`` block or the
# env below). Matches the baked v090 prefix image so profile env stays valid.
DEFAULT_ENGINE_ROOT = "/opt/edgellm-v090"

# Bundled manifest, relative to the repo root — also the value profiles put in
# their ``asr_artifact_manifest`` field.
BUNDLED_MANIFEST_REL = "deploy/artifacts/edgellm_v090_manifest.json"

# hf-mirror.com rejects the default urllib UA with 403; emulate hf_hub.
_UA = "openvoicestream-edgellm-v090/1.0; hf_hub-emulating"


class EdgellmV090ArtifactError(_ap.ArtifactProvisionError):
    """Raised when edgellm v090 ASR artifacts cannot be downloaded or verified."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _endpoint() -> str:
    return _ap.endpoint()


def _verify(path: Path, item: dict) -> bool:
    return _ap.verify(path, item)


def _check_after_download(path: Path, item: dict) -> None:
    _ap.check_hashes(path, item, EdgellmV090ArtifactError)


def _download(url: str, dest: Path) -> None:
    _ap.curl_download(url, dest, EdgellmV090ArtifactError, _UA)


def _chmod_exec(path: Path) -> None:
    _ap.chmod_exec(path)


def _load_manifest(manifest_path: str | None = None) -> dict:
    """Load the edgellm v090 manifest.

    Resolution order (mirrors moss_artifacts, with the profile-supplied path
    slotted in below the env override):
      1. ``EDGELLM_V090_ARTIFACT_MANIFEST`` env (local path) — explicit override.
      2. ``manifest_path`` from the profile's ``asr_artifact_manifest`` field —
         absolute, or relative to the repo root (that is how profiles spell it).
      3. Bundled ``deploy/artifacts/edgellm_v090_manifest.json`` (default).
      4. Remote ``<HF_PREFIX>/manifest.json`` on the artifact repo (fallback).
    """
    override = os.environ.get("EDGELLM_V090_ARTIFACT_MANIFEST", "").strip()
    if override:
        path = Path(override)
        if not path.exists():
            raise EdgellmV090ArtifactError(
                f"EDGELLM_V090_ARTIFACT_MANIFEST not found: {path}"
            )
        return json.loads(path.read_text())

    if manifest_path:
        candidates = [Path(manifest_path), _repo_root() / manifest_path]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise EdgellmV090ArtifactError(
                f"asr_artifact_manifest not found (tried {[str(c) for c in candidates]})"
            )
        return json.loads(found.read_text())

    bundled = _repo_root() / BUNDLED_MANIFEST_REL
    if bundled.exists():
        return json.loads(bundled.read_text())

    repo = os.environ.get("EDGELLM_V090_ARTIFACT_REPO_ID", DEFAULT_REPO).strip("/")
    prefix = os.environ.get("EDGELLM_V090_ARTIFACT_HF_PREFIX", DEFAULT_HF_PREFIX).strip("/")
    revision = os.environ.get("EDGELLM_V090_ARTIFACT_REVISION", DEFAULT_REVISION)
    url = f"{_endpoint()}/{repo}/resolve/{revision}/{prefix}/manifest.json"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise EdgellmV090ArtifactError(
            f"failed to fetch edgellm v090 manifest: {url}: {exc}"
        ) from exc


def _resolve_engine_root(manifest: dict) -> Path:
    targets = manifest.get("targets") or {}
    return Path(
        os.environ.get("EDGELLM_V090_ENGINE_ROOT")
        or targets.get("engine_root")
        or DEFAULT_ENGINE_ROOT
    )


def ensure_edgellm_v090_artifacts(manifest_path: str | None = None) -> None:
    """Provision the v090 ASR artifacts from HF if not already present.

    No-op when ``EDGELLM_V090_ARTIFACT_AUTO_DOWNLOAD`` is disabled (the fat
    prefix image bakes everything). Idempotent: present + hash-matching files
    are skipped; a mismatching file is only replaced once the freshly fetched
    payload passes the manifest hash check, so a stale manifest cannot destroy a
    working engine. Raises ``EdgellmV090ArtifactError`` on a hard failure for a
    required file.
    """
    if os.environ.get("EDGELLM_V090_ARTIFACT_AUTO_DOWNLOAD", "1").lower() in ("0", "false", "no"):
        logger.info("edgellm v090 artifact auto-download disabled.")
        return

    manifest = _load_manifest(manifest_path)
    repo = (
        os.environ.get("EDGELLM_V090_ARTIFACT_REPO_ID")
        or manifest.get("hf_repo")
        or DEFAULT_REPO
    ).strip("/")
    prefix = (
        os.environ.get("EDGELLM_V090_ARTIFACT_HF_PREFIX")
        or manifest.get("hf_prefix")
        or DEFAULT_HF_PREFIX
    ).strip("/")
    revision = (
        os.environ.get("EDGELLM_V090_ARTIFACT_REVISION")
        or manifest.get("revision")
        or DEFAULT_REVISION
    )
    engine_root = _resolve_engine_root(manifest)

    files = manifest.get("files") or []
    if not files:
        raise EdgellmV090ArtifactError("edgellm v090 manifest declares no files")

    logger.info(
        "Ensuring edgellm v090 ASR artifacts (engine_root=%s) from %s/%s",
        engine_root, repo, prefix,
    )

    for item in files:
        # Everything is engine_root-relative and keeps its manifest subpath —
        # the worker resolves engines/<dir>/ and bin/ by exact layout.
        rel = item["path"].lstrip("/")
        dest = engine_root / rel

        if _verify(dest, item):
            logger.info("edgellm v090 artifact OK: %s", dest)
            if item.get("executable"):
                _chmod_exec(dest)
            continue

        source_rel = item.get("source_path", rel).lstrip("/")
        url = f"{_endpoint()}/{repo}/resolve/{revision}/{prefix}/{source_rel}"
        try:
            logger.info("Downloading edgellm v090 artifact %s -> %s", source_rel, dest)
            _ap.install_verified(url, dest, item, _download, _check_after_download)
        except EdgellmV090ArtifactError as exc:
            if item.get("optional"):
                logger.warning("optional edgellm v090 artifact skipped (%s): %s", rel, exc)
                continue
            raise
        if item.get("executable"):
            _chmod_exec(dest)

    logger.info("edgellm v090 ASR artifacts ready under %s", engine_root)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        ensure_edgellm_v090_artifacts()
    except EdgellmV090ArtifactError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
