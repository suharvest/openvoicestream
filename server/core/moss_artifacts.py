"""MOSS-TTS-Nano runtime artifact provisioner (slim image support).

The fat Jetson image bakes the MOSS engines, codec, tokenizer, and the C++
``moss_tts_nano_worker`` binary into the image. The slim image strategy (#47)
instead provisions them at runtime from HuggingFace based on the active
profile.

This module is intentionally NOT routed through ``engine_resolver``:
engine_resolver expects a host-keyed ``engines/<host_sig>.tar.gz`` bundle with
a ``{key: {sha256}}`` *dict* manifest. The MOSS artifacts on HF are a flat
*list* of pre-staged files (manifest ``files`` is a list; engine_resolver
deliberately skips list-shaped manifests so it does not crash — see 97a9b9f).
So MOSS gets its own snapshot-style provisioner here that downloads each file
in the list directly into ``/opt/models/moss-tts-nano`` (engines + codec) and
the worker binary into ``/opt/jv-workers``.

Provisioning is idempotent: a file already present with a matching md5/sha256
is left untouched (no re-download, no delete). A file that does NOT match is
re-fetched into a ``.staged`` sibling and hash-verified there; only a passing
check is allowed to replace the destination, so a stale manifest can never
destroy a working on-device artifact (see artifact_provision.install_verified).

The download / hash / install mechanics are shared with the v0.9.0 edgellm ASR
provisioner and live in ``server.core.artifact_provision``; the thin wrappers
below keep this module's own error type and its monkeypatch seams.

HF layout (see ``deploy/artifacts/moss_manifest.json`` which mirrors the HF
``models/moss-tts-nano/manifest.json`` from #48)::

    <HF_REPO>/models/moss-tts-nano/
        manifest.json
        engines/<5 plans + tokenizer + shared .data + meta json>
        codec_onnx/<codec plan + tokenizer onnx + shared .data + meta json>
        moss_tts_nano_worker          # re-linked C++ worker (#48)

China-mirror friendly: honours ``HF_ENDPOINT`` like the sibling downloaders.
No extra runtime dependency (uses stdlib ``urllib`` like ``rk_artifacts``).
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
DEFAULT_HF_PREFIX = "models/moss-tts-nano"

# Default on-device targets (overridable via the manifest ``targets`` block or
# the env vars below). These match configs/profiles/jetson-moss-tts-nano-trt.json.
DEFAULT_MODEL_ROOT = "/opt/models/moss-tts-nano"
DEFAULT_WORKER_DIR = "/opt/jv-workers"

# hf-mirror.com rejects the default urllib UA with 403; emulate hf_hub.
_UA = "openvoicestream-moss/1.0; hf_hub-emulating"


class MossArtifactError(_ap.ArtifactProvisionError):
    """Raised when MOSS artifacts cannot be downloaded or verified."""


def _endpoint() -> str:
    return _ap.endpoint()


def _hexdigest(path: Path, algo: str, bufsize: int = 1 << 20) -> str:
    return _ap.hexdigest(path, algo, bufsize)


def _verify(path: Path, item: dict) -> bool:
    """Return True if ``path`` already matches the manifest hashes for ``item``."""
    return _ap.verify(path, item)


def _check_after_download(path: Path, item: dict) -> None:
    _ap.check_hashes(path, item, MossArtifactError)


def _download(url: str, dest: Path) -> None:
    _ap.curl_download(url, dest, MossArtifactError, _UA)


def _load_manifest() -> dict:
    """Load the MOSS manifest.

    Resolution order:
      1. ``MOSS_ARTIFACT_MANIFEST`` env (local path) — explicit override.
      2. Bundled ``deploy/artifacts/moss_manifest.json`` (default).
      3. Remote ``<HF_PREFIX>/manifest.json`` on the artifact repo (fallback).
    """
    override = os.environ.get("MOSS_ARTIFACT_MANIFEST", "").strip()
    if override:
        path = Path(override)
        if not path.exists():
            raise MossArtifactError(f"MOSS_ARTIFACT_MANIFEST not found: {path}")
        return json.loads(path.read_text())

    bundled = Path(__file__).resolve().parents[2] / "deploy" / "artifacts" / "moss_manifest.json"
    if bundled.exists():
        return json.loads(bundled.read_text())

    # Remote fallback.
    repo = os.environ.get("MOSS_ARTIFACT_REPO_ID", DEFAULT_REPO).strip("/")
    prefix = os.environ.get("MOSS_ARTIFACT_HF_PREFIX", DEFAULT_HF_PREFIX).strip("/")
    revision = os.environ.get("MOSS_ARTIFACT_REVISION", DEFAULT_REVISION)
    url = f"{_endpoint()}/{repo}/resolve/{revision}/{prefix}/manifest.json"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise MossArtifactError(f"failed to fetch MOSS manifest: {url}: {exc}") from exc


def _resolve_targets(manifest: dict) -> dict[str, Path]:
    targets = manifest.get("targets") or {}
    model_root = (
        os.environ.get("MOSS_MODEL_ROOT")
        or targets.get("model_root")
        or DEFAULT_MODEL_ROOT
    )
    worker_dir = (
        os.environ.get("MOSS_WORKER_DIR")
        or targets.get("worker_dir")
        or DEFAULT_WORKER_DIR
    )
    return {"model_root": Path(model_root), "worker_dir": Path(worker_dir)}


def _write_engine_meta_sidecars(model_root: Path) -> None:
    """Write engine_resolver ``.meta`` sidecars for the staged MOSS plans.

    See ``artifact_provision.write_engine_meta_sidecars`` for why this is
    needed; the ``moss_prestaged`` tag is the provenance marker recorded in the
    sidecar and must stay stable (already written on deployed devices).
    """
    _ap.write_engine_meta_sidecars(model_root, "moss_prestaged")


def ensure_moss_artifacts() -> None:
    """Provision MOSS-TTS-Nano artifacts from HF if not already present.

    No-op when ``MOSS_ARTIFACT_AUTO_DOWNLOAD`` is disabled (fat image bakes
    everything). Idempotent: present + hash-matching files are skipped. An
    installed file is only ever replaced by a payload that already passed the
    manifest hash check in its ``.staged`` sibling, so a stale/wrong manifest
    costs a wasted download rather than the working artifact. Raises
    ``MossArtifactError`` on a hard failure for a required file.
    """
    if os.environ.get("MOSS_ARTIFACT_AUTO_DOWNLOAD", "1").lower() in ("0", "false", "no"):
        logger.info("MOSS artifact auto-download disabled.")
        return

    manifest = _load_manifest()
    repo = (
        os.environ.get("MOSS_ARTIFACT_REPO_ID")
        or manifest.get("hf_repo")
        or DEFAULT_REPO
    ).strip("/")
    prefix = (
        os.environ.get("MOSS_ARTIFACT_HF_PREFIX")
        or manifest.get("hf_prefix")
        or DEFAULT_HF_PREFIX
    ).strip("/")
    revision = (
        os.environ.get("MOSS_ARTIFACT_REVISION")
        or manifest.get("revision")
        or DEFAULT_REVISION
    )
    targets = _resolve_targets(manifest)

    files = manifest.get("files") or []
    if not files:
        raise MossArtifactError("MOSS manifest declares no files")

    logger.info(
        "Ensuring MOSS artifacts (model_root=%s worker_dir=%s) from %s/%s",
        targets["model_root"], targets["worker_dir"], repo, prefix,
    )

    for item in files:
        rel = item["path"].lstrip("/")
        dest_key = item.get("dest", "model_root")
        base = targets.get(dest_key, targets["model_root"])
        # Worker binaries live flat under worker_dir; model files keep their
        # codec_onnx/ / engines/ subpath under model_root.
        dest = base / (Path(rel).name if dest_key == "worker_dir" else rel)

        if _verify(dest, item):
            logger.info("MOSS artifact OK: %s", dest)
            if item.get("executable"):
                _chmod_exec(dest)
            continue

        source_rel = item.get("source_path", rel).lstrip("/")
        url = f"{_endpoint()}/{repo}/resolve/{revision}/{prefix}/{source_rel}"
        try:
            logger.info("Downloading MOSS artifact %s -> %s", source_rel, dest)
            _ap.install_verified(url, dest, item, _download, _check_after_download)
        except MossArtifactError as exc:
            if item.get("optional"):
                logger.warning("optional MOSS artifact skipped (%s): %s", rel, exc)
                continue
            raise
        if item.get("executable"):
            _chmod_exec(dest)

    _write_engine_meta_sidecars(targets["model_root"])
    logger.info("MOSS artifacts ready under %s", targets["model_root"])


def _chmod_exec(path: Path) -> None:
    _ap.chmod_exec(path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        ensure_moss_artifacts()
    except MossArtifactError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
