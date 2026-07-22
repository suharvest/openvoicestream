"""Shared primitives for the flat-manifest runtime artifact provisioners.

``moss_artifacts`` (MOSS-TTS-Nano, #47/#48) and ``edgellm_v090_artifacts``
(v0.9.0 edgellm ASR) both provision a *flat list* of pre-staged files from the
same HF artifact repo, so they share the download / hash-verify / atomic-install
mechanics. Only the manifest and the on-device target layout differ; that stays
in the per-provisioner modules.

Why the primitives take an ``error_cls`` instead of raising one shared type:
each provisioner has its own public exception (``MossArtifactError`` /
``EdgellmV090ArtifactError``) that callers and tests already match on, and the
``optional``-file branch in the provisioning loops catches exactly that type.
Passing the class through keeps those call sites byte-for-byte equivalent to
the pre-extraction code.

Install ordering (``install_verified``) is the load-bearing part: the payload is
downloaded to a ``.staged`` sibling and hash-checked THERE, and only a passing
check is allowed to ``os.replace`` over the real destination. An earlier version
replaced first and deleted the destination on a failed check, which meant a
stale manifest destroyed a perfectly good on-device file and left nothing behind
(observed on hardware). Now a stale manifest costs a wasted download, never a
working artifact.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_REPO = "harvestsu/seeed-local-voice-artifacts"
DEFAULT_REVISION = "main"


class ArtifactProvisionError(RuntimeError):
    """Base for the flat-manifest provisioner failures."""


def endpoint() -> str:
    """HF endpoint, honouring the ``HF_ENDPOINT`` mirror env (China-friendly)."""
    return os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def hexdigest(path: Path, algo: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, item: dict) -> bool:
    """Return True if ``path`` already matches the manifest hashes for ``item``.

    sha256 is preferred when present; md5 is the fallback (HF manifest ships
    md5 for the bundled files). When neither hash is declared, existence alone
    is accepted (best-effort, matches rk_artifacts behaviour).
    """
    if not path.exists():
        return False
    expected_sha = item.get("sha256")
    expected_md5 = item.get("md5")
    if expected_sha:
        return hexdigest(path, "sha256") == expected_sha
    if expected_md5:
        return hexdigest(path, "md5") == expected_md5
    return True


def check_hashes(path: Path, item: dict, error_cls: type[Exception]) -> None:
    """Hash-check a freshly downloaded ``path``; unlink + raise on mismatch.

    Only ever pointed at a just-fetched payload (a ``.staged`` sibling), never
    at an installed artifact — so the unlink here discards junk we created, it
    can no longer take out a good on-device file.
    """
    expected_sha = item.get("sha256")
    expected_md5 = item.get("md5")
    if expected_sha:
        got = hexdigest(path, "sha256")
        if got != expected_sha:
            path.unlink(missing_ok=True)
            raise error_cls(
                f"sha256 mismatch for {path.name}: got {got}, expected {expected_sha}"
            )
    elif expected_md5:
        got = hexdigest(path, "md5")
        if got != expected_md5:
            path.unlink(missing_ok=True)
            raise error_cls(
                f"md5 mismatch for {path.name}: got {got}, expected {expected_md5}"
            )


def curl_download(
    url: str,
    dest: Path,
    error_cls: type[Exception],
    user_agent: str,
    timeout: int = 1800,
) -> None:
    """Fetch ``url`` to ``dest`` via curl.

    urllib mishandles hf-mirror's cross-host redirect chain (hf-mirror 308 ->
    huggingface.co 307 -> /api/resolve-cache 200) and fails with a spurious
    redirect loop. curl follows it robustly, so shell out to it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        subprocess.run(
            ["curl", "-fsSL", "--max-redirs", "10", "--retry", "3",
             "--connect-timeout", "30", "-A", user_agent, "-o", str(tmp), url],
            check=True, timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise error_cls(f"download failed (curl): {url}: {exc}") from exc
    os.replace(tmp, dest)


def staged_path(dest: Path) -> Path:
    return dest.parent / (dest.name + ".staged")


def install_verified(
    url: str,
    dest: Path,
    item: dict,
    download: Callable[[str, Path], None],
    check: Callable[[Path, dict], None],
) -> None:
    """Download → verify → install, never touching ``dest`` until the hash passes.

    ``download`` / ``check`` are injected (rather than called directly) so each
    provisioner keeps its own module-level functions as the seam its tests
    monkeypatch.
    """
    staged = staged_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        download(url, staged)
        check(staged, item)
    except BaseException:
        # A failed fetch/verify must leave any existing dest untouched.
        staged.unlink(missing_ok=True)
        raise
    os.replace(staged, dest)


def chmod_exec(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | 0o111)
    except OSError as exc:
        logger.warning("could not chmod +x %s: %s", path, exc)


def write_engine_meta_sidecars(root: Path, tag: str) -> None:
    """engine_resolver only trusts a local .plan/.engine if a ``.meta`` sidecar
    records the current host signature + engine hash (server/core/engine_resolver
    ._meta_matches). The flat-manifest provisioners stage the raw, md5-verified
    plans WITHOUT that sidecar, so engine_resolver rejects them as 'no valid
    local engine' and — the manifest being a file-list, not a host-keyed bundle
    — finds no HF bundle either, failing startup. Write the sidecars now so the
    staged plans resolve as a cache hit. (If a plan is not actually built for
    this host, TRT deserialization fails later anyway — this only bridges the
    provenance gap.)
    """
    try:
        from server.core.engine_resolver import _write_meta, detect_host_signature
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not import engine_resolver to write %s meta: %s", tag, exc)
        return
    host = detect_host_signature()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in (".plan", ".engine"):
            try:
                _write_meta(path, host, tag, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to write %s engine meta for %s: %s", tag, path, exc)
