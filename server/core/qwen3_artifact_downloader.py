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
import threading
from pathlib import Path
from typing import Iterable

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
        # Recheck inside the lock — a concurrent preload may have just
        # finished downloading.
        still_missing = [p for p in missing_paths if not Path(p).exists()]
        if not still_missing:
            logger.info(
                "Qwen3 artifacts now complete (another caller downloaded set=%s)",
                set_name,
            )
            return True

        logger.warning(
            "Auto-downloading Qwen3 artifact set %r (%d missing files; "
            "root=%s repo=%s rev=%s). This may take 5-15 minutes on first boot. "
            "Set OVS_AUTO_DOWNLOAD_ARTIFACTS=0 to opt out (must pre-stage artifacts).",
            set_name, len(still_missing), artifact_root, repo_id, revision,
        )

        # Late import: huggingface_hub may be absent in non-Jetson images.
        from huggingface_hub import snapshot_download

        # Pull complete directories for only the engines named by the active
        # profile.  Pulling the whole artifact set would download every TTS
        # family even when Matcha only needs ASR; pulling only ``llm.engine``
        # would omit config/tokenizer/weight sidecars required at load time.
        allow, scopes = _download_patterns(
            still_missing, artifact_root, hf_prefix
        )
        if not scopes:
            raise RuntimeError(
                "missing artifact paths are outside QWEN3_ARTIFACT_ROOT="
                f"{artifact_root}"
            )

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(repo_root),
            allow_patterns=allow,
            max_workers=4,
        )
        _link_artifact_root(artifact_root, repo_root, hf_prefix)
        _verify_scopes(artifact_root, scopes)

        # Highperf sets ship the talker engine under
        # ``engines/<dev>/highperf/talker_*/talker_decode_*.engine``, but
        # ``TRTEdgeLLMTTSBackend.preload()`` always looks for
        # ``<EDGE_LLM_TTS_TALKER_DIR>/llm.engine`` (default ``<root>/tts/talker``).
        # Without this the multilang/highperf customvoice TTS fails first boot
        # with "missing talker engine" even though the engine WAS downloaded.
        # Symlink the downloaded engine to the path preload expects.
        required = set_spec.get("required_files") or []
        for rel in required:
            base = os.path.basename(rel)
            if base.startswith("talker_decode") and base.endswith(".engine"):
                src = artifact_root / rel
                dst = artifact_root / "tts" / "talker" / "llm.engine"
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(src, dst)
                    logger.info("Linked talker engine %s -> %s", src, dst)
                break

        logger.info("Qwen3 artifact download complete (set=%s)", set_name)
        return True


__all__ = ["ensure_artifacts"]
