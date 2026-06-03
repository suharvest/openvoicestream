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


def _is_enabled() -> bool:
    return os.environ.get("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1") == "1"


def _detect_artifact_set(profile: str, manifest: dict) -> str | None:
    """Pick the freshest published HF set matching the device family.

    ``profile`` is the ``OVS_PROFILE`` name (e.g. ``jetson-qwen3asr-matcha-nx``).
    Returns ``None`` if no set matches (caller should log + skip download).
    """
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
    can't be mapped to a set, or any other recoverable issue. Re-raises
    only if the snapshot_download call itself raises after we committed
    to it.

    Caller MUST re-check file existence after this returns and raise its
    own ``FileNotFoundError`` if the download didn't actually cover what
    was missing (e.g. manifest schema drift, partial repo).
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
    root = set_spec.get("root", "/opt/models/qwen3-edgellm")
    repo_id = manifest.get("hf_repo_id")
    revision = manifest.get("revision", "main")
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
            set_name, len(still_missing), root, repo_id, revision,
        )

        # Late import: huggingface_hub may be absent in non-Jetson images.
        from huggingface_hub import snapshot_download

        # Allow exactly the set's required_files (each is a repo-relative path),
        # NOT their top-level dir. A `engines/**` prefix would pull every other
        # device's engine set stored under engines/ in the same repo — on a
        # fresh slim start that is multi-GB of unrelated llm.engine files. The
        # exact-path patterns fetch only what this set declares.
        required = set_spec.get("required_files") or []
        allow = sorted(required) or None

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=root,
            allow_patterns=allow,
            max_workers=4,
        )

        # Highperf sets ship the talker engine under
        # ``engines/<dev>/highperf/talker_*/talker_decode_*.engine``, but
        # ``TRTEdgeLLMTTSBackend.preload()`` always looks for
        # ``<EDGE_LLM_TTS_TALKER_DIR>/llm.engine`` (default ``<root>/tts/talker``).
        # Without this the multilang/highperf customvoice TTS fails first boot
        # with "missing talker engine" even though the engine WAS downloaded.
        # Symlink the downloaded engine to the path preload expects.
        for rel in required:
            base = os.path.basename(rel)
            if base.startswith("talker_decode") and base.endswith(".engine"):
                src = Path(root) / rel
                dst = Path(root) / "tts" / "talker" / "llm.engine"
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(src, dst)
                    logger.info("Linked talker engine %s -> %s", src, dst)
                break

        logger.info("Qwen3 artifact download complete (set=%s)", set_name)
        return True


__all__ = ["ensure_artifacts"]
