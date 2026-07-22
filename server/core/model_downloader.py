"""On-demand model downloader.

Checks if required models exist for the current LANGUAGE_MODE.
Downloads missing models from CDN on first start; cached in /opt/models volume.

Models baked into the Docker image (zh_en) are always available.
English-only models (Kokoro TTS + Zipformer ASR) are downloaded on demand
when LANGUAGE_MODE=en, keeping the image small for default users.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

CDN_BASE = "https://sensecraft-statics.seeed.cc/solution-app/jetson-voice"

# Model registry: {dir_name: (cdn_filename, description)}
MODELS = {
    "zh_en": {
        "matcha-icefall-zh-en": ("models-matcha.tar.gz", "Matcha TTS (zh+en)"),
        "paraformer-streaming": ("models-paraformer.tar.gz", "Paraformer streaming ASR (zh+en)"),
    },
    "en": {
        "kokoro-multi-lang-v1_0": ("kokoro-multi-lang-v1_0.tar.bz2", "Kokoro TTS v1.0 (English, 53 speakers)"),
        "zipformer-en": ("models-zipformer-en.tar.gz", "Zipformer streaming ASR (English)"),
    },
    "shared": {
        "sensevoice": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
            "SenseVoice offline ASR (5 languages)",
        ),
    },
}

# Bundle models tied to a selectable backend: (kind, backend_key). Used to
# suppress over-fetching when a profile explicitly selects a *different*
# backend of the same kind — e.g. a Kokoro profile must not pull Matcha, a
# Qwen3 profile must not pull Paraformer, just because they are bundled in
# MODELS[language_mode]. Models not listed here (sensevoice, zipformer) are
# never profile-gated and keep their legacy language_mode behavior.
_BUNDLE_MODEL_BACKEND = {
    "matcha-icefall-zh-en": ("tts", "jetson.matcha_trt"),
    "kokoro-multi-lang-v1_0": ("tts", "jetson.kokoro_trt"),
    "paraformer-streaming": ("asr", "jetson.paraformer_trt"),
}

# Per-model files the freshness check insists on seeing.
# Without this, model dirs that engine_resolver populated with only
# auxiliary subdirs (engines/, onnx/ skeletons) pass the "non-empty"
# heuristic but still miss load-bearing resources such as tokens.txt.
_REQUIRED_FILES = {
    "matcha-icefall-zh-en": ("model-steps-3.onnx", "tokens.txt", "lexicon.txt"),
    "paraformer-streaming": ("encoder.onnx", "tokens.txt"),
    "zipformer-en": ("encoder.int8.onnx", "tokens.txt"),
    "kokoro-multi-lang-v1_0": ("model.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt"),
    "sensevoice": ("model.int8.onnx",),
}


def _detect_tar_mode(filename: str) -> str:
    """Return tar open mode based on filename extension."""
    if filename.endswith(".tar.bz2"):
        return "bz2"
    return "gz"


def _download_and_extract(url: str, dest_dir: str) -> None:
    """Download a .tar.gz or .tar.bz2 from URL and extract to dest_dir.

    Uses curl (fast, with progress) if available, falls back to Python stdlib.
    """
    compress = _detect_tar_mode(url)

    if shutil.which("curl"):
        # curl + tar streaming: no temp file, shows progress
        tar_flag = "j" if compress == "bz2" else "z"
        cmd = f'curl -fSL --progress-bar "{url}" | tar x{tar_flag}f - -C "{dest_dir}"'
        subprocess.run(cmd, shell=True, check=True)
    else:
        # Pure Python fallback
        import tarfile
        import tempfile
        import urllib.request

        suffix = ".tar.bz2" if compress == "bz2" else ".tar.gz"
        logger.info("  Fetching %s ...", url)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
            resp = urllib.request.urlopen(req, timeout=600)
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                    pct = downloaded * 100 // total
                    mb = downloaded // (1024 * 1024)
                    total_mb = total // (1024 * 1024)
                    logger.info("  Progress: %d/%d MB (%d%%)", mb, total_mb, pct)
        try:
            logger.info("  Extracting to %s ...", dest_dir)
            with tarfile.open(tmp_path, f"r:{compress}") as tar:
                tar.extractall(path=dest_dir)
        finally:
            os.unlink(tmp_path)


def ensure_models(
    language_mode: str = "zh_en",
    model_dir: str = "/opt/models",
    qwen3_required_files: list[str] | None = None,
) -> None:
    """Ensure all required models for the given language mode are present.

    Routing is profile-driven first, language_mode-driven second. When a
    profile is loaded, its ``asr_backend`` / ``tts_backend`` fields decide
    which backend-specific artifacts to fetch (Qwen3 ASR, Matcha TTS,
    Kokoro TTS). Profile-triggered requirements are UNIONED with the
    legacy language_mode requirements so callers without a profile keep
    working unchanged (e.g. plain ``LANGUAGE_MODE=en``).
    """
    try:
        from server.core.profile_loader import current_profile
        profile = current_profile() or {}
    except Exception:
        profile = {}

    asr_backend = profile.get("asr_backend")
    tts_backend = profile.get("tts_backend")
    # Opt-in, per-profile ASR artifact manifest (repo-relative path). Its
    # presence means "this profile's trt_edge_llm artifacts are a flat HF file
    # list I declare myself" and routes AWAY from the qwen3 artifact-set
    # machinery, which selects by profile-name family (nx/nano) and can neither
    # match nor serve the v090 layout. Absent → legacy qwen3 path, unchanged.
    asr_artifact_manifest = profile.get("asr_artifact_manifest")

    # Profile-driven extras (UNIONed with language_mode-driven requirements
    # further down). Pure profile users (no LANGUAGE_MODE set) end up with
    # only the entries triggered here.
    extra_required: dict = {}
    matcha = MODELS.get("zh_en", {}).get("matcha-icefall-zh-en")
    kokoro = MODELS.get("en", {}).get("kokoro-multi-lang-v1_0")
    if tts_backend == "jetson.matcha_trt" and matcha:
        extra_required["matcha-icefall-zh-en"] = matcha
        # Slim image: the SPLIT_TRT acoustic path needs standalone onnx/ files
        # that neither engine_resolver nor the sherpa CDN tarball provide.
        # Pull them from HF here (idempotent + fail-open; no-op unless the
        # profile selects MATCHA_ACOUSTIC_EP=SPLIT_TRT).
        matcha_base = os.environ.get("MATCHA_MODEL_BASE") or os.path.join(model_dir, "matcha-icefall-zh-en")
        _ensure_matcha_split_onnx(matcha_base)
    if tts_backend == "jetson.kokoro_trt" and kokoro:
        extra_required["kokoro-multi-lang-v1_0"] = kokoro
    if asr_backend == "jetson.trt_edge_llm":
        # Mutually exclusive by design: firing both would make a manifest-driven
        # profile ALSO attempt the 26-file qwen3 set (wrong repo, wrong on-disk
        # layout, and for v090 an outright "Cannot pick HF artifact set" abort).
        if asr_artifact_manifest:
            _ensure_edgellm_v090_artifacts(asr_artifact_manifest)
        else:
            # Qwen3 artifacts are deployed via an external script, not via the
            # MODELS/CDN tarball mechanism — fire it as a side-effect here.
            _ensure_qwen3_artifacts(qwen3_required_files)
    if tts_backend == "jetson.moss_tts_nano":
        # MOSS engines + codec + worker are a flat HF file list (not a
        # host-keyed engine bundle), so they bypass the MODELS/CDN tarball
        # mechanism AND engine_resolver. Provision them as a side-effect here,
        # mirroring the Qwen3 dispatch above. engine_resolver still runs after
        # this for the compile-fallback path; its list-shaped-manifest skip
        # (97a9b9f) is untouched.
        _ensure_moss_artifacts()
    if asr_backend == "jetson.sensevoice_trt":
        # SenseVoice on Jetson = standalone TRT engine. Fetch the rescaled fixed
        # ONNX + decode assets from HF and build the engine with the host-mounted
        # TensorRT (so it matches the runtime). Idempotent (skips if built).
        _ensure_sensevoice_trt_artifacts()
    if os.environ.get("ASR_BACKEND") == "sensevoice_rknn":
        # SenseVoice RKNN model + decode assets are a flat HF file list; fetch
        # the RK_PLATFORM-specific .rknn + decode assets so switching to a
        # *-sensevoice profile auto-provisions the model. Idempotent.
        _ensure_sensevoice_rknn_artifacts()

    if language_mode == "rk":
        _ensure_rk_artifacts()
        if os.environ.get("RK_ENSURE_MATCHA_RESOURCES", "1").lower() in ("0", "false", "no"):
            return
        required = {"matcha-icefall-zh-en": matcha} if matcha else {}
        required.update(extra_required)
        model_dir = os.environ.get("TTS_MODEL_DIR") or model_dir

    elif language_mode == "multilanguage":
        # Preserve legacy behavior: multilanguage mode triggers Qwen3
        # artifacts even when no profile is loaded. When a profile is
        # active, _ensure_qwen3_artifacts may have already run above —
        # the second call is cheap (re-verify) but harmless.
        # Exception: a profile that declares its own ASR artifact manifest has
        # already been provisioned above and must not fall into the qwen3 path
        # here either (LANGUAGE_MODE=multilanguage is exactly what the v090
        # profile sets, so this branch would otherwise undo the dispatch).
        if not asr_artifact_manifest:
            _ensure_qwen3_artifacts(qwen3_required_files)
        required: dict = {}
        # Some multilanguage profiles pair Qwen3 ASR with Matcha TTS. Only
        # those need the Matcha acoustic ONNX + lexicon; pure Qwen3 profiles
        # should not download or validate Matcha assets during startup.
        if tts_backend == "jetson.matcha_trt" and matcha:
            required["matcha-icefall-zh-en"] = matcha
        required.update(extra_required)
        if not required:
            return
    else:
        required = {}
        required.update(MODELS.get(language_mode, {}))
        if os.environ.get("ENSURE_OFFLINE_ASR", "").lower() in ("1", "true", "yes"):
            required.update(MODELS.get("shared", {}))
        # Profile-driven suppression of the language_mode bundle: when a profile
        # explicitly selects backends, a bundled model tied to a *different*
        # backend of the same kind (ASR/TTS) is not needed and must not be
        # fetched. Restores the per-backend exclusivity that 9cc1f35 lost when it
        # switched to UNION routing (which over-fetched Matcha for a Kokoro
        # profile, or Paraformer for a Qwen3 profile). Backward-compatible: pure
        # LANGUAGE_MODE deployments (no profile backends) skip the filter.
        if asr_backend or tts_backend:
            for dir_name in list(required):
                kind_backend = _BUNDLE_MODEL_BACKEND.get(dir_name)
                if kind_backend is None:
                    continue  # not a profile-gated backend model
                kind, backend = kind_backend
                selected = asr_backend if kind == "asr" else tts_backend
                if selected != backend:
                    required.pop(dir_name, None)
        required.update(extra_required)
    if not required:
        return

    missing = []
    for dir_name, (cdn_file, desc) in required.items():
        model_path = os.path.join(model_dir, dir_name)
        required_files = _REQUIRED_FILES.get(dir_name)
        # When required files are declared, look for the actual load-bearing
        # files recursively under the model dir (the tarball lays files
        # under subdirs in some upstream variants). Non-empty dir alone
        # is NOT a sufficient signal — engine_resolver may have written
        # the engines/ subdir before model_downloader runs.
        is_ready = False
        if os.path.isdir(model_path):
            if required_files:
                found = set()
                for root, _dirs, files in os.walk(model_path):
                    found.update(name for name in required_files if name in files)
                is_ready = found == set(required_files)
            elif os.listdir(model_path):
                is_ready = True
        if is_ready:
            logger.info("Model OK: %s (%s)", dir_name, desc)
        else:
            missing.append((dir_name, cdn_file, desc))

    if not missing:
        logger.info("All models for mode '%s' are ready.", language_mode)
        if language_mode == "en" or "kokoro-multi-lang-v1_0" in required:
            _patch_kokoro_voices(model_dir)
        return

    logger.info(
        "Downloading %d missing model(s) for mode '%s'...",
        len(missing), language_mode,
    )

    os.makedirs(model_dir, exist_ok=True)

    for dir_name, cdn_file, desc in missing:
        # SenseVoice (RPi/CPU sherpa tarball) defaults to a raw GitHub release
        # with no CDN fallback — impractically slow on edge devices without good
        # GitHub access. Default to the HF copy (honors HF_ENDPOINT, so RPi via
        # hf-mirror is fast); SENSEVOICE_MODEL_URL overrides, SENSEVOICE_RKNN_HF_REPO
        # picks the repo.
        if dir_name == "sensevoice":
            url = os.environ.get("SENSEVOICE_MODEL_URL")
            if not url:
                endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
                repo = os.environ.get("SENSEVOICE_RKNN_HF_REPO", "harvestsu/sensevoice-rknn")
                url = f"{endpoint}/{repo}/resolve/main/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
        # Use GitHub releases for models not hosted on CDN
        elif cdn_file.startswith("http"):
            url = cdn_file
        elif cdn_file == "kokoro-multi-lang-v1_0.tar.bz2":
            url = os.environ.get(
                "KOKORO_MODEL_URL",
                f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{cdn_file}",
            )
        else:
            url = f"{CDN_BASE}/{cdn_file}"
        logger.info("Downloading %s ...", desc)
        try:
            _download_and_extract(url, model_dir)
            logger.info("Downloaded %s OK.", desc)
        except Exception as e:
            logger.error("Failed to download %s: %s", desc, e)
            logger.error(
                "You can manually download from %s and extract to %s",
                url, model_dir,
            )
            sys.exit(1)

    if language_mode == "en" or "kokoro-multi-lang-v1_0" in required:
        _patch_kokoro_voices(model_dir)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_qwen3_artifacts(required_files_override: list[str] | None = None) -> None:
    """Verify or download Qwen3 artifacts for the active multilanguage profile.

    The deploy script + manifest live in the sibling `qwen3-edgellm-jetson`
    repo so they are not duplicated here. Set `QWEN3_EDGELLM_JETSON_ROOT` to
    override the default `~/project/qwen3-edgellm-jetson` lookup path.
    """
    if os.environ.get("QWEN3_ARTIFACT_AUTO_DOWNLOAD", "1").lower() in ("0", "false", "no"):
        logger.info("Qwen3 artifact auto-download disabled.")
        return

    qej_root = Path(
        os.environ.get(
            "QWEN3_EDGELLM_JETSON_ROOT",
            os.path.expanduser("~/project/qwen3-edgellm-jetson"),
        )
    )
    script = qej_root / "scripts" / "deploy_qwen3_artifacts.py"
    manifest = os.environ.get(
        "QWEN3_ARTIFACT_MANIFEST",
        str(qej_root / "deploy" / "artifacts" / "qwen3_manifest.json"),
    )
    artifact_set = os.environ.get("QWEN3_ARTIFACT_SET") or "orin-nano-highperf-2026-05-10"
    root = os.environ.get("QWEN3_ARTIFACT_ROOT")
    if not script.exists():
        # Slim image: the qwen3-edgellm-jetson submodule COPY was narrowed to
        # deploy/ only, so the deploy script is absent. Fall back to the
        # self-contained in-app HF downloader (qwen3_artifact_downloader) which
        # reads the same manifest (shipped in the slim image at
        # /opt/qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json),
        # picks the matching set, and snapshot_downloads the required engine
        # files from HF. Without this, the slim image silently skipped all
        # qwen3 ASR provisioning and the backend later raised FileNotFoundError.
        logger.warning(
            "Qwen3 artifact deploy script missing at %s — falling back to "
            "in-app HF downloader (slim image path).",
            script,
        )
        _ensure_qwen3_artifacts_via_hf(manifest, artifact_set, required_files_override)
        return

    cmd = [sys.executable, str(script), "--manifest", manifest, "--set", artifact_set]
    if root:
        cmd.extend(["--root", root])
    if os.environ.get("QWEN3_ARTIFACT_VERIFY_SHA256", "1").lower() not in ("0", "false", "no"):
        cmd.append("--verify-sha256")
    logger.info("Ensuring Qwen3 artifact set %s via %s", artifact_set, manifest)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Qwen3 artifact check/download failed with exit code %s", exc.returncode)
        sys.exit(exc.returncode)


# Standalone split-encoder ONNX files for the Matcha SPLIT_TRT acoustic path.
# These live on the HF artifact repo under models/matcha-icefall-zh-en/onnx/.
# The matcha provisioning flow otherwise only (a) extracts a host-keyed engine
# bundle (engine_resolver) and (b) pulls the sherpa CDN tarball — neither of
# which contains these standalone onnx/ files. matcha_trt's SPLIT_TRT path
# hard-requires both at preload (FileNotFoundError otherwise).
_MATCHA_SPLIT_ONNX_FILES = (
    "matcha_encoder_trt.onnx",
    "matcha_estimator_step0_trt.onnx",
)


def _ensure_matcha_split_onnx(model_base: str) -> None:
    """Provision the Matcha split-encoder standalone ONNX files from HF.

    Only relevant for the SPLIT_TRT acoustic path (``MATCHA_ACOUSTIC_EP`` =
    ``SPLIT_TRT``/``TRT_SPLIT``/``HYBRID_TRT``). Idempotent: present files are
    skipped. Fail-open: a download error is logged but does not abort startup
    (the backend's own preload re-checks and raises if still missing).
    """
    ep = (os.environ.get("MATCHA_ACOUSTIC_EP") or "").upper()
    if ep not in ("SPLIT_TRT", "TRT_SPLIT", "HYBRID_TRT"):
        return

    # The encoder ONNX env points at .../onnx/matcha_encoder_trt.onnx; derive
    # the onnx dir from it when set, else fall back to <model_base>/onnx.
    enc_env = os.environ.get("MATCHA_SPLIT_ENCODER_ONNX")
    onnx_dir = Path(enc_env).parent if enc_env else Path(model_base) / "onnx"

    targets = {name: onnx_dir / name for name in _MATCHA_SPLIT_ONNX_FILES}
    missing = {name: dest for name, dest in targets.items() if not dest.exists()}
    if not missing:
        logger.info("Matcha split-encoder ONNX already present under %s.", onnx_dir)
        return

    logger.info(
        "Matcha split-encoder ONNX provisioning: %d/%d missing under %s — fetching from HF.",
        len(missing), len(targets), onnx_dir,
    )
    try:
        from server.core.hf_artifacts import download_file, ArtifactError
    except Exception as exc:
        logger.error("Matcha split ONNX: hf_artifacts unavailable (%s) — skipping.", exc)
        return

    for name, dest in missing.items():
        rel = f"models/matcha-icefall-zh-en/onnx/{name}"
        try:
            download_file(rel, dest)
            logger.info("Matcha split ONNX downloaded: %s", dest)
        except ArtifactError as exc:
            logger.error(
                "Matcha split ONNX download failed for %s (%s) — backend preload "
                "will re-check.", name, exc,
            )
        except Exception as exc:  # noqa: BLE001 — fail-open on any unexpected error
            logger.error("Matcha split ONNX unexpected error for %s: %s", name, exc)


def _ensure_qwen3_artifacts_via_hf(
    manifest_path: str,
    artifact_set: str,
    required_files_override: list[str] | None = None,
) -> None:
    """Slim-image fallback: provision Qwen3 ASR artifacts via the in-app HF downloader.

    The fat-image path shells out to ``deploy_qwen3_artifacts.py``; that script
    is absent in the slim image. Here we read the same manifest, determine the
    set's required files and their on-disk dest under ``root``, compute which are
    missing, and call ``qwen3_artifact_downloader.ensure_artifacts`` (which does
    the actual ``snapshot_download`` from HF with allow_patterns derived from the
    required_files).

    Gated by the existing env flags. Fail-open: a download error is logged and
    the backend's own preload-time FileNotFoundError still gates correctness.
    """
    if os.environ.get("OVS_AUTO_DOWNLOAD_ARTIFACTS", "1") != "1":
        logger.info("OVS_AUTO_DOWNLOAD_ARTIFACTS=0 → skipping Qwen3 HF auto-download.")
        return

    # The profile may set QWEN3_ARTIFACT_MANIFEST to a path relative to the
    # qwen3-edgellm-jetson root (e.g. "deploy/artifacts/qwen3_manifest.json"),
    # which does not resolve against the container cwd. Resolve candidates in
    # order: the given path, <QWEN3_EDGELLM_JETSON_ROOT>/<path>, and the known
    # slim-image absolute location.
    qej_root = os.environ.get("QWEN3_EDGELLM_JETSON_ROOT", "/opt/qwen3-edgellm-jetson")
    candidates = [
        Path(manifest_path),
        Path(qej_root) / manifest_path,
        Path("/opt/qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json"),
    ]
    mp = next((c for c in candidates if c.exists()), None)
    if mp is None:
        logger.warning(
            "Qwen3 manifest not found (tried %s) — cannot HF auto-download.",
            [str(c) for c in candidates],
        )
        return
    try:
        manifest = json.loads(mp.read_text())
    except Exception as exc:
        logger.warning("Failed to parse Qwen3 manifest %s (%s).", mp, exc)
        return

    sets = manifest.get("artifact_sets", {})
    set_spec = sets.get(artifact_set)
    if set_spec is None:
        logger.warning(
            "Qwen3 artifact set %r not in manifest %s — skipping HF download.",
            artifact_set, mp,
        )
        return

    root = Path(set_spec.get("root") or os.environ.get("QWEN3_ARTIFACT_ROOT") or "/opt/models/qwen3-edgellm")
    required_files = required_files_override if required_files_override else (set_spec.get("required_files") or [])
    if not required_files:
        logger.warning("Qwen3 set %r declares no required_files — nothing to fetch.", artifact_set)
        return

    # required_files are paths relative to the set root. The profile env
    # (QWEN3_ARTIFACT_ROOT, EDGE_LLM_ASR_ENGINE_DIR, EDGE_LLM_ASR_AUDIO_ENC_DIR)
    # is layered on top of the same root, so root-relative resolution matches.
    expected_paths = [str(root / rf) for rf in required_files]
    missing_paths = [p for p in expected_paths if not Path(p).exists()]
    if not missing_paths:
        logger.info("Qwen3 ASR artifacts already present under %s (%d files).", root, len(expected_paths))
        return

    logger.info(
        "Qwen3 ASR slim provisioning: %d/%d files missing under %s — fetching from HF.",
        len(missing_paths), len(expected_paths), root,
    )
    try:
        from server.core.qwen3_artifact_downloader import ensure_artifacts
        ensure_artifacts(missing_paths)
    except Exception as exc:
        logger.error("Qwen3 ASR HF auto-download failed (%s) — backend preload will re-check.", exc)
        return

    still_missing = [p for p in expected_paths if not Path(p).exists()]
    if still_missing:
        logger.warning(
            "Qwen3 ASR HF download finished but %d files still missing (e.g. %s).",
            len(still_missing), still_missing[0],
        )
    else:
        logger.info("Qwen3 ASR artifacts ready under %s.", root)


def _ensure_rk_artifacts() -> None:
    """Verify or download RK model artifacts when an RK manifest is configured."""
    try:
        from server.core.rk_artifacts import ensure_rk_artifacts
        ensure_rk_artifacts()
    except Exception as exc:
        logger.error("RK artifact check/download failed: %s", exc)
        sys.exit(1)


def _ensure_moss_artifacts() -> None:
    """Verify or download MOSS-TTS-Nano artifacts (slim image runtime provision).

    No-op on the fat image (artifacts baked) when MOSS_ARTIFACT_AUTO_DOWNLOAD
    is disabled. On the slim image, pulls the MOSS engines + codec + worker
    from HF per ``deploy/artifacts/moss_manifest.json``. Idempotent.
    """
    try:
        from server.core.moss_artifacts import ensure_moss_artifacts
        ensure_moss_artifacts()
    except Exception as exc:
        logger.error("MOSS artifact check/download failed: %s", exc)
        sys.exit(1)


def _ensure_edgellm_v090_artifacts(manifest_path: str) -> None:
    """Verify or download the edgellm v0.9.0 ASR artifacts (slim image path).

    ``manifest_path`` is the profile's ``asr_artifact_manifest`` (repo-relative,
    e.g. ``deploy/artifacts/edgellm_v090_manifest.json``). Only the files that
    manifest lists are fetched, which is how the MOSS-TTS profile avoids pulling
    the v090 TTS engines it never loads. Fatal on failure: unlike the optional
    MOSS worker, the ASR engines have no fallback backend.
    """
    try:
        from server.core.edgellm_v090_artifacts import ensure_edgellm_v090_artifacts
        ensure_edgellm_v090_artifacts(manifest_path)
    except Exception as exc:
        logger.error("edgellm v090 ASR artifact check/download failed: %s", exc)
        sys.exit(1)


# SenseVoice RKNN: encoder .rknn (per SoC) + decode assets, hosted as a flat HF
# file list so a *-sensevoice profile auto-provisions on first start.
_SENSEVOICE_RKNN_SHARED = ("am.mvn", "embedding.npy", "chn_jpn_yue_eng_ko_spectok.bpe.model")
# Per-SoC encoder artifact: RK3576 runs plain fp16; RK3588 runs fp16 with a
# math-exact activation rescale on the block-48 FFN (plain fp16 overflows the
# RK3588 NPU on Chinese activations; int8 collapses the CTC projection — scaling
# keeps fp16 accuracy, zero quant loss).
_SENSEVOICE_RKNN_FILE = {
    "rk3576": "sense-voice-encoder.rk3576.fp16.rknn",
    "rk3588": "sense-voice-encoder.rk3588.fp16-scaled.rknn",
}


def _ensure_sensevoice_rknn_artifacts() -> None:
    """Download the SenseVoice RKNN model + decode assets if missing (idempotent).

    Fetches the ``RK_PLATFORM``-specific encoder ``.rknn`` plus the shared decode
    assets (CMVN, prompt embeddings, sentencepiece model) from HF into
    ``SENSEVOICE_RKNN_MODEL_DIR``. Honors HF_ENDPOINT mirrors. The HF repo is
    overridable via ``SENSEVOICE_RKNN_HF_REPO``.
    """
    dest = os.environ.get("SENSEVOICE_RKNN_MODEL_DIR", "/opt/asr/sensevoice-rknn")
    platform = os.environ.get("RK_PLATFORM", "rk3576").lower()
    repo = os.environ.get("SENSEVOICE_RKNN_HF_REPO", "harvestsu/sensevoice-rknn")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base = f"{endpoint}/{repo}/resolve/main"

    rknn_file = _SENSEVOICE_RKNN_FILE.get(platform, f"sense-voice-encoder.{platform}.fp16.rknn")
    files = [rknn_file, *_SENSEVOICE_RKNN_SHARED]
    os.makedirs(dest, exist_ok=True)
    for name in files:
        path = os.path.join(dest, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            logger.info("SenseVoice RKNN asset OK: %s", name)
            continue
        url = f"{base}/{name}"
        logger.info("Downloading SenseVoice RKNN asset %s ...", url)
        tmp = path + ".part"
        try:
            if shutil.which("curl"):
                subprocess.run(
                    ["curl", "-fSL", "--connect-timeout", "20", "--max-time", "1800",
                     "--retry", "3", "-o", tmp, url],
                    check=True, timeout=1900,
                )
            else:
                import urllib.request

                req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
                with urllib.request.urlopen(req, timeout=1800) as resp, open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
            os.replace(tmp, path)
            logger.info("SenseVoice RKNN asset ready: %s (%d bytes)", name, os.path.getsize(path))
        except Exception as exc:
            logger.error("Failed to download SenseVoice RKNN asset %s: %s", name, exc)
            logger.error("Manually place %s under %s", name, dest)
            raise


# SenseVoice on Jetson: the rescaled fixed ONNX (fp16-safe for Chinese) + decode
# assets; the engine is built on-device with the host-mounted TensorRT so it
# matches the runtime (TRT engines are version-specific — no universal .plan).
_SENSEVOICE_TRT_ONNX = "sense-voice-encoder.scaled.fixed.onnx"


def _ensure_sensevoice_trt_artifacts() -> None:
    """Provision the SenseVoice Jetson TRT backend (idempotent).

    Downloads the rescaled fixed ONNX + decode assets from HF into
    ``SENSEVOICE_TRT_MODEL_DIR`` and builds the fp16 TensorRT engine with the
    host-mounted TRT if it's missing. Honors HF_ENDPOINT; repo overridable via
    ``SENSEVOICE_RKNN_HF_REPO`` (shared with the RK assets).
    """
    dest = os.environ.get("SENSEVOICE_TRT_MODEL_DIR", "/opt/models/sensevoice-trt")
    repo = os.environ.get("SENSEVOICE_RKNN_HF_REPO", "harvestsu/sensevoice-rknn")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base = f"{endpoint}/{repo}/resolve/main"

    os.makedirs(dest, exist_ok=True)
    for name in (_SENSEVOICE_TRT_ONNX, *_SENSEVOICE_RKNN_SHARED):
        path = os.path.join(dest, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            logger.info("SenseVoice TRT asset OK: %s", name)
            continue
        url = f"{base}/{name}"
        logger.info("Downloading SenseVoice TRT asset %s ...", url)
        tmp = path + ".part"
        try:
            if shutil.which("curl"):
                subprocess.run(
                    ["curl", "-fSL", "--connect-timeout", "20", "--max-time", "1800",
                     "--retry", "3", "-o", tmp, url],
                    check=True, timeout=1900,
                )
            else:
                import urllib.request

                req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
                with urllib.request.urlopen(req, timeout=1800) as resp, open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
            os.replace(tmp, path)
            logger.info("SenseVoice TRT asset ready: %s (%d bytes)", name, os.path.getsize(path))
        except Exception as exc:
            logger.error("Failed to download SenseVoice TRT asset %s: %s", name, exc)
            raise

    engine = os.environ.get("SENSEVOICE_TRT_ENGINE") or os.path.join(dest, "sensevoice.plan")
    if os.path.exists(engine) and os.path.getsize(engine) > 0:
        logger.info("SenseVoice TRT engine present: %s", engine)
        return
    _build_sensevoice_trt_engine(os.path.join(dest, _SENSEVOICE_TRT_ONNX), engine)


def _build_sensevoice_trt_engine(onnx_path: str, plan_path: str) -> None:
    """Build the fp16 SenseVoice engine from ONNX with the host-mounted TensorRT.

    One-time, cached. The host TRT matches the runtime, so the engine always
    deserializes. The ONNX is already activation-rescaled (fp16-safe on zh).
    """
    import tensorrt as trt

    logger.info("Building SenseVoice TRT engine (host TRT %s) from %s ...", trt.__version__, onnx_path)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("  TRT parse error: %s", parser.get_error(i))
            raise RuntimeError(f"SenseVoice ONNX parse failed: {onnx_path!r}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    ws_gib = int(os.environ.get("SENSEVOICE_TRT_WORKSPACE_GIB", "3"))
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_gib << 30)
    except Exception:
        config.max_workspace_size = ws_gib << 30  # older TRT
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("SenseVoice TRT build_serialized_network returned None")
    tmp = plan_path + ".part"
    with open(tmp, "wb") as f:
        f.write(bytes(plan))
    os.replace(tmp, plan_path)
    logger.info("SenseVoice TRT engine built: %s (%d bytes)", plan_path, os.path.getsize(plan_path))


# Custom voice patches: replace unused speakers in voices.bin with custom voices.
# Each voice embedding is (510, 1, 256) float32 = 522240 bytes.
# Patches are stored in /opt/speech/voices/ (baked into Docker image).
_VOICE_PATCHES = {
    52: "af_cute.bin",  # replaces zm_yunyang (sid=52) with cute voice
}
_VOICE_BYTES = 510 * 1 * 256 * 4  # 522240


def _patch_kokoro_voices(model_dir: str) -> None:
    """Patch voices.bin with custom voice embeddings if not already applied."""
    voices_bin = os.path.join(model_dir, "kokoro-multi-lang-v1_0", "voices.bin")
    if not os.path.isfile(voices_bin):
        return

    patch_dir = os.path.join(os.path.dirname(__file__), "..", "voices")
    marker = voices_bin + ".patched"

    if os.path.isfile(marker):
        return

    for sid, patch_file in _VOICE_PATCHES.items():
        patch_path = os.path.join(patch_dir, patch_file)
        if not os.path.isfile(patch_path):
            logger.warning("Voice patch %s not found, skipping", patch_path)
            continue
        with open(patch_path, "rb") as f:
            patch_data = f.read()
        if len(patch_data) != _VOICE_BYTES:
            logger.warning("Voice patch %s has wrong size %d, skipping", patch_file, len(patch_data))
            continue
        offset = sid * _VOICE_BYTES
        with open(voices_bin, "r+b") as f:
            f.seek(offset)
            f.write(patch_data)
        logger.info("Patched voices.bin sid=%d with %s", sid, patch_file)

    # Write marker so we don't re-patch on every startup
    with open(marker, "w") as f:
        f.write("patched\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    ensure_models(mode, model_dir)
