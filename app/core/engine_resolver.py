"""Runtime TRT engine resolver for OpenVoiceStream.

For each engine declared in the active profile's ``required_engines`` list,
the resolver guarantees a valid engine file exists at the target path
before backends are imported. Resolution order:

  1. Local cache hit  -- engine_path exists + sidecar .meta.json matches host
  2. HuggingFace prebuilt bundle for <host_sig>.tar.gz

Engines are resolved from baked/local ``engine_path`` or a prebuilt HF bundle
only. The product is a pure runtime: it never compiles engines locally. Engine
build tooling lives in the jetson-voice-engine repo
(``third_party/qwen3-edgellm-jetson/models/``), not here. ONNX artifacts are
rebuild inputs, not TensorRT runtime dependencies, and a compatible prebuilt
engine must not trigger ONNX downloads.

Backends read engine paths from env vars at import time, so the resolver
also injects every entry's ``env_var`` → ``engine_path`` into ``os.environ``
BEFORE returning. This MUST be called before any backend module is imported.

Concurrency: a single ``flock`` on ``<MODEL_DIR>/.engine_resolver.lock``
covers the whole resolve_all() call to avoid two starting containers
racing on the shared volume.

Per-engine schema in profile JSON::

  {
    "model_id": "matcha-icefall-zh-en",
    "engine_file": "matcha_encoder_s64_bf16.engine",
    "engine_path": "/opt/models/matcha-icefall-zh-en/engines/matcha_encoder_s64_bf16.engine",
    "env_var": "MATCHA_ENCODER_ENGINE",          // backend reads this
    "onnx_input": "matcha_encoder_s64_trt.onnx", // used only for HF bundle onnx_sha metadata
    "extra_files": ["engines/cpu_length_regulator.onnx"], // model-relative runtime files
    "hf_only": false,                            // retained for HF-resolution semantics
    "required": true                             // default true; false => skip on miss
  }
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# --- env keys controlling resolver behaviour ------------------------------
ENV_MODELS_DIR = ("OVS_MODELS_DIR",)  # default /opt/models
ENV_PREFETCH_ONNX = ("OVS_PREFETCH_ONNX",)  # 0/1, default 0
ENV_FORCE_REBUILD = ("OVS_FORCE_REBUILD",)  # 0/1, default 0


def _env(names: str | tuple[str, ...], default: str | None = None) -> str | None:
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


# ---------------------------------------------------------------------------
# Host signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HostSignature:
    sm: str             # "87" for Orin
    trt_version: str    # "10.3" (major.minor)
    jp_version: str     # "6.2"
    cuda_version: str   # "12.6"

    @property
    def key(self) -> str:
        return f"sm{self.sm}-trt{self.trt_version}-jp{self.jp_version}-cuda{self.cuda_version}"

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "trt_version": self.trt_version,
            "jp_version": self.jp_version,
            "cuda_version": self.cuda_version,
        }


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("%s failed: %s", " ".join(cmd), exc)
        return ""


def _detect_sm() -> str:
    out = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    # e.g. "8.7" → "87"
    m = re.search(r"(\d+)\.(\d+)", out)
    if m:
        return m.group(1) + m.group(2)
    # Fallback: read /proc/device-tree for Tegra
    return _env(("OVS_SM",), "87") or "87"


def _detect_trt_version() -> str:
    # dpkg -l | grep libnvinfer-bin returns lines like
    #   ii  libnvinfer-bin   10.3.0.30-1+cuda12.5   arm64   TensorRT binaries
    out = _run(["dpkg", "-l", "libnvinfer-bin"])
    m = re.search(r"(\d+\.\d+)\.\d+\.\d+", out)
    if m:
        return m.group(1)
    return _env(("OVS_TRT",), "10.3") or "10.3"


def _detect_cuda_version() -> str:
    # dpkg line includes "+cudaX.Y" suffix
    out = _run(["dpkg", "-l", "libnvinfer-bin"])
    m = re.search(r"\+cuda(\d+\.\d+)", out)
    if m:
        return m.group(1)
    return _env(("OVS_CUDA",), "12.6") or "12.6"


def _detect_jp_version() -> str:
    # /etc/nv_tegra_release first line: "# R36 (release), REVISION: 4.3, ..."
    try:
        with open("/etc/nv_tegra_release") as f:
            line = f.readline()
    except OSError:
        return _env(("OVS_JP",), "6.2") or "6.2"
    m = re.search(r"R(\d+)\s*\(release\)\s*,\s*REVISION:\s*(\d+)\.(\d+)", line)
    if not m:
        return _env(("OVS_JP",), "6.2") or "6.2"
    rmajor = int(m.group(1))
    # R36 → JetPack 6.x, R35 → JetPack 5.x
    jp_major = {36: 6, 35: 5}.get(rmajor, 6)
    # REVISION major maps to JetPack minor (4.3 → 6.2 on R36)
    rev_major = int(m.group(2))
    jp_minor = max(0, rev_major - 2)  # R36/REV 4 → JP 6.2; tunable
    return f"{jp_major}.{jp_minor}"


def detect_host_signature() -> HostSignature:
    sig = HostSignature(
        sm=_detect_sm(),
        trt_version=_detect_trt_version(),
        jp_version=_detect_jp_version(),
        cuda_version=_detect_cuda_version(),
    )
    logger.info("host signature: %s", sig.key)
    return sig


# ---------------------------------------------------------------------------
# Engine metadata sidecar
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _meta_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(engine_path.suffix + ".meta.json")


def _read_meta(engine_path: Path) -> Optional[dict]:
    mp = _meta_path(engine_path)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_meta(engine_path: Path, host: HostSignature, source: str, onnx_sha: Optional[str]) -> None:
    """Write meta sidecar atomically.

    ``source`` is "cache" / "hf_bundle" / "local_compile" for diagnostic use.
    """
    meta = {
        "host": host.to_dict(),
        "engine_sha256": _sha256_file(engine_path),
        "onnx_sha256": onnx_sha,
        "source": source,
        "written_at": int(time.time()),
    }
    mp = _meta_path(engine_path)
    tmp = mp.with_suffix(mp.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, mp)


def _iter_extracted_engine_files(engine_dir: Path) -> list[Path]:
    """Return TensorRT engine files extracted into an engine directory."""
    if not engine_dir.exists():
        return []
    out: list[Path] = []
    for path in engine_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix in {".engine", ".plan"}:
            out.append(path)
    return out


def _meta_matches(engine_path: Path, host: HostSignature) -> bool:
    """Cache freshness check. Engine must exist, sidecar must exist, host must match,
    and the engine binary hash must still match what we recorded.
    """
    if not engine_path.exists():
        return False
    meta = _read_meta(engine_path)
    if not meta:
        return False
    if meta.get("host") != host.to_dict():
        logger.info("host mismatch for %s: cache=%s host=%s",
                    engine_path.name, meta.get("host"), host.to_dict())
        return False
    if meta.get("engine_sha256") != _sha256_file(engine_path):
        logger.warning("engine hash drift detected at %s — treating as stale", engine_path)
        return False
    return True


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass
class EngineSpec:
    model_id: str
    engine_file: str
    engine_path: Path
    env_var: str
    onnx_input: Optional[str]
    hf_only: bool
    required: bool
    extra_files: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineSpec":
        engine_path = Path(d["engine_path"])
        return cls(
            model_id=d["model_id"],
            engine_file=d["engine_file"],
            engine_path=engine_path,
            env_var=d["env_var"],
            onnx_input=d.get("onnx_input"),
            extra_files=list(d.get("extra_files") or []),
            hf_only=bool(d.get("hf_only", False)),
            required=bool(d.get("required", True)),
        )


def _model_root(spec: EngineSpec) -> Path:
    return spec.engine_path.parent.parent


def _path_under(path: Path, root: Path) -> Optional[Path]:
    """Return ``path`` relative to ``root`` if it stays inside the model dir."""
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _onnx_path_candidates(spec: EngineSpec) -> list[Path]:
    """Candidate local paths for a profile's ONNX input.

    Older profile entries used ``onnx_input`` as a filename under
    ``<model>/onnx``. Some build scripts, however, consume files from the
    model root (Paraformer) or intentionally use ``../model-steps-3.onnx`` to
    point from ``onnx/`` back to the Matcha source model. Keep both layouts
    working and reject paths that escape the model directory.
    """
    if not spec.onnx_input:
        return []
    raw = Path(spec.onnx_input)
    if raw.is_absolute():
        return [raw]

    root = _model_root(spec)
    candidates: list[Path] = []
    for candidate in (root / "onnx" / raw, root / raw):
        if _path_under(candidate, root) is not None and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _extra_file_paths(spec: EngineSpec) -> list[Path]:
    root = _model_root(spec)
    out: list[Path] = []
    for rel in spec.extra_files:
        raw = Path(rel)
        path = raw if raw.is_absolute() else root / raw
        if _path_under(path, root) is None:
            logger.warning("extra file for %s escapes model root: %s", spec.engine_file, path)
            continue
        out.append(path)
    return out


def _extra_files_exist(spec: EngineSpec) -> bool:
    missing = [str(path) for path in _extra_file_paths(spec) if not path.exists()]
    if missing:
        logger.info("extra runtime files missing for %s: %s", spec.engine_file, missing)
        return False
    return True


def _try_hf_resolve(spec: EngineSpec, host: HostSignature) -> bool:
    """Try to download a prebuilt bundle matching host_sig.

    Returns True if engine_path now contains a valid file.
    """
    from app.core import hf_artifacts

    try:
        manifest = hf_artifacts.fetch_manifest(spec.model_id)
    except hf_artifacts.ArtifactError as exc:
        logger.info("no HF manifest for %s: %s", spec.model_id, exc)
        return False

    # manifest.json keys are model-relative ("engines/<host_sig>.tar.gz");
    # the HF fetch URL needs the full "models/<id>/..." path.
    manifest_key = f"engines/{host.key}.tar.gz"
    files = manifest.get("files")
    if not isinstance(files, dict):
        # Some manifests (e.g. moss) ship ``files`` as a list rather than the
        # ``{key: {sha256}}`` dict this resolver expects. Don't crash — just
        # report no engine-bundle match (the list-shaped manifest describes a
        # pre-staged engine dir, not a downloadable host-keyed bundle).
        logger.info(
            "HF manifest 'files' for %s is %s, not a bundle dict — skipping HF bundle resolve",
            spec.model_id, type(files).__name__,
        )
        return False
    file_info = files.get(manifest_key)
    if not file_info:
        logger.info("HF manifest has no bundle for %s @ %s", spec.model_id, host.key)
        return False
    bundle_rel = f"models/{spec.model_id}/{manifest_key}"

    try:
        hf_artifacts.download_and_extract_tarball(
            bundle_rel,
            spec.engine_path.parent,
            expected_sha256=file_info.get("sha256"),
        )
    except hf_artifacts.ArtifactError as exc:
        logger.warning("HF bundle download failed for %s: %s", spec.model_id, exc)
        return False

    if not spec.engine_path.exists():
        logger.warning(
            "HF bundle extracted but %s not found — engine name mismatch?",
            spec.engine_path,
        )
        return False
    if not _extra_files_exist(spec):
        logger.warning(
            "HF bundle extracted but extra runtime files are missing for %s",
            spec.engine_file,
        )
        return False

    onnx_sha = None
    for onnx_p in _onnx_path_candidates(spec):
        if onnx_p.exists():
            onnx_sha = _sha256_file(onnx_p)
            break
    for engine_file in _iter_extracted_engine_files(spec.engine_path.parent):
        try:
            _write_meta(engine_file, host, source="hf_bundle", onnx_sha=onnx_sha)
        except OSError as exc:
            logger.warning("failed to write HF bundle metadata for %s: %s", engine_file, exc)
    return True


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def _models_dir() -> Path:
    return Path(_env(ENV_MODELS_DIR, "/opt/models") or "/opt/models")


def _acquire_lock():
    """Context manager-like helper returning an open fd holding the resolver lock.

    Falls back to a no-op on systems without fcntl (mostly for local dev on Mac).
    """
    lock_path = _models_dir() / ".engine_resolver.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
    except (ImportError, OSError):
        logger.warning("flock not available; running without resolver lock")
    return fd


def _release_lock(fd: int) -> None:
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    try:
        os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_all(profile: dict) -> dict[str, Path]:
    """Resolve every engine declared by ``profile['required_engines']``.

    On success, returns a dict of ``env_var → engine_path`` and also injects
    each entry into ``os.environ`` so backend modules can read them at import
    time. Raises RuntimeError on first hard failure (and marks any partially
    resolved entries as not exported).
    """
    entries = profile.get("required_engines") or []
    if not entries:
        logger.info("profile declares no required_engines — skipping resolver")
        return {}

    host = detect_host_signature()
    force_rebuild = (_env(ENV_FORCE_REBUILD, "0") or "0") in ("1", "true", "yes")

    fd = _acquire_lock()
    try:
        resolved: dict[str, Path] = {}
        for raw in entries:
            spec = EngineSpec.from_dict(raw)
            try:
                _resolve_one(spec, host, force_rebuild=force_rebuild)
            except Exception as exc:
                if not spec.required:
                    logger.warning("optional engine %s skipped: %s", spec.engine_file, exc)
                    continue
                raise RuntimeError(
                    f"failed to resolve required engine {spec.engine_file}: {exc}"
                ) from exc
            os.environ[spec.env_var] = str(spec.engine_path)
            resolved[spec.env_var] = spec.engine_path
        return resolved
    finally:
        _release_lock(fd)


def _resolve_one(spec: EngineSpec, host: HostSignature, force_rebuild: bool) -> None:
    if not force_rebuild and _meta_matches(spec.engine_path, host) and _extra_files_exist(spec):
        logger.info("cache hit: %s (host=%s)", spec.engine_path.name, host.key)
        return

    # Stale/unverified cache: clear only the meta sidecar so a stale or
    # missing meta can't falsely match. Do NOT delete the engine file itself
    # here — if the HF resolve below fails, deleting it first would destroy a
    # possibly-valid manually-staged engine (data loss; e.g. moss .plan files
    # staged without a .meta sidecar — that is how moss_tts_prefill.plan got
    # deleted). HF extract overwrites the engine in place, so an eager unlink
    # buys nothing and only risks loss.
    _meta_path(spec.engine_path).unlink(missing_ok=True)

    # Try the prebuilt HF bundle. The product is a pure runtime and never
    # compiles engines locally — engines must be baked/local (cache hit above)
    # or come from a prebuilt HF bundle.
    if _try_hf_resolve(spec, host):
        logger.info("hf bundle: %s (host=%s)", spec.engine_path.name, host.key)
        return

    raise RuntimeError(
        f"engine {spec.engine_file} not found: no valid local engine and no HF "
        f"bundle for {host.key} (local engine compilation is not supported at runtime)"
    )
