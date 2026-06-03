"""Runtime TRT engine resolver for OpenVoiceStream.

For each engine declared in the active profile's ``required_engines`` list,
the resolver guarantees a valid engine file exists at the target path
before backends are imported. Resolution order:

  1. Local cache hit  -- engine_path exists + sidecar .meta.json matches host
  2. HuggingFace prebuilt bundle for <host_sig>.tar.gz

Engines are resolved from baked/local ``engine_path`` or a prebuilt HF bundle
only. The product is a pure runtime: it never compiles engines locally. Engine
build tooling lives in the jetson-voice-engine repo
(``third_party/jetson-voice-engine/models/``), not here. ONNX artifacts are
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
    from server.core import hf_artifacts

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

# ---------------------------------------------------------------------------
# Actionable-failure UX (P0): classify each provisioning failure into a stable
# code (F1..F7) with a copy-pasteable remediation, collect ALL failures in one
# pass (never crash-on-first), and surface them in the boot log + /readyz.
# Design: docs/specs/engine-host-coverage-and-builder-sidecar.md (internal).
# ---------------------------------------------------------------------------


@dataclass
class EngineStatus:
    name: str
    env_var: str
    state: str          # "cache" | "hf_bundle" | "skipped_optional" | f"FAILED:{code}"
    code: Optional[str] = None        # F1..F7 when FAILED
    cause: Optional[str] = None
    remediation: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "env_var": self.env_var, "state": self.state}
        if self.code:
            d.update(code=self.code, cause=self.cause, remediation=self.remediation)
        return d


@dataclass
class ProvisioningReport:
    host_signature: str
    supported_signatures: list           # signatures with a prebuilt bundle (best-effort)
    engines: list                        # list[EngineStatus]

    @property
    def failures(self) -> list:
        return [e for e in self.engines if e.state.startswith("FAILED")]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "ready": self.ok,
            "host_signature": self.host_signature,
            "supported_signatures": self.supported_signatures,
            "engines": [e.to_dict() for e in self.engines],
        }


class EngineResolutionError(RuntimeError):
    """Raised by resolve_all when one or more required engines fail to resolve.

    Subclasses RuntimeError so existing callers/tests that catch RuntimeError
    keep working; ``.report`` carries the structured ProvisioningReport and the
    message is the full actionable F1..F7 block.
    """

    def __init__(self, message: str, report: "ProvisioningReport"):
        super().__init__(message)
        self.report = report


_LAST_REPORT: Optional[ProvisioningReport] = None


def get_last_report() -> Optional[ProvisioningReport]:
    """The most recent ProvisioningReport (for /readyz). None before first run."""
    return _LAST_REPORT


def _available_signatures(model_id: str) -> list:
    """Best-effort list of host signatures that have a prebuilt bundle on HF,
    parsed from the model's manifest. Empty on any error (used only for hints)."""
    from server.core import hf_artifacts
    try:
        manifest = hf_artifacts.fetch_manifest(model_id)
    except Exception:
        return []
    files = manifest.get("files")
    if not isinstance(files, dict):
        return []
    sigs = []
    for key in files:
        # keys look like "engines/sm87-trt10.3-jp6.2-cuda12.6.tar.gz"
        if key.startswith("engines/") and key.endswith(".tar.gz"):
            sigs.append(key[len("engines/"):-len(".tar.gz")])
    return sorted(sigs)


def _classify_failure(spec: EngineSpec, host: HostSignature, exc: Exception) -> EngineStatus:
    """Map a resolution failure to (code, cause, remediation). Re-probes the
    manifest on the (rare) failure path; never raises."""
    from server.core import hf_artifacts

    msg = str(exc)
    base = EngineStatus(name=spec.engine_file, env_var=spec.env_var, state="FAILED")

    # F2 — GPU/host not detected.
    if not host.sm or host.sm in ("", "unknown"):
        base.code = "F2"
        base.state = "FAILED:F2"
        base.cause = "no CUDA device detected (host signature incomplete)"
        base.remediation = (
            "ensure 'runtime: nvidia' + the /host-cuda, /host-nvidia-libs and "
            "/usr/src/tensorrt mounts are present (see deploy/docker-compose.yml). "
            "Is this a Jetson with JetPack installed?")
        return base

    # F4/F5 — disk/extract or integrity, detectable from the message.
    low = msg.lower()
    if any(k in low for k in ("no space", "enospc", "disk full")):
        base.code = "F4"
        base.state = "FAILED:F4"
        base.cause = f"extract/disk failure ({msg})"
        base.remediation = (
            f"free space or mount a larger volume at the model dir of "
            f"{spec.engine_path.parent}; ensure bzip2 is installed (baked in image).")
        return base
    if any(k in low for k in ("sha256", "checksum", "md5", "hash drift")):
        base.code = "F5"
        base.state = "FAILED:F5"
        base.cause = f"checksum mismatch ({msg})"
        base.remediation = (
            f"corrupt/partial download or wrong-build artifact on HF. Delete "
            f"{spec.engine_path} and its .meta.json, then reboot to re-fetch. If it "
            f"recurs, the HF artifact set is wrong-build — report it.")
        return base

    # Probe HF to split F1 (host uncovered) / F3 (HF unreachable) / F6 (incomplete).
    try:
        manifest = hf_artifacts.fetch_manifest(spec.model_id)
    except hf_artifacts.ArtifactError as fe:
        if "not found" in str(fe).lower():
            base.code = "F1"
            base.state = "FAILED:F1"
            base.cause = f"no published artifact manifest for model '{spec.model_id}'"
            base.remediation = (
                f"this model has no prebuilt bundle published. Verify the profile's "
                f"model_id, or request a build / upload of the artifact set.")
        else:
            base.code = "F3"
            base.state = "FAILED:F3"
            base.cause = f"HuggingFace unreachable ({fe})"
            base.remediation = (
                f"check connectivity / HF_ENDPOINT (the mirror does not serve all "
                f"files), then reboot. To pre-stage offline: hf download "
                f"<repo> --include 'models/{spec.model_id}/engines/*' "
                f"--local-dir {spec.engine_path.parent.parent}")
        return base

    files = manifest.get("files")
    have_bundle = isinstance(files, dict) and f"engines/{host.key}.tar.gz" in files
    if not have_bundle:
        sigs = _available_signatures(spec.model_id)
        base.code = "F1"
        base.state = "FAILED:F1"
        base.cause = f"no prebuilt engine for host {host.key}"
        sup = ", ".join(sigs) if sigs else "(none published)"
        base.remediation = (
            f"your device signature {host.key} has no prebuilt bundle. Supported: "
            f"{sup}. Fix: flash a supported JetPack/TRT, OR request a build for "
            f"{host.key} (file an issue, paste this signature), OR enable the "
            f"builder sidecar (OVS_ALLOW_LOCAL_BUILD=1).")
        return base

    # Bundle exists but the engine still didn't materialize → incomplete set.
    base.code = "F6"
    base.state = "FAILED:F6"
    base.cause = f"bundle resolved but engine/extra file missing ({msg})"
    base.remediation = (
        f"upstream artifact gap: the bundle for {host.key} is missing a required "
        f"file. Report the artifact set + missing file ({spec.engine_file}). Not "
        f"fixable on-device.")
    return base


def format_report_text(report: ProvisioningReport) -> str:
    """Human-readable boot-log block: failures first, each with a → fix line."""
    lines = [
        "",
        "================ ENGINE PROVISIONING ================",
        f"device host signature: {report.host_signature}",
    ]
    # failures first, then the ok ones
    ordered = report.failures + [e for e in report.engines if not e.state.startswith("FAILED")]
    for e in ordered:
        if e.state.startswith("FAILED"):
            lines.append(f"  ✗ {e.name} [{e.code}]: {e.cause}")
            lines.append(f"      → fix: {e.remediation}")
        else:
            lines.append(f"  ✓ {e.name} ({e.state})")
    n_ok = len(report.engines) - len(report.failures)
    codes = ",".join(sorted({e.code for e in report.failures if e.code}))
    lines.append(
        f"summary: {n_ok}/{len(report.engines)} ok"
        + (f", {len(report.failures)} FAILED ({codes})" if report.failures else "")
    )
    lines.append("=====================================================")
    return "\n".join(lines)


def build_report(profile: dict, *, export_ok: bool = False) -> ProvisioningReport:
    """Resolve every required engine in ONE pass, collecting all outcomes.

    Never crashes on the first failure. When ``export_ok`` is True, successfully
    resolved engines have their env_var exported (used by resolve_all)."""
    global _LAST_REPORT
    entries = profile.get("required_engines") or []
    host = detect_host_signature()
    force_rebuild = (_env(ENV_FORCE_REBUILD, "0") or "0") in ("1", "true", "yes")

    statuses: list = []
    for raw in entries:
        spec = EngineSpec.from_dict(raw)
        try:
            _resolve_one(spec, host, force_rebuild=force_rebuild)
        except Exception as exc:  # noqa: BLE001 — classify, don't crash
            if not spec.required:
                logger.warning("optional engine %s skipped: %s", spec.engine_file, exc)
                statuses.append(EngineStatus(spec.engine_file, spec.env_var, "skipped_optional"))
                continue
            statuses.append(_classify_failure(spec, host, exc))
            continue
        state = "cache" if _meta_matches(spec.engine_path, host) else "hf_bundle"
        if export_ok:
            os.environ[spec.env_var] = str(spec.engine_path)
        statuses.append(EngineStatus(spec.engine_file, spec.env_var, state))

    report = ProvisioningReport(
        host_signature=host.key,
        supported_signatures=_available_signatures(entries[0]["model_id"]) if entries else [],
        engines=statuses,
    )
    _LAST_REPORT = report
    return report


def resolve_all(profile: dict) -> dict[str, Path]:
    """Resolve every engine declared by ``profile['required_engines']``.

    On success, returns a dict of ``env_var → engine_path`` and also injects
    each entry into ``os.environ`` so backend modules can read them at import
    time. On failure, resolves ALL engines first (so the operator sees every
    problem at once) and raises ``EngineResolutionError`` (a RuntimeError) whose
    message is the full actionable F1..F7 report. The report is also stashed for
    ``/readyz`` via ``get_last_report()``.
    """
    entries = profile.get("required_engines") or []
    if not entries:
        logger.info("profile declares no required_engines — skipping resolver")
        return {}

    fd = _acquire_lock()
    try:
        report = build_report(profile, export_ok=True)
    finally:
        _release_lock(fd)

    if not report.ok:
        text = format_report_text(report)
        logger.error("%s", text)
        raise EngineResolutionError(
            f"{len(report.failures)} required engine(s) could not be provisioned.\n{text}",
            report,
        )

    return {
        e.env_var: Path(os.environ[e.env_var])
        for e in report.engines
        if e.state in ("cache", "hf_bundle")
    }


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
