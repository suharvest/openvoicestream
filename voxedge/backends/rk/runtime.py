"""Rockchip runtime compatibility checks — voxedge adapter.

adapted from app/core/rk_runtime.py (2026-05-30), dedup after registry switch.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ALL module-scope / function-scope ``os.environ.get(...)`` reads replaced
    by an explicit ``RKRuntimeConfig`` dataclass injected at call time. voxedge
    has no module-scope or hardcoded env reads (memory
    trt_edge_llm_tts_env_staleness: module-scope env breaks hot reload).
  * No ``app.*`` import.

The RK image vendors known-good userspace libraries. If an operator bind mounts
a different library over them, fail early with an actionable message instead of
letting RKNN/RKLLM fail later with opaque native errors.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("/opt/rk-runtime/MANIFEST.json")
DEFAULT_RKNNRT = Path("/usr/lib/librknnrt.so")
DEFAULT_RKLLMRT = Path("/opt/asr/lib/librkllmrt.so")


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                  → RKRuntimeConfig field
#   LANGUAGE_MODE                   → language_mode      (default None → skip)
#   RK_RUNTIME_STRICT               → strict             (default True)
#   RK_RUNTIME_MANIFEST             → manifest_path      (default DEFAULT_MANIFEST)
#   RKNNRT_LIB_PATH                 → rknnrt_lib_path    (default DEFAULT_RKNNRT)
#   RKLLM_LIB_PATH                  → rkllmrt_lib_path   (default DEFAULT_RKLLMRT)


@dataclass
class RKRuntimeConfig:
    """Explicit construction-time config for :func:`check_rk_runtime`.

    Every field default is identical to the production env default; nothing
    here reads ``os.environ``. ``manifest_path`` / library paths default to
    ``None`` and resolve to the module-level ``DEFAULT_*`` constants in
    ``__post_init__`` so the production defaults are preserved exactly.

    ``language_mode`` mirrors the old ``LANGUAGE_MODE`` env gate: the runtime
    check is a no-op unless it equals ``"rk"`` (matching the original early
    return when neither profile nor env had ``LANGUAGE_MODE == "rk"``).
    """

    language_mode: Optional[str] = None
    strict: bool = True
    manifest_path: Optional[Path] = None
    rknnrt_lib_path: Optional[Path] = None
    rkllmrt_lib_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.manifest_path is None:
            self.manifest_path = DEFAULT_MANIFEST
        elif not isinstance(self.manifest_path, Path):
            self.manifest_path = Path(self.manifest_path)
        if self.rknnrt_lib_path is None:
            self.rknnrt_lib_path = DEFAULT_RKNNRT
        elif not isinstance(self.rknnrt_lib_path, Path):
            self.rknnrt_lib_path = Path(self.rknnrt_lib_path)
        if self.rkllmrt_lib_path is None:
            self.rkllmrt_lib_path = DEFAULT_RKLLMRT
        elif not isinstance(self.rkllmrt_lib_path, Path):
            self.rkllmrt_lib_path = Path(self.rkllmrt_lib_path)


class RKRuntimeError(RuntimeError):
    """Raised when the Rockchip runtime does not match the image manifest."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_file(label: str, path: Path, expected: dict, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} missing at {path}")
        return
    got_size = path.stat().st_size
    got_sha = _sha256(path)
    exp_size = int(expected.get("size") or 0)
    exp_sha = str(expected.get("sha256") or "")
    if exp_size and got_size != exp_size:
        errors.append(f"{label} size mismatch: got {got_size}, expected {exp_size}")
    if exp_sha and got_sha != exp_sha:
        errors.append(f"{label} sha256 mismatch: got {got_sha}, expected {exp_sha}")
    logger.info("%s OK: %s sha256=%s", label, path, got_sha[:12])


def check_rk_runtime(config: Optional[RKRuntimeConfig] = None) -> None:
    """Validate the RK userspace runtime before RK backends import native libs.

    The RK image vendors known-good userspace libraries. If an operator bind
    mounts a different library over them, fail early with an actionable message
    instead of letting RKNN/RKLLM fail later with opaque native errors.

    No-op unless ``config.language_mode == "rk"`` (the production code gated on
    profile/env ``LANGUAGE_MODE == "rk"``; here the gate is an explicit field).
    """
    cfg = config or RKRuntimeConfig()
    if cfg.language_mode != "rk":
        return

    manifest_path = cfg.manifest_path
    if not manifest_path.exists():
        msg = f"RK runtime manifest missing at {manifest_path}"
        if cfg.strict:
            raise RKRuntimeError(msg)
        logger.warning("%s; continuing because strict=False", msg)
        return

    manifest = json.loads(manifest_path.read_text())
    runtime = manifest.get("runtime") or {}
    errors: list[str] = []

    expected_lite = str(runtime.get("rknn_toolkit_lite2") or "")
    try:
        got_lite = importlib.metadata.version("rknn-toolkit-lite2")
    except importlib.metadata.PackageNotFoundError:
        got_lite = ""
    if expected_lite and got_lite != expected_lite:
        errors.append(
            f"rknn-toolkit-lite2 mismatch: got {got_lite or 'missing'}, expected {expected_lite}"
        )
    else:
        logger.info("rknn-toolkit-lite2 OK: %s", got_lite)

    _check_file(
        "librknnrt",
        cfg.rknnrt_lib_path,
        runtime.get("librknnrt") or {},
        errors,
    )
    _check_file(
        "librkllmrt",
        cfg.rkllmrt_lib_path,
        runtime.get("rkllm_runtime") or {},
        errors,
    )

    if not errors:
        return

    guidance = (
        "Rockchip runtime version mismatch. Use the runtime libraries baked into "
        "this image, remove overriding host mounts for librknnrt/librkllmrt, or "
        "update the device BSP/runtime to the version declared in "
        f"{manifest_path}. If you intentionally use a different runtime, publish "
        "a matching RK artifact set and manifest; set strict=False only "
        "for debugging."
    )
    message = guidance + " Details: " + "; ".join(errors)
    if cfg.strict:
        raise RKRuntimeError(message)
    logger.warning(message)
