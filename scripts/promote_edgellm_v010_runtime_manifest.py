#!/usr/bin/env python3
"""Promote an existing v0.10 package to the profile-aware runtime contract.

The immutable payload is hard-linked into a new package directory; only the
small manifest and repository-root checksum file change.  The source package
is verified and never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from package_edgellm_v010_artifact import (
    MANIFEST_NAME,
    PAYLOAD_NAME,
    SUMS_NAME,
    PackageError,
    sha256_file,
)


def _regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise PackageError(f"{label} must be a regular file: {path}")


def _verify_source(source: Path) -> tuple[dict, str, int]:
    if not source.is_dir() or source.is_symlink():
        raise PackageError(f"source package must be a regular directory: {source}")
    manifest_path = source / MANIFEST_NAME
    payload_path = source / PAYLOAD_NAME
    sums_path = source / SUMS_NAME
    for path, label in (
        (manifest_path, "manifest"),
        (payload_path, "payload"),
        (sums_path, "checksum file"),
    ):
        _regular(path, label)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read source manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise PackageError("source manifest must use schema_version 2")
    payload = manifest.get("payload")
    if not isinstance(payload, dict) or payload.get("path") != PAYLOAD_NAME:
        raise PackageError("source manifest has an invalid payload lock")
    digest = sha256_file(payload_path)
    size = payload_path.stat().st_size
    if payload.get("sha256") != digest or payload.get("size") != size:
        raise PackageError("source payload does not match its manifest lock")
    expected_sums = (
        f"{sha256_file(manifest_path)}  {MANIFEST_NAME}\n"
        f"{digest}  {PAYLOAD_NAME}\n"
    )
    if sums_path.read_text(encoding="utf-8") != expected_sums:
        raise PackageError("source SHA256SUMS does not match manifest and payload")
    return manifest, digest, size


def promote(
    source: Path,
    output: Path,
    *,
    engine_profile: str,
    max_input_len: int,
    max_kv_cache_capacity: int,
) -> dict:
    source = source.resolve()
    output = output.resolve(strict=False)
    if os.path.lexists(output):
        raise PackageError(f"output already exists; refusing to overwrite: {output}")
    if engine_profile not in {"4k", "8k"}:
        raise PackageError("engine profile must be 4k or 8k")
    if max_input_len <= 0 or max_kv_cache_capacity <= 0:
        raise PackageError("engine contract limits must be positive")
    manifest, payload_digest, _ = _verify_source(source)
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise PackageError("source manifest provenance must be a JSON object")
    previous_profile = manifest.get("engine_profile")
    if previous_profile != engine_profile:
        provenance = dict(provenance)
        provenance.setdefault("build_profile", previous_profile)
    promoted = dict(manifest)
    promoted["provenance"] = provenance
    promoted["engine_profile"] = engine_profile
    promoted["engine_contract"] = {
        "max_input_len": max_input_len,
        "max_kv_cache_capacity": max_kv_cache_capacity,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        os.link(source / PAYLOAD_NAME, temporary / PAYLOAD_NAME)
        manifest_text = json.dumps(
            promoted, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        (temporary / MANIFEST_NAME).write_text(
            manifest_text, encoding="utf-8", newline="\n"
        )
        sums_text = (
            f"{sha256_file(temporary / MANIFEST_NAME)}  {MANIFEST_NAME}\n"
            f"{payload_digest}  {PAYLOAD_NAME}\n"
        )
        (temporary / SUMS_NAME).write_text(sums_text, encoding="utf-8", newline="\n")
        if os.path.lexists(output):
            raise PackageError(f"output appeared during promotion: {output}")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--engine-profile", required=True)
    parser.add_argument("--max-input-len", required=True, type=int)
    parser.add_argument("--max-kv-cache-capacity", required=True, type=int)
    args = parser.parse_args()
    try:
        manifest = promote(
            args.source,
            args.output,
            engine_profile=args.engine_profile,
            max_input_len=args.max_input_len,
            max_kv_cache_capacity=args.max_kv_cache_capacity,
        )
    except (PackageError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(args.output),
        "engine_profile": manifest["engine_profile"],
        "engine_contract": manifest["engine_contract"],
        "manifest_sha256": sha256_file(args.output / MANIFEST_NAME),
        "sums_sha256": sha256_file(args.output / SUMS_NAME),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
