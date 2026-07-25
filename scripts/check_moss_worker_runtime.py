#!/usr/bin/env python3
"""Fail-fast dynamic-link preflight for the native MOSS worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


_ORT_RESOLVED = re.compile(r"^\s*libonnxruntime\.so\.1\s+=>\s+/\S+", re.MULTILINE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha256_from_release_lock(
    lock_path: Path, artifact_path: str
) -> str:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read release lock {lock_path}: {error}") from error
    if lock.get("schema_version") != 1:
        raise ValueError("release lock schema_version must be 1")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("release lock artifacts must be an object")
    record = artifacts.get(artifact_path)
    if not isinstance(record, dict):
        raise ValueError(f"release lock has no artifact record for {artifact_path}")
    expected = record.get("sha256")
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise ValueError(
            f"release lock has invalid sha256 for {artifact_path}"
        )
    return expected


def validate_ldd_output(output: str) -> list[str]:
    """Return actionable dynamic-link errors found in ``ldd -r`` output."""
    errors: list[str] = []
    if not _ORT_RESOLVED.search(output):
        errors.append("libonnxruntime.so.1 did not resolve to an absolute path")
    if re.search(r"^\s*\S+\s+=>\s+not found\s*$", output, re.MULTILINE):
        errors.append("one or more shared libraries are not found")
    if re.search(r"\bversion\s+\S+\s+not found\b", output):
        errors.append("the worker requires an unavailable symbol version")
    if "undefined symbol:" in output:
        errors.append("one or more versioned symbols are undefined")
    return errors


def check_worker(
    worker: Path,
    *,
    expected_sha256: str | None = None,
    ldd: str = "ldd",
) -> tuple[str, list[str]]:
    if not worker.is_file():
        return "", [f"MOSS worker is missing: {worker}"]
    if not worker.stat().st_mode & 0o111:
        return "", [f"MOSS worker is not executable: {worker}"]

    errors: list[str] = []
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(worker)
        if actual_sha256 != expected_sha256:
            errors.append(
                "MOSS worker SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    result = subprocess.run(
        [ldd, "-r", str(worker)],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    errors.extend(validate_ldd_output(output))
    if result.returncode != 0:
        errors.append(f"ldd -r exited with status {result.returncode}")
    return output, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--release-lock", type=Path)
    parser.add_argument(
        "--artifact-path", default="bin/moss_tts_nano_worker"
    )
    parser.add_argument("--ldd", default="ldd")
    args = parser.parse_args()

    expected_sha256 = None
    if args.release_lock is not None:
        try:
            expected_sha256 = expected_sha256_from_release_lock(
                args.release_lock, args.artifact_path
            )
        except ValueError as error:
            print(f"MOSS runtime preflight failed: {error}")
            return 1

    output, errors = check_worker(
        args.worker,
        expected_sha256=expected_sha256,
        ldd=args.ldd,
    )
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if errors:
        for error in errors:
            print(f"MOSS runtime preflight failed: {error}")
        return 1
    print(f"MOSS runtime preflight OK: {args.worker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
