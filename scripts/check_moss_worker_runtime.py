#!/usr/bin/env python3
"""Fail-fast dynamic-link preflight for the native MOSS worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
from pathlib import Path
from typing import NamedTuple


_ORT_RESOLVED = re.compile(r"^\s*libonnxruntime\.so\.1\s+=>\s+/\S+", re.MULTILINE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODE = re.compile(r"^0[0-7]{3}$")
_ORT_SYMBOL_VERSION = re.compile(r"^VERS_[0-9]+\.[0-9]+\.[0-9]+$")


class ReleaseArtifact(NamedTuple):
    sha256: str
    size: int
    mode: int
    required_onnxruntime_symbol_version: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_from_release_lock(
    lock_path: Path, artifact_path: str
) -> ReleaseArtifact:
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
    size = record.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"release lock has invalid size for {artifact_path}")
    raw_mode = record.get("mode")
    if (
        not isinstance(raw_mode, str)
        or not _MODE.fullmatch(raw_mode)
        or raw_mode != "0755"
    ):
        raise ValueError(f"release lock has invalid mode for {artifact_path}")
    symbol_version = record.get("required_onnxruntime_symbol_version")
    if (
        not isinstance(symbol_version, str)
        or not _ORT_SYMBOL_VERSION.fullmatch(symbol_version)
    ):
        raise ValueError(
            "release lock has invalid required_onnxruntime_symbol_version "
            f"for {artifact_path}"
        )
    return ReleaseArtifact(
        sha256=expected,
        size=size,
        mode=int(raw_mode, 8),
        required_onnxruntime_symbol_version=symbol_version,
    )


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


def validate_nm_output(output: str, expected_version: str) -> list[str]:
    versions = set(
        re.findall(r"\bOrtGetApiBase@+(VERS_[^\s]+)", output)
    )
    if not versions:
        return ["the worker has no versioned OrtGetApiBase import"]
    if versions != {expected_version}:
        return [
            "the worker imports the wrong OrtGetApiBase symbol version: "
            f"expected {expected_version}, found {sorted(versions)}"
        ]
    return []


def check_worker(
    worker: Path,
    *,
    release_artifact: ReleaseArtifact | None = None,
    ldd: str = "ldd",
    nm: str = "nm",
) -> tuple[str, list[str]]:
    if not worker.is_file():
        return "", [f"MOSS worker is missing: {worker}"]
    if not worker.stat().st_mode & 0o111:
        return "", [f"MOSS worker is not executable: {worker}"]

    errors: list[str] = []
    if release_artifact is not None:
        worker_stat = worker.stat()
        actual_sha256 = sha256_file(worker)
        if actual_sha256 != release_artifact.sha256:
            errors.append(
                "MOSS worker SHA256 mismatch: "
                f"expected {release_artifact.sha256}, got {actual_sha256}"
            )
        if worker_stat.st_size != release_artifact.size:
            errors.append(
                "MOSS worker size mismatch: "
                f"expected {release_artifact.size}, got {worker_stat.st_size}"
            )
        actual_mode = stat.S_IMODE(worker_stat.st_mode)
        if actual_mode != release_artifact.mode:
            errors.append(
                "MOSS worker mode mismatch: "
                f"expected {release_artifact.mode:04o}, got {actual_mode:04o}"
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

    nm_result = subprocess.run(
        [nm, "-D", "--undefined-only", "--with-symbol-versions", str(worker)],
        check=False,
        capture_output=True,
        text=True,
    )
    nm_output = nm_result.stdout + nm_result.stderr
    if release_artifact is not None:
        errors.extend(
            validate_nm_output(
                nm_output,
                release_artifact.required_onnxruntime_symbol_version,
            )
        )
    if nm_result.returncode != 0:
        errors.append(f"nm exited with status {nm_result.returncode}")
    return output + nm_output, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--release-lock", type=Path)
    parser.add_argument(
        "--artifact-path", default="bin/moss_tts_nano_worker"
    )
    parser.add_argument("--ldd", default="ldd")
    parser.add_argument("--nm", default="nm")
    args = parser.parse_args()

    release_artifact = None
    if args.release_lock is not None:
        try:
            release_artifact = artifact_from_release_lock(
                args.release_lock, args.artifact_path
            )
        except ValueError as error:
            print(f"MOSS runtime preflight failed: {error}")
            return 1

    output, errors = check_worker(
        args.worker,
        release_artifact=release_artifact,
        ldd=args.ldd,
        nm=args.nm,
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
