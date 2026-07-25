#!/usr/bin/env python3
"""Fail-fast dynamic-link preflight for the native MOSS worker."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


_ORT_RESOLVED = re.compile(r"^\s*libonnxruntime\.so\.1\s+=>\s+/\S+", re.MULTILINE)


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


def check_worker(worker: Path, *, ldd: str = "ldd") -> tuple[str, list[str]]:
    if not worker.is_file():
        return "", [f"MOSS worker is missing: {worker}"]
    if not worker.stat().st_mode & 0o111:
        return "", [f"MOSS worker is not executable: {worker}"]

    result = subprocess.run(
        [ldd, "-r", str(worker)],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    errors = validate_ldd_output(output)
    if result.returncode != 0:
        errors.append(f"ldd -r exited with status {result.returncode}")
    return output, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--ldd", default="ldd")
    args = parser.parse_args()

    output, errors = check_worker(args.worker, ldd=args.ldd)
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
