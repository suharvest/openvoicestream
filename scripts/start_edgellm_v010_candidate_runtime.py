#!/usr/bin/env python3
"""Fail-closed entry point for an explicitly opted-in v0.10 gray image."""

from __future__ import annotations

import os
import subprocess
import sys


ROOT = "/opt/speech/release-source"
LOCK = f"{ROOT}/deploy/artifacts/v010-candidate-release-lock.json"
CHECKER = "/opt/speech/scripts/check_edgellm_v010_release_lock.py"


def main() -> int:
    allow_candidate = os.environ.get(
        "EDGELLM_V010_ALLOW_UNPUBLISHED_CANDIDATE", "0"
    ).strip().lower() in {"1", "true", "yes"}
    if not allow_candidate:
        print(
            "v0.10 candidate runtime requires explicit "
            "EDGELLM_V010_ALLOW_UNPUBLISHED_CANDIDATE=1",
            file=sys.stderr,
        )
        return 2
    command = [
        sys.executable,
        CHECKER,
        "--lock",
        LOCK,
        "--repo-root",
        ROOT,
        "--skip-gitlink-check",
        "--require-candidate",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        return result.returncode
    os.execvp(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
