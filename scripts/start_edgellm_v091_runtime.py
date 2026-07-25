#!/usr/bin/env python3
"""Validate the mounted v0.9.1 release worker before starting the API."""

from __future__ import annotations

import os
import subprocess
import sys


CHECKER = "/opt/speech/scripts/check_moss_worker_runtime.py"
RELEASE_LOCK = "/opt/speech/deploy/v091-release-lock.json"
MOUNTED_WORKER = "/opt/edgellm-v091/bin/moss_tts_nano_worker"


def main() -> int:
    result = subprocess.run(
        [
            sys.executable,
            CHECKER,
            "--worker",
            MOUNTED_WORKER,
            "--release-lock",
            RELEASE_LOCK,
        ],
        check=False,
    )
    if result.returncode != 0:
        return result.returncode
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
