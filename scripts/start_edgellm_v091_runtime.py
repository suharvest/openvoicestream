#!/usr/bin/env python3
"""Validate the image-owned v0.9.1 runtime before starting the API."""

from __future__ import annotations

import os
import subprocess
import sys


CHECKER = "/opt/speech/scripts/check_moss_worker_runtime.py"
RELEASE_LOCK = "/opt/speech/deploy/v091-release-lock.json"
IMAGE_WORKER = "/opt/edgellm-v091/bin/moss_tts_nano_worker"
ORT_LIBRARY_DIR = (
    "/usr/local/lib/python3.10/dist-packages/onnxruntime/capi"
)


def prepend_ort_library_path() -> None:
    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        ORT_LIBRARY_DIR + (f":{inherited}" if inherited else "")
    )


def main() -> int:
    prepend_ort_library_path()
    result = subprocess.run(
        [
            sys.executable,
            CHECKER,
            "--worker",
            IMAGE_WORKER,
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
