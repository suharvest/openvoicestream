#!/usr/bin/env python3
"""Start a v0.10 speech image only from an embedded build-ready lock.

The embedded lock intentionally predates the image digest. The external
published lock owns that final identity; this entrypoint verifies everything
that was knowable before the image was built, including the bytes in its
runtime artifact directory.
"""

from __future__ import annotations

import os
import subprocess
import sys


ROOT = "/opt/speech/release-source"
LOCK = f"{ROOT}/deploy/artifacts/v010-embedded-build-lock.json"
CHECKER = "/opt/speech/scripts/check_edgellm_v010_release_lock.py"
RUNTIME_ROOT = "/opt/edgellm-v010"


def main() -> int:
    profile = os.environ.get("OVS_PROFILE", "")
    if not profile.startswith("jetson-edgellm-v010-") or "candidate" in profile:
        print("production v0.10 runtime requires a non-candidate v0.10 profile", file=sys.stderr)
        return 2
    service_revision = os.environ.get("EDGELLM_V010_SERVICE_REVISION", "")
    result = subprocess.run(
        [
            sys.executable,
            CHECKER,
            "--lock",
            LOCK,
            "--repo-root",
            ROOT,
            "--skip-gitlink-check",
            "--require-image-build-ready",
            "--runtime-root",
            RUNTIME_ROOT,
            "--image-key",
            "speech",
            "--expected-service-revision",
            service_revision,
        ],
        check=False,
    )
    if result.returncode != 0:
        return result.returncode
    os.execvp(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"],
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
