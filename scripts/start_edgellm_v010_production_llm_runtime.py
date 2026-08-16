#!/usr/bin/env python3
"""Gate the v0.10 LLM image before handing off to its native entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys


ROOT = "/opt/edgellm-release-source"
LOCK = f"{ROOT}/deploy/artifacts/v010-embedded-build-lock.json"
CHECKER = "/opt/edgellm/check_edgellm_v010_release_lock.py"


def main() -> int:
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
            "/opt/edgellm-v010",
            "--image-key",
            "llm",
            "--expected-service-revision",
            os.environ.get("EDGELLM_V010_SERVICE_REVISION", ""),
        ],
        check=False,
    )
    if result.returncode != 0:
        return result.returncode
    os.execv("/app/entrypoint.sh", ["/app/entrypoint.sh", *sys.argv[1:]])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
