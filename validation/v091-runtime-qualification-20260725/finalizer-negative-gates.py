#!/usr/bin/env python3
"""Prove that the artifact finalizer rejects missing files and stale sidecars."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def invoke(finalizer: Path, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(finalizer), str(root), "--published-to-hf", "false"],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: finalizer-negative-gates.py FINALIZER WORK_ROOT")
    finalizer = Path(sys.argv[1]).resolve()
    work_root = Path(sys.argv[2]).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    fixture_root = Path(tempfile.mkdtemp(prefix="finalizer-negative-", dir=work_root))

    required = fixture_root / "required"
    required.mkdir()
    (required / "manifest.json").write_text(
        json.dumps({"required_files": ["missing.engine"]}) + "\n",
        encoding="utf-8",
    )
    required_result = invoke(finalizer, required)

    sidecar = fixture_root / "sidecar"
    sidecar.mkdir()
    engine = sidecar / "sample.engine"
    engine.write_bytes(b"engine")
    (sidecar / "sample.engine.meta.json").write_text(
        json.dumps(
            {
                "engine_sha256": "0" * 64,
                "host": {
                    "sm": "87",
                    "trt_version": "10.3.0.30",
                    "jp_version": "6.2",
                    "cuda_version": "12.6",
                    "platform": "Jetson Orin NX",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (sidecar / "manifest.json").write_text("{}\n", encoding="utf-8")
    sidecar_result = invoke(finalizer, sidecar)

    evidence = {
        "fixture_root": str(fixture_root),
        "required_missing": {
            "returncode": required_result.returncode,
            "stderr": required_result.stderr.strip(),
            "rejected": required_result.returncode != 0
            and "required release payload is missing" in required_result.stderr,
        },
        "sidecar_digest_mismatch": {
            "actual_engine_sha256": hashlib.sha256(b"engine").hexdigest(),
            "returncode": sidecar_result.returncode,
            "stderr": sidecar_result.stderr.strip(),
            "rejected": sidecar_result.returncode != 0
            and "sidecar digest mismatch" in sidecar_result.stderr,
        },
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if all(
        gate["rejected"]
        for gate in (
            evidence["required_missing"],
            evidence["sidecar_digest_mismatch"],
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
