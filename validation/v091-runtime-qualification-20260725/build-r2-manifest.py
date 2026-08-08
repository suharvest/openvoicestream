#!/usr/bin/env python3
"""Build the immutable-metadata template for the Orin NX v0.9.1 r2 set."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ARTIFACT_SET = "orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2"
OUTER_SHA = "021112eda3207a57ae91056f24d198303574b555"
ENGINE_OVERLAY_SHA = "4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f"
FORMAL_DIFF_SHA256 = (
    "32f24d15bcb094d41a937f0b3d11fa0b0b907dd6c7f97e7bea9871a93fe88a58"
)


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: build-r2-manifest.py OLD_MANIFEST QWEN3_MANIFEST OUTPUT"
        )
    old_path, qwen_path, output_path = map(Path, sys.argv[1:])
    manifest = json.loads(old_path.read_text(encoding="utf-8"))
    qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
    release = qwen["artifact_sets"][ARTIFACT_SET]

    manifest.update(
        {
            "artifact_set": ARTIFACT_SET,
            "hf_prefix": release["hf_prefix"],
            "published_to_hf": False,
            "upstream_repo": release["upstream_repo"],
            "upstream_sha": release["upstream_sha"],
            "required_files": release["required_files"],
            "proposed_upstream_patch_count": release[
                "proposed_upstream_patch_count"
            ],
            "proposed_upstream_series_sha256": release[
                "proposed_upstream_series_sha256"
            ],
            "overlay_patch_count": release["overlay_patch_count"],
            "overlay_series_sha256": release["overlay_series_sha256"],
            "overlay_checksums_sha256": release["overlay_checksums_sha256"],
            "provenance": {
                "outer_sha": OUTER_SHA,
                "engine_overlay_sha": ENGINE_OVERLAY_SHA,
                "formal_diff_sha256": FORMAL_DIFF_SHA256,
                "upstream_sha": release["upstream_sha"],
            },
            "files": [],
        }
    )
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
