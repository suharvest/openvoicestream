#!/usr/bin/env python3
"""Apply the candidate Qwen3-ASR MRope config normalization atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    path = args.config.resolve()
    before = path.read_bytes()
    config = json.loads(before)
    if config.get("model") != "qwen3asrthinker":
        raise SystemExit(f"refusing non-Qwen3-ASR config: model={config.get('model')!r}")
    rope_scaling = config.get("rope_scaling")
    if not isinstance(rope_scaling, dict) or "mrope_section" not in rope_scaling:
        raise SystemExit("refusing config without rope_scaling.mrope_section")

    previous = {
        "rope_type": rope_scaling.get("rope_type"),
        "type": rope_scaling.get("type"),
    }
    rope_scaling["rope_type"] = "mrope"
    rope_scaling["type"] = "mrope"
    after = (json.dumps(config, indent=2) + "\n").encode()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(after)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    print(
        json.dumps(
            {
                "config": str(path),
                "previous": previous,
                "current": {"rope_type": "mrope", "type": "mrope"},
                "sha256_before": digest(before),
                "sha256_after": digest(after),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
