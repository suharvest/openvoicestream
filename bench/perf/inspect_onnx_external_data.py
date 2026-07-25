#!/usr/bin/env python3
"""Inspect ONNX external-data ranges without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--load",
        action="store_true",
        help="Also load every external tensor payload and run ONNX checker.",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    model = onnx.load_model(model_path, load_external_data=False)
    files: dict[str, dict[str, int]] = {}
    tensors = 0
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        metadata = {entry.key: entry.value for entry in initializer.external_data}
        location = metadata.get("location")
        if not location:
            raise SystemExit(f"{initializer.name}: external tensor has no location")
        offset = int(metadata.get("offset", "0"))
        length = int(metadata.get("length", "0"))
        record = files.setdefault(
            location, {"tensor_count": 0, "required_bytes": 0, "actual_bytes": 0}
        )
        record["tensor_count"] += 1
        record["required_bytes"] = max(record["required_bytes"], offset + length)
        tensors += 1

    for location, record in files.items():
        external_path = model_path.parent / location
        record["actual_bytes"] = external_path.stat().st_size

    result: dict[str, object] = {
        "model": str(model_path),
        "external_tensor_count": tensors,
        "files": files,
    }
    if args.load:
        loaded = onnx.load_model(model_path, load_external_data=True)
        onnx.checker.check_model(loaded)
        result["external_payload_load"] = "pass"
        result["onnx_checker"] = "pass"

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(
        item["actual_bytes"] >= item["required_bytes"] for item in files.values()
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
