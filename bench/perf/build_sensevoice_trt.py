#!/usr/bin/env python3
"""Build and validate a SenseVoice TensorRT plan with the project builder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder-module", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--plan", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    module_path = Path(args.builder_module)
    spec = importlib.util.spec_from_file_location("sensevoice_model_downloader", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import builder module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    onnx_path = Path(args.onnx)
    plan_path = Path(args.plan)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    module._build_sensevoice_trt_engine(str(onnx_path), str(plan_path))

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    if engine is None:
        raise RuntimeError("fresh SenseVoice plan failed to deserialize")
    tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensors.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
            }
        )
    report = {
        "tensorrt": trt.__version__,
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "plan": str(plan_path),
        "plan_bytes": plan_path.stat().st_size,
        "plan_sha256": sha256(plan_path),
        "io_tensors": tensors,
    }
    report_path = plan_path.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
