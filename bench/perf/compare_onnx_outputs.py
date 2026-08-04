#!/usr/bin/env python3
"""Compare two single-input ONNX models on deterministic random input."""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--shape", required=True, help="comma-separated input shape")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.shape.split(","))
    data = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    outputs = []
    for model_path in (args.reference, args.candidate):
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        outputs.append(session.run(None, {session.get_inputs()[0].name: data})[0])
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=args.rtol, atol=args.atol)
    print(
        f"equivalent shape={shape} output={outputs[0].shape} "
        f"max_abs={float(np.max(np.abs(outputs[0] - outputs[1]))):.9g}"
    )


if __name__ == "__main__":
    main()
