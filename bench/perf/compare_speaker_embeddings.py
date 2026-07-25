#!/usr/bin/env python3
"""Compute cosine similarity between Base64 FP32 speaker embeddings."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np


def load(path: str) -> np.ndarray:
    raw = base64.b64decode(Path(path).read_text().strip(), validate=True)
    vector = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if vector.shape != (1024,) or not np.isfinite(vector).all():
        raise RuntimeError(f"invalid speaker embedding: {path} {vector.shape}")
    return vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    results = []
    for path in args.candidate:
        candidate = load(path)
        similarity = float(
            np.dot(reference, candidate)
            / (np.linalg.norm(reference) * np.linalg.norm(candidate))
        )
        results.append({"path": path, "cosine_similarity": similarity})
    report = {"reference": args.reference, "results": results}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
