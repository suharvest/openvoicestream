#!/usr/bin/env python3
"""Compare two Kokoro ONNX graphs on the same real tokenized text input."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _make_inputs(model_base: str, text: str, speaker_id: int, speed: float) -> dict[str, np.ndarray]:
    os.environ["KOKORO_MODEL_BASE"] = model_base
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend

    backend = KokoroTRTBackend()
    backend._load_tokens()
    token_ids = backend._text_to_token_ids(text)
    ids = np.array([[0, *token_ids, 0]], dtype=np.int64)
    return {
        "tokens": ids,
        "style": backend._load_style(speaker_id, len(token_ids)),
        "speed": np.array([speed], dtype=np.float32),
    }


def _run(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = {item.name for item in sess.get_inputs()}
    actual_feeds = {name: value for name, value in feeds.items() if name in names}
    output_names = [item.name for item in sess.get_outputs()]
    outputs = sess.run(output_names, actual_feeds)
    return dict(zip(output_names, outputs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--model-base", default="/opt/models/kokoro-multi-lang-v1_0")
    ap.add_argument("--text", default="Hello, this is a TensorRT Kokoro test.")
    ap.add_argument("--speaker-id", type=int, default=52)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    feeds = _make_inputs(args.model_base, args.text, args.speaker_id, args.speed)
    outs_a = _run(Path(args.a), feeds)
    outs_b = _run(Path(args.b), feeds)
    for name, out_a in outs_a.items():
        if name not in outs_b:
            print(f"{name}: missing_in_b")
            continue
        out_b = outs_b[name]
        if out_a.shape != out_b.shape:
            print(f"{name}: shape_a={out_a.shape} shape_b={out_b.shape} shape_mismatch")
            continue
        diff = np.asarray(out_a, dtype=np.float32) - np.asarray(out_b, dtype=np.float32)
        print(f"{name}: shape_a={out_a.shape} shape_b={out_b.shape}")
        print(f"{name}: max_abs_diff={float(np.max(np.abs(diff)))}")
        print(f"{name}: mean_abs_diff={float(np.mean(np.abs(diff)))}")
        print(f"{name}: allclose={bool(np.allclose(out_a, out_b, rtol=1e-4, atol=1e-4))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
