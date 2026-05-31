#!/usr/bin/env python3
"""Check Kokoro segmented ONNX chain against the full graph."""

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
    return {
        "tokens": np.array([[0, *token_ids, 0]], dtype=np.int64),
        "style": backend._load_style(speaker_id, len(token_ids)),
        "speed": np.array([speed], dtype=np.float32),
    }


def _run(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in sess.get_inputs()}
    output_names = [item.name for item in sess.get_outputs()]
    actual = {name: value for name, value in feeds.items() if name in input_names}
    return dict(zip(output_names, sess.run(output_names, actual)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--encoder-prefix", required=True)
    ap.add_argument("--length-regulator", required=True)
    ap.add_argument("--decoder-prefix", required=True)
    ap.add_argument("--istft", required=True)
    ap.add_argument("--model-base", default="/opt/models/kokoro-multi-lang-v1_0")
    ap.add_argument("--text", default="Hello, this is a TensorRT Kokoro test.")
    ap.add_argument("--speaker-id", type=int, default=52)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    feeds = _make_inputs(args.model_base, args.text, args.speaker_id, args.speed)
    full = _run(Path(args.full), feeds)["audio"]

    stage = dict(feeds)
    for path in (args.encoder_prefix, args.length_regulator, args.decoder_prefix, args.istft):
        outs = _run(Path(path), stage)
        stage.update(outs)
        print(f"stage={Path(path).name}")
        for name, arr in outs.items():
            print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")
    segmented = stage["audio"]
    diff = np.asarray(full, dtype=np.float32) - np.asarray(segmented, dtype=np.float32)
    print(f"full_shape={full.shape} segmented_shape={segmented.shape}")
    print(f"max_abs_diff={float(np.max(np.abs(diff)))}")
    print(f"mean_abs_diff={float(np.mean(np.abs(diff)))}")
    print(f"allclose={bool(np.allclose(full, segmented, rtol=1e-4, atol=1e-4))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
