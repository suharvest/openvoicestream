#!/usr/bin/env python3
"""Probe Kokoro decoder TRT outputs after encoder TRT + CPU length regulator."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from smoke_kokoro_target_hybrid import TrtRunner


def _ort(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _run_ort(sess: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = {item.name for item in sess.get_inputs()}
    output_names = [item.name for item in sess.get_outputs()]
    actual = {name: value for name, value in feeds.items() if name in input_names}
    return dict(zip(output_names, sess.run(output_names, actual)))


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


def _print_stats(stage: str, tensors: dict[str, np.ndarray]) -> None:
    for name, arr in tensors.items():
        if not np.issubdtype(arr.dtype, np.floating):
            print(f"{stage}:{name}: shape={arr.shape} dtype={arr.dtype}")
            continue
        finite = np.isfinite(arr)
        print(
            f"{stage}:{name}: shape={arr.shape} dtype={arr.dtype} "
            f"finite={int(finite.sum())}/{arr.size} "
            f"min={float(np.nanmin(arr))} max={float(np.nanmax(arr))}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base", default="/opt/models/kokoro-multi-lang-v1_0")
    ap.add_argument("--encoder-engine", default="/opt/models/kokoro-multi-lang-v1_0/hybrid2/kokoro_prefix_encoder_dyn4_128_fp16.engine")
    ap.add_argument("--length-regulator", default="/opt/models/kokoro-multi-lang-v1_0/cpu_length_regulator.onnx")
    ap.add_argument("--decoder-engine", required=True)
    ap.add_argument("--text", default="Hello, this is a TensorRT Kokoro test.")
    ap.add_argument("--speaker-id", type=int, default=52)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    feeds = _make_inputs(args.model_base, args.text, args.speaker_id, args.speed)
    encoder = TrtRunner(args.encoder_engine)
    lr = _ort(args.length_regulator)
    decoder = TrtRunner(args.decoder_engine)

    t0 = time.perf_counter()
    enc = encoder.run({k: feeds[k] for k in ("tokens", "style", "speed")})
    print(f"encoder_trt_ms={(time.perf_counter() - t0) * 1000:.3f}")
    _print_stats("encoder", enc)

    stage = dict(feeds)
    stage.update(enc)
    t0 = time.perf_counter()
    lr_out = _run_ort(lr, stage)
    print(f"length_cpu_ms={(time.perf_counter() - t0) * 1000:.3f}")
    _print_stats("length", lr_out)

    stage.update(lr_out)
    t0 = time.perf_counter()
    dec = decoder.run({
        "/encoder/MatMul_1_output_0": stage["/encoder/MatMul_1_output_0"],
        "/decoder/decoder/F0_conv/Conv_output_0": stage["/decoder/decoder/F0_conv/Conv_output_0"],
        "/decoder/decoder/N_conv/Conv_output_0": stage["/decoder/decoder/N_conv/Conv_output_0"],
        "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
        "style": feeds["style"],
    })
    print(f"decoder_trt_ms={(time.perf_counter() - t0) * 1000:.3f}")
    _print_stats("decoder", dec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
