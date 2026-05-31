#!/usr/bin/env python3
"""Probe Kokoro split-generator bucket selection and finite outputs."""

from __future__ import annotations

import os
import sys

import numpy as np

from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend, _run_cpu_onnx


def finite(name: str, value: np.ndarray) -> None:
    arr = np.asarray(value)
    print(
        name,
        "shape=", tuple(arr.shape),
        "dtype=", arr.dtype,
        "finite=", bool(np.isfinite(arr).all()),
    )


def main() -> int:
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = (
            "This is a deliberately long Kokoro TensorRT validation sentence without extra "
            "punctuation so that the length regulator has a chance to produce more than two "
            "hundred and fifty six acoustic frames while still remaining inside the encoder "
            "token limit."
        )
    os.environ.setdefault("KOKORO_TRT_RUNTIME", "split_generator")
    backend = KokoroTRTBackend()
    backend.preload()
    token_ids = backend._text_to_token_ids(text)
    max_tokens = max(1, (backend._hybrid_max_seq_len or 128) - 2)
    token_ids = token_ids[:max_tokens]
    ids = np.array([[0, *token_ids, 0]], dtype=np.int64)
    style = backend._load_style(52, len(token_ids))
    speed = np.array([float(os.environ.get("KOKORO_PROBE_SPEED", "1.0"))], dtype=np.float32)
    stage = {"tokens": ids, "style": style, "speed": speed}

    stage.update(backend._run_named_trt_engine("encoder", stage))
    stage.update(_run_cpu_onnx(backend._split_length_sess, stage))
    frame_t = int(stage["/encoder/MatMul_1_output_0"].shape[2])
    bucket_engines, bucket_ctxs = backend._select_split_bucket(frame_t)
    print("tokens", len(token_ids), "frame_t", frame_t, "long_bucket", bucket_engines is backend._split_long_engines)

    stage.update(backend._run_split_bucket_engine(bucket_engines, bucket_ctxs, "decoder", {
        "/encoder/MatMul_1_output_0": stage["/encoder/MatMul_1_output_0"],
        "/decoder/decoder/F0_conv/Conv_output_0": stage["/decoder/decoder/F0_conv/Conv_output_0"],
        "/decoder/decoder/N_conv/Conv_output_0": stage["/decoder/decoder/N_conv/Conv_output_0"],
        "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
        "style": stage["style"],
    }))
    finite("decoder", stage["/decoder/decoder/decode.3/Div_4_output_0"])

    stage.update(backend._run_split_bucket_engine(bucket_engines, bucket_ctxs, "source", {
        "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
    }))
    finite("source", stage["/decoder/decoder/generator/Concat_3_output_0"])

    gen = backend._run_split_bucket_engine(bucket_engines, bucket_ctxs, "generator", {
        "/decoder/decoder/decode.3/Div_4_output_0": stage["/decoder/decoder/decode.3/Div_4_output_0"],
        "/decoder/decoder/generator/Concat_3_output_0": stage["/decoder/decoder/generator/Concat_3_output_0"],
        "style": stage["style"],
    })
    for name, arr in gen.items():
        finite(name, arr)
    audio = _run_cpu_onnx(backend._split_istft_sess, gen)["audio"]
    finite("audio", audio)
    return 0 if np.isfinite(audio).all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
