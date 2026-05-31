#!/usr/bin/env python3
"""Smoke test Kokoro split generator hybrid path."""

from __future__ import annotations

import argparse
import os
import time

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


def _stats(name: str, arr: np.ndarray) -> None:
    if not np.issubdtype(arr.dtype, np.floating):
        print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
        return
    finite = np.isfinite(arr)
    print(
        f"{name}: shape={arr.shape} dtype={arr.dtype} finite={int(finite.sum())}/{arr.size} "
        f"min={float(np.nanmin(arr))} max={float(np.nanmax(arr))}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base", default="/opt/models/kokoro-multi-lang-v1_0")
    ap.add_argument("--full", default="/opt/models/kokoro-multi-lang-v1_0/model_stft_conv.onnx")
    ap.add_argument("--encoder-engine", default="/opt/models/kokoro-multi-lang-v1_0/hybrid2/kokoro_prefix_encoder_dyn4_128_fp16.engine")
    ap.add_argument("--length-regulator", default="/opt/models/kokoro-multi-lang-v1_0/cpu_length_regulator.onnx")
    ap.add_argument("--decoder-engine", required=True)
    ap.add_argument("--source", default="/opt/models/kokoro-multi-lang-v1_0/cpu_generator_source.onnx")
    ap.add_argument("--source-engine", default="")
    ap.add_argument("--generator-engine", required=True)
    ap.add_argument("--istft", default="/opt/models/kokoro-multi-lang-v1_0/cpu_postspec_istft.onnx")
    ap.add_argument("--text", default="Hello, this is a TensorRT Kokoro test.")
    ap.add_argument("--speaker-id", type=int, default=52)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    feeds = _make_inputs(args.model_base, args.text, args.speaker_id, args.speed)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    full_audio = _run_ort(_ort(args.full), feeds)["audio"]
    timings["full_cpu_ms"] = (time.perf_counter() - t0) * 1000

    encoder = TrtRunner(args.encoder_engine)
    decoder = TrtRunner(args.decoder_engine)
    source_trt = TrtRunner(args.source_engine) if args.source_engine else None
    generator = TrtRunner(args.generator_engine)
    lr = _ort(args.length_regulator)
    source = _ort(args.source)
    istft = _ort(args.istft)

    t0 = time.perf_counter()
    enc = encoder.run({k: feeds[k] for k in ("tokens", "style", "speed")})
    timings["encoder_trt_ms"] = (time.perf_counter() - t0) * 1000

    stage = dict(feeds)
    stage.update(enc)
    t0 = time.perf_counter()
    lr_out = _run_ort(lr, stage)
    timings["length_cpu_ms"] = (time.perf_counter() - t0) * 1000
    stage.update(lr_out)

    t0 = time.perf_counter()
    dec = decoder.run({
        "/encoder/MatMul_1_output_0": stage["/encoder/MatMul_1_output_0"],
        "/decoder/decoder/F0_conv/Conv_output_0": stage["/decoder/decoder/F0_conv/Conv_output_0"],
        "/decoder/decoder/N_conv/Conv_output_0": stage["/decoder/decoder/N_conv/Conv_output_0"],
        "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
        "style": feeds["style"],
    })
    timings["decoder_trt_ms"] = (time.perf_counter() - t0) * 1000
    stage.update(dec)

    t0 = time.perf_counter()
    if source_trt is not None:
        src = source_trt.run({
            "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
        })
        timings["source_trt_ms"] = (time.perf_counter() - t0) * 1000
    else:
        src = _run_ort(source, stage)
        timings["source_cpu_ms"] = (time.perf_counter() - t0) * 1000
    stage.update(src)

    t0 = time.perf_counter()
    gen = generator.run({
        "/decoder/decoder/decode.3/Div_4_output_0": stage["/decoder/decoder/decode.3/Div_4_output_0"],
        "/decoder/decoder/generator/Concat_3_output_0": stage["/decoder/decoder/generator/Concat_3_output_0"],
        "style": feeds["style"],
    })
    timings["generator_trt_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    audio = _run_ort(istft, gen)["audio"]
    timings["istft_cpu_ms"] = (time.perf_counter() - t0) * 1000
    timings["hybrid_ms"] = sum(v for k, v in timings.items() if k != "full_cpu_ms")

    for name, arr in {**dec, **src, **gen, "audio": audio}.items():
        _stats(name, arr)
    diff = np.asarray(full_audio, dtype=np.float32) - np.asarray(audio, dtype=np.float32)
    for key, value in timings.items():
        print(f"{key}={value:.3f}")
    print(f"max_abs_diff={float(np.max(np.abs(diff)))}")
    print(f"mean_abs_diff={float(np.mean(np.abs(diff)))}")
    print(f"allclose_1e3={bool(np.allclose(full_audio, audio, rtol=1e-3, atol=1e-3))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
