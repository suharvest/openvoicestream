#!/usr/bin/env python3
"""Smoke test target Kokoro path:

TRT encoder prefix -> CPU length regulator -> TRT decoder/generator prefix ->
CPU ISTFT.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from voxedge.backends.jetson.matcha_trt import CudaMemoryPool


class TrtRunner:
    def __init__(self, engine_path: str):
        import tensorrt as trt

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.pool = CudaMemoryPool()

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import tensorrt as trt

        pool = self.pool
        ctx = self.ctx
        for name, arr in feeds.items():
            arr = np.ascontiguousarray(arr)
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            ctx.set_tensor_address(name, ptr)
            try:
                ctx.set_input_shape(name, tuple(arr.shape))
            except Exception:
                pass

        outputs: dict[str, np.ndarray] = {}
        output_ptrs: list[tuple[str, int, np.ndarray]] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(int(d) for d in ctx.get_tensor_shape(name))
            dtype = _trt_dtype_to_np(self.engine.get_tensor_dtype(name))
            out = np.empty(shape, dtype=dtype)
            ptr = pool.allocate(out.nbytes)
            ctx.set_tensor_address(name, ptr)
            output_ptrs.append((name, ptr, out))

        if not ctx.execute_async_v3(pool.stream_handle()):
            pool.free_all()
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        pool.synchronize()
        for name, ptr, out in output_ptrs:
            pool.copy_dtoh(ptr, out)
            outputs[name] = out
        pool.free_all()
        return outputs


def _trt_dtype_to_np(dtype):
    import tensorrt as trt

    if dtype == trt.float32:
        return np.float32
    if dtype == trt.float16:
        return np.float16
    if dtype == trt.int32:
        return np.int32
    if dtype == trt.int64:
        return np.int64
    if dtype == trt.bool:
        return np.bool_
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base", default="/opt/models/kokoro-multi-lang-v1_0")
    ap.add_argument("--full", default="/opt/models/kokoro-multi-lang-v1_0/model_stft_conv.onnx")
    ap.add_argument("--encoder-engine", default="/opt/models/kokoro-multi-lang-v1_0/hybrid2/kokoro_prefix_encoder_dyn4_128_fp16.engine")
    ap.add_argument("--length-regulator", default="/opt/models/kokoro-multi-lang-v1_0/cpu_length_regulator.onnx")
    ap.add_argument("--decoder-engine", default="/opt/models/kokoro-multi-lang-v1_0/kokoro_decoder_generator_prefix_t150_fp16.engine")
    ap.add_argument("--istft", default="/opt/models/kokoro-multi-lang-v1_0/cpu_istft.onnx")
    ap.add_argument("--text", default="Hello, this is a TensorRT Kokoro test.")
    ap.add_argument("--speaker-id", type=int, default=52)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    feeds = _make_inputs(args.model_base, args.text, args.speaker_id, args.speed)
    timings: dict[str, float] = {}

    full_sess = _ort(args.full)
    t0 = time.perf_counter()
    full_audio = _run_ort(full_sess, feeds)["audio"]
    timings["full_cpu_ms"] = (time.perf_counter() - t0) * 1000

    encoder = TrtRunner(args.encoder_engine)
    lr = _ort(args.length_regulator)
    decoder = TrtRunner(args.decoder_engine)
    istft = _ort(args.istft)

    t0 = time.perf_counter()
    enc = encoder.run({k: feeds[k] for k in ("tokens", "style", "speed")})
    timings["encoder_trt_ms"] = (time.perf_counter() - t0) * 1000
    _print_stats("encoder", enc)

    stage = dict(feeds)
    stage.update(enc)
    t0 = time.perf_counter()
    lr_out = _run_ort(lr, stage)
    timings["length_cpu_ms"] = (time.perf_counter() - t0) * 1000
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
    timings["decoder_trt_ms"] = (time.perf_counter() - t0) * 1000
    _print_stats("decoder", dec)

    t0 = time.perf_counter()
    audio = _run_ort(istft, dec)["audio"]
    timings["istft_cpu_ms"] = (time.perf_counter() - t0) * 1000
    _print_stats("istft", {"audio": audio})
    timings["hybrid_ms"] = sum(v for k, v in timings.items() if k != "full_cpu_ms")

    diff = np.asarray(full_audio, dtype=np.float32) - np.asarray(audio, dtype=np.float32)
    for key, value in timings.items():
        print(f"{key}={value:.3f}")
    print(f"audio_shape={audio.shape}")
    print(f"max_abs_diff={float(np.max(np.abs(diff)))}")
    print(f"mean_abs_diff={float(np.mean(np.abs(diff)))}")
    print(f"allclose_1e3={bool(np.allclose(full_audio, audio, rtol=1e-3, atol=1e-3))}")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
