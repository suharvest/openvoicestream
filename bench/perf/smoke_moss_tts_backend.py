#!/usr/bin/env python3
"""Standalone smoke for MossTtsNanoBackend.

Exercises the Python backend → C++ worker subprocess → WAV output path
without spinning up the FastAPI HTTP layer. Run this BEFORE wiring into
docker compose to confirm the Python ↔ worker handshake works on the
target device.

Usage on orin-nx (after P4 layout is in place):

    python3 bench/perf/smoke_moss_tts_backend.py \\
        --text "你好，今天天气真不错" \\
        --output /tmp/moss_backend_smoke.wav
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="你好")
    p.add_argument("--output", default="/tmp/moss_backend_smoke.wav")
    p.add_argument("--mode", choices=["sync", "streaming"], default="streaming")
    p.add_argument("--ref-audio", default=None,
                   help="Optional path to int16 stereo 48kHz reference WAV for voice clone")
    p.add_argument("--engine-dir", default=None)
    p.add_argument("--tokenizer-model", default=None)
    p.add_argument("--codec-onnx-dir", default=None)
    p.add_argument("--worker-bin", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def configure_env(args: argparse.Namespace) -> None:
    if args.worker_bin: os.environ["MOSS_WORKER_BIN"] = args.worker_bin
    if args.engine_dir: os.environ["MOSS_ENGINE_DIR"] = args.engine_dir
    if args.tokenizer_model: os.environ["MOSS_TOKENIZER"] = args.tokenizer_model
    if args.codec_onnx_dir: os.environ["MOSS_CODEC_ONNX_DIR"] = args.codec_onnx_dir


def write_wav(path: str, pcm: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    configure_env(args)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from voxedge.backends.jetson.moss_tts_nano import MossTtsNanoBackend

    backend = MossTtsNanoBackend({})
    print(f"[smoke] backend.name={backend.name} caps={sorted(c.name for c in backend.capabilities)}",
          flush=True)

    start_preload = time.perf_counter()
    backend.preload()
    preload_ms = (time.perf_counter() - start_preload) * 1000
    print(f"[smoke] preload OK in {preload_ms:.0f} ms; is_ready={backend.is_ready()}", flush=True)

    ref_audio_bytes = None
    if args.ref_audio:
        with wave.open(args.ref_audio, "rb") as wr:
            ref_audio_bytes = wr.readframes(wr.getnframes())
        print(f"[smoke] reference audio loaded: {len(ref_audio_bytes)} bytes", flush=True)

    start_synth = time.perf_counter()
    if args.mode == "streaming":
        kwargs = {"reference_audio_pcm_s16": ref_audio_bytes} if ref_audio_bytes else {}
        pcm_chunks: list[bytes] = []
        first_chunk_ms = None
        for chunk in backend.generate_streaming(args.text, **kwargs):
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - start_synth) * 1000
                print(f"[smoke] first chunk {len(chunk)} bytes at {first_chunk_ms:.0f} ms",
                      flush=True)
            pcm_chunks.append(chunk)
        pcm = b"".join(pcm_chunks)
        total_ms = (time.perf_counter() - start_synth) * 1000
        write_wav(args.output, pcm, backend.sample_rate, 2)
        print(f"[smoke] streaming done: {len(pcm)} bytes, {len(pcm)//4} stereo samples, "
              f"duration={len(pcm)/4/backend.sample_rate:.2f}s, total wall={total_ms:.0f}ms, "
              f"ttfa={first_chunk_ms:.0f}ms, wav={args.output}", flush=True)
    else:
        if ref_audio_bytes:
            result = backend.clone_voice(args.text, reference_audio=ref_audio_bytes)
        else:
            result = backend.synthesize(args.text)
        total_ms = (time.perf_counter() - start_synth) * 1000
        wav_bytes = result.wav_bytes if hasattr(result, "wav_bytes") else bytes(result)
        Path(args.output).write_bytes(wav_bytes)
        print(f"[smoke] sync done: {len(wav_bytes)} bytes wav, wall={total_ms:.0f}ms, "
              f"wav={args.output}", flush=True)
        if hasattr(result, "metadata"):
            print(f"[smoke] metadata={result.metadata}", flush=True)

    backend.shutdown()
    print("[smoke] backend.shutdown OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
