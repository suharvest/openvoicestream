#!/usr/bin/env python3
"""Run fresh SenseVoice TRT inference and check finite logits/transcripts."""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wav", action="append", required=True)
    parser.add_argument("--expected", action="append", default=[])
    parser.add_argument(
        "--expected-base64",
        action="append",
        default=[],
        help="Append a Base64-encoded UTF-8 expected transcript.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-similarity", type=float, default=0.5)
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise RuntimeError(f"expected PCM16 WAV, got sample width {width}: {path}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        target = int(round(len(samples) * 16000 / rate))
        samples = np.interp(
            np.linspace(0.0, 1.0, target, endpoint=False),
            np.linspace(0.0, 1.0, len(samples), endpoint=False),
            samples,
        ).astype(np.float32)
    return samples


def main() -> int:
    args = parse_args()
    args.expected.extend(
        base64.b64decode(value, validate=True).decode("utf-8")
        for value in args.expected_base64
    )
    if len(args.wav) != len(args.expected):
        raise RuntimeError("--wav and --expected counts must match")

    from voxedge.backends.jetson.sensevoice_trt import (
        SenseVoiceTRTBackend,
        SenseVoiceTRTConfig,
    )

    backend = SenseVoiceTRTBackend(
        SenseVoiceTRTConfig(engine=args.engine, model_dir=args.model_dir)
    )
    preload_start = time.perf_counter()
    backend.preload()
    preload_ms = (time.perf_counter() - preload_start) * 1000
    results = []
    for wav_path, expected in zip(args.wav, args.expected):
        samples = read_wav(wav_path)
        speech, valid = backend._build_speech(samples, lang="zh")
        start = time.perf_counter()
        logits = backend._infer(speech)
        infer_ms = (time.perf_counter() - start) * 1000
        if logits is None or not np.isfinite(logits).all():
            raise RuntimeError(f"non-finite SenseVoice logits: {wav_path}")
        text = backend._ctc_decode(logits, valid)
        similarity = SequenceMatcher(
            None, normalize(expected), normalize(text)
        ).ratio()
        result = {
            "wav": wav_path,
            "expected": expected,
            "text": text,
            "similarity": similarity,
            "valid_frames": valid,
            "infer_ms": infer_ms,
            "logits_min": float(logits.min()),
            "logits_max": float(logits.max()),
        }
        if similarity < args.min_similarity:
            raise RuntimeError(f"SenseVoice transcript gate failed: {result}")
        results.append(result)
    backend.unload()
    report = {
        "engine": args.engine,
        "model_dir": args.model_dir,
        "preload_ms": preload_ms,
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
