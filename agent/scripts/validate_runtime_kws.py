#!/usr/bin/env python3
"""Hardware-free validation of runtime phrase compilation and KWS inference."""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np

# Keep this asset-level verifier independent from the full Agent dependency
# graph (LLM/audio/UI). Production imports the normal package; this script only
# needs the isolated kws subpackage.
if "ovs_agent" not in sys.modules:
    package = types.ModuleType("ovs_agent")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "ovs_agent")]
    sys.modules["ovs_agent"] = package

from ovs_agent.kws import PhraseCompiler, SherpaKwsBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--phrase", required=True, action="append")
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--score", type=float, default=1.0)
    args = parser.parse_args()
    model = args.model_dir
    compiler = PhraseCompiler(
        tokens=str(model / "tokens.txt"),
        lexicon=str(model / "en.phone"),
    )
    started = time.perf_counter()
    compiled = compiler.compile(args.phrase)
    compiled_ms = (time.perf_counter() - started) * 1000
    backend = SherpaKwsBackend(
        {
            "tokens": str(model / "tokens.txt"),
            "encoder": str(model / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            "decoder": str(model / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
            "joiner": str(model / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
            "num_threads": 1,
            "keywords_score": args.score,
            "keywords_threshold": args.threshold,
            "num_trailing_blanks": 1,
        }
    )
    started = time.perf_counter()
    stream = backend.create_stream(compiled)
    load_ms = (time.perf_counter() - started) * 1000
    detected = None
    inference_started = time.perf_counter()
    with wave.open(str(args.wav), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("validation WAV must be mono 16-bit PCM")
        sample_rate = wav.getframerate()
        while chunk := wav.readframes(1600):
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            detected = backend.detect(stream, samples, sample_rate) or detected
        # A live microphone never ends at the last phoneme. Feed trailing
        # silence so the configured blank-count can finalize phrase-only WAVs.
        audio_s = wav.getnframes() / sample_rate + 0.8
        for _ in range(8):
            detected = backend.detect(
                stream, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate
            ) or detected
    inference_ms = (time.perf_counter() - inference_started) * 1000
    print(
        json.dumps(
            {
                "ok": bool(detected),
                "phrases": list(compiled.phrases),
                "detected": detected,
                "wav": str(args.wav),
                "compile_ms": round(compiled_ms, 1),
                "model_load_ms": round(load_ms, 1),
                "inference_ms": round(inference_ms, 1),
                "audio_s": round(audio_s, 3),
                "rtf": round((inference_ms / 1000) / audio_s, 4),
            },
            ensure_ascii=False,
        )
    )
    return 0 if detected else 1


if __name__ == "__main__":
    sys.exit(main())
