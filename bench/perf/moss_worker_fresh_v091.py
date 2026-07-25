#!/usr/bin/env python3
"""Validate a fresh MOSS TensorRT engine set through the JSONL worker."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import select
import subprocess
import time
import wave
from array import array
from pathlib import Path


DEFAULT_TEXTS = (
    "今天天气很好，我们一起测试语音合成。",
    "语音合成的稳定性。",
    "说起咱北京的烤鸭啊，那可真是外焦里嫩。",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--codec-onnx-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


class EventReader:
    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stdout is None:
            raise RuntimeError("worker stdout is unavailable")
        self.proc = proc
        self.stdout = proc.stdout
        self.buffer = bytearray()

    def read(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while b"\n" in self.buffer:
                line, _, remainder = self.buffer.partition(b"\n")
                self.buffer = bytearray(remainder)
                try:
                    return json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # TensorRT/EdgeLLM diagnostics are not worker protocol.
                    continue
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([self.stdout], [], [], remaining)
            if readable:
                chunk = os.read(self.stdout.fileno(), 65536)
                if chunk:
                    self.buffer.extend(chunk)
                    continue
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker exited with status {self.proc.returncode}")
        raise TimeoutError("timed out waiting for MOSS worker event")


def metrics(pcm: bytes, sample_rate: int, channels: int) -> dict:
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        raise RuntimeError("worker returned no PCM samples")
    scale = 32768.0
    rms = math.sqrt(sum((sample / scale) ** 2 for sample in samples) / len(samples))
    clipped = sum(abs(sample) >= 32767 for sample in samples) / len(samples)
    return {
        "bytes": len(pcm),
        "duration_s": len(samples) / channels / sample_rate,
        "rms": rms,
        "clipping_fraction": clipped,
    }


def write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.worker,
        f"--engine-dir={args.engine_dir}",
        f"--tokenizer-model={args.engine_dir}/tokenizer.model",
        f"--codec-onnx-dir={args.codec_onnx_dir}",
        "--max-slots=2",
    ]
    stderr_path = output_dir / "worker.stderr.log"
    with stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        try:
            events = EventReader(proc)
            ready = events.read(args.timeout)
            if ready.get("event") != "worker_ready" or not ready.get("ok"):
                raise RuntimeError(f"unexpected worker readiness event: {ready}")
            sample_rate = int(ready["sample_rate"])
            channels = int(ready["channels"])
            results = []
            for index, text in enumerate(args.texts or DEFAULT_TEXTS, start=1):
                request_id = f"moss-v091-{index}"
                request = {
                    "id": request_id,
                    "text": text,
                    "stream": True,
                    "stream_only": True,
                    "chunk_transport": "base64",
                    "chunk_format": "pcm_s16le",
                }
                assert proc.stdin is not None
                proc.stdin.write(
                    (json.dumps(request, ensure_ascii=False) + "\n").encode()
                )
                proc.stdin.flush()
                pcm_parts: list[bytes] = []
                done = None
                while done is None:
                    event = events.read(args.timeout)
                    if event.get("id") != request_id:
                        continue
                    if event.get("event") == "error":
                        raise RuntimeError(f"MOSS request failed: {event}")
                    if event.get("event") == "chunk":
                        pcm_parts.append(base64.b64decode(event["audio_b64"]))
                    elif event.get("event") == "done":
                        done = event
                pcm = b"".join(pcm_parts)
                wav_path = output_dir / f"moss_v091_{index}.wav"
                write_wav(wav_path, pcm, sample_rate, channels)
                result = {
                    "id": request_id,
                    "text": text,
                    "wav": str(wav_path),
                    **metrics(pcm, sample_rate, channels),
                    "ttfa_ms": done.get("ttfa_ms"),
                    "wall_ms": done.get("wall_ms"),
                    "chunks": len(pcm_parts),
                }
                if result["rms"] <= 0.01:
                    raise RuntimeError(f"silent/near-silent MOSS output: {result}")
                if result["clipping_fraction"] >= 0.01:
                    raise RuntimeError(f"clipped MOSS output: {result}")
                results.append(result)
            report = {
                "worker_ready": ready,
                "command": command,
                "results": results,
            }
            (output_dir / "moss-v091-smoke.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            if proc.stdin:
                proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
