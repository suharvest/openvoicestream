#!/usr/bin/env python3
"""Run reproducible one-shot quality cases through a Qwen3 ASR worker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import uuid
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128


def mel_filterbank() -> np.ndarray:
    hz_to_mel = lambda value: 2595.0 * np.log10(1.0 + value / 700.0)
    mel_to_hz = lambda value: 700.0 * (10.0 ** (value / 2595.0) - 1.0)
    points = np.linspace(hz_to_mel(0.0), hz_to_mel(8000.0), N_MELS + 2)
    hz_points = mel_to_hz(points)
    bins = np.floor((N_FFT // 2) * hz_points / 8000.0).astype(np.int32)
    bins = np.clip(bins, 0, N_FFT // 2)
    bank = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float64)
    for row in range(1, N_MELS + 1):
        left, center, right = bins[row - 1], bins[row], bins[row + 1]
        if left != center:
            bank[row - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if center != right:
            bank[row - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)
    bank *= (2.0 / (hz_points[2:] - hz_points[:-2]))[:, None]
    return bank.astype(np.float32)


MEL_FILTERBANK = mel_filterbank()


def wav_to_mel(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {sample_width * 8}-bit")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        output_size = int(round(audio.size * SAMPLE_RATE / sample_rate))
        audio = np.interp(
            np.linspace(0.0, 1.0, output_size, endpoint=False),
            np.linspace(0.0, 1.0, audio.size, endpoint=False),
            audio,
        ).astype(np.float32)
    audio = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    frames_view = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[
        ::HOP_LENGTH
    ]
    spectrum = np.fft.rfft(frames_view * window, n=N_FFT, axis=1)
    power = np.abs(spectrum[:-1].T).astype(np.float32) ** 2
    mel = MEL_FILTERBANK @ power
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    if log_mel.shape[1] < 100:
        log_mel = np.pad(log_mel, ((0, 0), (0, 100 - log_mel.shape[1])))
    return log_mel[None].astype(np.float16)


def write_safetensors(tensor: np.ndarray, path: Path) -> None:
    header = {
        "input_features": {
            "dtype": "F16",
            "shape": list(tensor.shape),
            "data_offsets": [0, tensor.nbytes],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("wb") as stream:
        stream.write(len(header_bytes).to_bytes(8, "little"))
        stream.write(header_bytes)
        stream.write(tensor.tobytes())


def normalize(text: str) -> str:
    for language in (
        "Chinese",
        "English",
        "Cantonese",
        "Japanese",
        "Korean",
        "French",
        "German",
        "Italian",
        "Portuguese",
        "Russian",
        "Spanish",
    ):
        prefix = f"language {language}"
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    ignored = set("。，、！？；：,.!?;: \t\r\n")
    return "".join(char for char in text if char not in ignored)


def lcs_similarity(left: str, right: str) -> float:
    left, right = normalize(left), normalize(right)
    if not left or not right:
        return float(left == right)
    row = [0] * (len(right) + 1)
    for left_char in left:
        previous = 0
        for index, right_char in enumerate(right, 1):
            saved = row[index]
            row[index] = (
                previous + 1
                if left_char == right_char
                else max(row[index], row[index - 1])
            )
            previous = saved
    return row[-1] / max(len(left), len(right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--audio-engine-dir", type=Path, required=True)
    parser.add_argument("--wav", action="append", type=Path, required=True)
    parser.add_argument("--expected", action="append", required=True)
    parser.add_argument("--max-slots", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=128)
    parser.add_argument("--min-similarity", type=float, default=0.9)
    parser.add_argument(
        "--input-format",
        choices=("wav", "mel"),
        default="wav",
        help="v0.9.x uses direct WAV/PCM; mel is retained for legacy diagnostics.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.wav) != len(args.expected):
        parser.error("--wav and --expected counts must match")

    environment = os.environ.copy()
    environment["EDGELLM_PLUGIN_PATH"] = str(args.plugin)
    environment["EDGE_LLM_ASR_CUDA_GRAPH"] = "0"
    command = [
        str(args.worker),
        f"--engineDir={args.engine_dir}",
        f"--multimodalEngineDir={args.audio_engine_dir}",
        f"--max_slots={args.max_slots}",
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdin and process.stdout and process.stderr
    ready_line = process.stdout.readline()
    if not ready_line:
        raise RuntimeError(process.stderr.read()[-4000:])
    ready = json.loads(ready_line)
    worker_init_wall_ms = (time.perf_counter() - started) * 1000
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="qwen-asr-v091-") as temp_dir:
        for index, (wav_path, expected) in enumerate(
            zip(args.wav, args.expected, strict=True), 1
        ):
            audio_path = wav_path
            if args.input_format == "mel":
                audio_path = Path(temp_dir) / f"{index}.safetensors"
                write_safetensors(wav_to_mel(wav_path), audio_path)
            request = {
                "id": uuid.uuid4().hex,
                "requests": [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "audio", "audio": str(audio_path)}],
                            }
                        ]
                    }
                ],
                "batch_size": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 1,
                "max_generate_length": args.max_generate_length,
                "apply_chat_template": True,
                "add_generation_prompt": True,
            }
            request_started = time.perf_counter()
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            elapsed_ms = (time.perf_counter() - request_started) * 1000
            text = response.get("responses", [{}])[0].get("output_text", "")
            similarity = lcs_similarity(text, expected)
            results.append(
                {
                    "wav": str(wav_path),
                    "expected": expected,
                    "text": text,
                    "similarity": similarity,
                    "elapsed_ms": elapsed_ms,
                    "ok": bool(response.get("ok"))
                    and similarity >= args.min_similarity,
                }
            )

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    stderr = process.stderr.read()
    passed = (
        ready.get("event") == "ready"
        and ready.get("max_slots") == args.max_slots
        and all(bool(result["ok"]) for result in results)
    )
    report = {
        "worker_ready": ready,
        "worker_init_wall_ms": worker_init_wall_ms,
        "command": command,
        "input_format": args.input_format,
        "results": results,
        "stderr_tail": stderr[-8000:],
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
