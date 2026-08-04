#!/usr/bin/env python3
"""Concurrent Qwen3-ASR WebSocket streaming N=2 latency and isolation gate."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import websocket


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path}: expected PCM16 WAV, got width={sample_width}")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000:
        old = np.linspace(0, len(audio) - 1, len(audio))
        new = np.linspace(
            0,
            len(audio) - 1,
            int(len(audio) * 16000 / sample_rate),
        )
        audio = np.interp(new, old, audio).astype(np.float32)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def timing_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def drain_available(
    ws: websocket.WebSocket,
    messages: list[dict[str, Any]],
    first_send: float,
    eos_at: float | None,
    first_partial_at: float | None,
) -> tuple[float | None, dict[str, Any] | None]:
    final = None
    while True:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            break
        except websocket.WebSocketConnectionClosedException:
            break
        try:
            message = json.loads(raw)
        except Exception:
            message = {"raw": raw}
        messages.append(message)
        text = str(message.get("text", "")).strip()
        if (
            first_partial_at is None
            and text
            and message.get("type") == "partial"
            and not message.get("is_final")
        ):
            first_partial_at = time.perf_counter()
        if message.get("type") == "final" or message.get("is_final") is True:
            final = message
            break
    return first_partial_at, final


def stream_one(
    base_url: str,
    wav_path: Path,
    language: str,
    expected: str,
    chunk_ms: int,
    realtime: bool,
    barrier: threading.Barrier,
    timeout: float,
) -> dict[str, Any]:
    audio = load_audio(wav_path)
    chunk_samples = max(1, int(16000 * chunk_ms / 1000))
    url = (
        f"{base_url.rstrip('/')}/asr/stream"
        f"?language={quote(language)}&sample_rate=16000&vad=none"
    )
    ws = websocket.create_connection(url, timeout=timeout)
    ws.settimeout(0.001)
    messages: list[dict[str, Any]] = []
    barrier.wait()
    first_send = time.perf_counter()
    first_partial_at = None
    final = None
    for offset in range(0, len(audio), chunk_samples):
        chunk_started = time.perf_counter()
        ws.send_binary(audio[offset : offset + chunk_samples].tobytes())
        first_partial_at, received_final = drain_available(
            ws, messages, first_send, None, first_partial_at
        )
        final = final or received_final
        if realtime:
            time.sleep(
                max(0.0, chunk_ms / 1000 - (time.perf_counter() - chunk_started))
            )

    eos_at = time.perf_counter()
    ws.send_binary(b"")
    ws.settimeout(timeout)
    while final is None:
        first_partial_at, final = drain_available(
            ws, messages, first_send, eos_at, first_partial_at
        )
        if final is None:
            break
    final_at = time.perf_counter()
    try:
        ws.close()
    except Exception:
        pass

    text = str((final or {}).get("text", "")).strip()
    return {
        "wav": str(wav_path),
        "language": language,
        "expected": expected,
        "text": text,
        "matches": re.search(expected, text, re.IGNORECASE) is not None,
        "first_send_monotonic": first_send,
        "eos_monotonic": eos_at,
        "final_monotonic": final_at,
        "first_partial_ms": (
            (first_partial_at - first_send) * 1000
            if first_partial_at is not None
            else None
        ),
        "partial_before_eos": (
            first_partial_at is not None and first_partial_at < eos_at
        ),
        "eos_to_final_ms": (final_at - eos_at) * 1000,
        "total_stream_ms": (final_at - first_send) * 1000,
        "message_count": len(messages),
        "partial_count": sum(
            message.get("type") == "partial" for message in messages
        ),
        "final": final or {},
    }


def clean(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.endswith("_monotonic")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="ws://127.0.0.1:8622")
    parser.add_argument("--wav-a", type=Path, required=True)
    parser.add_argument("--wav-b", type=Path, required=True)
    parser.add_argument("--language-a", default="Chinese")
    parser.add_argument("--language-b", default="English")
    parser.add_argument("--expect-a", required=True)
    parser.add_argument("--expect-b", required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--chunk-ms", type=int, default=250)
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rounds = []
    for index in range(1, args.rounds + 1):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(
                stream_one,
                args.base_url,
                args.wav_a,
                args.language_a,
                args.expect_a,
                args.chunk_ms,
                args.realtime,
                barrier,
                args.timeout,
            )
            future_b = pool.submit(
                stream_one,
                args.base_url,
                args.wav_b,
                args.language_b,
                args.expect_b,
                args.chunk_ms,
                args.realtime,
                barrier,
                args.timeout,
            )
            result_a = future_a.result()
            result_b = future_b.result()
        overlap_ms = max(
            0.0,
            (
                min(
                    result_a["final_monotonic"],
                    result_b["final_monotonic"],
                )
                - max(
                    result_a["first_send_monotonic"],
                    result_b["first_send_monotonic"],
                )
            )
            * 1000,
        )
        ok = (
            result_a["matches"]
            and result_b["matches"]
            and bool(result_a["text"])
            and bool(result_b["text"])
            and overlap_ms > 0
        )
        rounds.append(
            {
                "round": index,
                "a": clean(result_a),
                "b": clean(result_b),
                "overlap_ms": overlap_ms,
                "ok": ok,
            }
        )
        print(
            f"[{index:03d}/{args.rounds}] ok={ok} overlap={overlap_ms:.1f}ms "
            f"A(partial/final)={result_a['first_partial_ms']}/"
            f"{result_a['eos_to_final_ms']:.1f}ms "
            f"B(partial/final)={result_b['first_partial_ms']}/"
            f"{result_b['eos_to_final_ms']:.1f}ms",
            flush=True,
        )

    lanes = [
        {"lane": lane, **row[lane]}
        for row in rounds
        for lane in ("a", "b")
    ]
    report = {
        "test": "qwen_asr_stream_n2_service",
        "config": {
            "base_url": args.base_url,
            "rounds": args.rounds,
            "chunk_ms": args.chunk_ms,
            "realtime": args.realtime,
        },
        "rounds_requested": args.rounds,
        "rounds_passed": sum(bool(row["ok"]) for row in rounds),
        "latency": {
            "first_partial_ms": timing_summary(lanes, "first_partial_ms"),
            "eos_to_final_ms": timing_summary(lanes, "eos_to_final_ms"),
            "total_stream_ms": timing_summary(lanes, "total_stream_ms"),
            "overlap_ms": timing_summary(rounds, "overlap_ms"),
        },
        "partial_before_eos_count": sum(
            bool(lane["partial_before_eos"]) for lane in lanes
        ),
        "lanes_total": len(lanes),
        "rounds": rounds,
    }
    report["passed"] = report["rounds_passed"] == args.rounds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "rounds_requested",
        "rounds_passed",
        "latency",
        "partial_before_eos_count",
        "lanes_total",
        "passed",
    )}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
