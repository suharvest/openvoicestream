#!/usr/bin/env python3
"""Strict two-lane HTTP TTS overlap, PCM, and output-isolation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def stream(
    base_url: str,
    text: str,
    barrier: threading.Barrier,
    speaker_id: int | None = None,
    language: str = "chinese",
    expected_sample_rate: int = 24000,
) -> dict:
    barrier.wait()
    started = time.perf_counter()
    first_pcm_at = None
    chunks = []
    with requests.post(
        base_url.rstrip("/") + "/tts/stream",
        json={
            "text": text,
            "language": language,
            **({"speaker_id": speaker_id} if speaker_id is not None else {}),
        },
        stream=True,
        timeout=90,
    ) as response:
        status = response.status_code
        content_type = response.headers.get("content-type")
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunks.append(chunk)
            if sum(map(len, chunks)) > 4 and first_pcm_at is None:
                first_pcm_at = time.perf_counter()
    ended = time.perf_counter()
    payload = b"".join(chunks)
    sample_rate = (
        struct.unpack("<I", payload[:4])[0]
        if status == 200 and len(payload) >= 4
        else None
    )
    pcm = payload[4:] if len(payload) >= 4 else b""
    audio_seconds = len(pcm) / (expected_sample_rate * 2)
    wall_ms = (ended - started) * 1000
    return {
        "started_monotonic": started,
        "ended_monotonic": ended,
        "status": status,
        "content_type": content_type,
        "sample_rate": sample_rate,
        "pcm_bytes": len(pcm),
        "sha256": hashlib.sha256(pcm).hexdigest(),
        "ttfa_ms": (
            (first_pcm_at - started) * 1000
            if first_pcm_at is not None
            else None
        ),
        "wall_ms": wall_ms,
        "audio_seconds": audio_seconds,
        "rtf": wall_ms / 1000 / audio_seconds if audio_seconds > 0 else None,
        "passed": bool(
            status == 200
            and sample_rate == expected_sample_rate
            and len(pcm) > 0
            and first_pcm_at is not None
        ),
    }


def clean(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if not key.endswith("_monotonic")
    }


def summary(values: list[float | None]) -> dict[str, float | None]:
    values = [value for value in values if value is not None]
    if not values:
        return {"min": None, "p50": None, "max": None}
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18622")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--speaker-a", type=int)
    parser.add_argument("--speaker-b", type=int)
    parser.add_argument("--language-a", default="chinese")
    parser.add_argument("--language-b", default="chinese")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rounds = []
    for index in range(1, args.rounds + 1):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(
                stream,
                args.base_url,
                "第一路并发语音输出必须完整且互不串音。",
                barrier,
                args.speaker_a,
                args.language_a,
                args.sample_rate,
            )
            future_b = pool.submit(
                stream,
                args.base_url,
                "第二路使用不同文本验证输出隔离和真实重叠。",
                barrier,
                args.speaker_b,
                args.language_b,
                args.sample_rate,
            )
            raw_a = future_a.result()
            raw_b = future_b.result()
        overlap_ms = max(
            0.0,
            (
                min(raw_a["ended_monotonic"], raw_b["ended_monotonic"])
                - max(raw_a["started_monotonic"], raw_b["started_monotonic"])
            )
            * 1000,
        )
        passed = bool(
            raw_a["passed"]
            and raw_b["passed"]
            and raw_a["sha256"] != raw_b["sha256"]
            and overlap_ms > 0
        )
        rounds.append(
            {
                "round": index,
                "a": clean(raw_a),
                "b": clean(raw_b),
                "overlap_ms": overlap_ms,
                "passed": passed,
            }
        )
        print(
            f"TTS_N2 {index}/{args.rounds}: pass={passed} "
            f"overlap={overlap_ms:.1f}ms "
            f"Attfa={raw_a['ttfa_ms']:.1f}ms Bttfa={raw_b['ttfa_ms']:.1f}ms",
            flush=True,
        )
        if not passed:
            break
    lanes = [row[lane] for row in rounds for lane in ("a", "b")]
    report = {
        "test": "v091_r2_tts_isolated_n2",
        "rounds_requested": args.rounds,
        "rounds_passed": sum(row["passed"] for row in rounds),
        "ttfa_ms": summary([lane["ttfa_ms"] for lane in lanes]),
        "rtf": summary([lane["rtf"] for lane in lanes]),
        "rounds": rounds,
    }
    report["passed"] = (
        len(rounds) == args.rounds and report["rounds_passed"] == args.rounds
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
