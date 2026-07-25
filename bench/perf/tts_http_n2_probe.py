#!/usr/bin/env python3
"""Small full-stream TTFA probe for Qwen3-TTS HTTP N=1/N=2.

Unlike the historical first-byte-only probe, every response is drained to
completion. This prevents abandoned synthesis from occupying a worker slot and
distorting the next measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


PROMPTS = (
    "我们都非常震惊。这位母亲表示。",
    "今天天气真不错，适合出门散步。",
)


def run_one(url: str, text: str, timeout: float, start_delay_ms: float = 0) -> dict:
    if start_delay_ms > 0:
        time.sleep(start_delay_ms / 1000)
    started = time.perf_counter()
    response = requests.post(
        url.rstrip("/") + "/tts/stream",
        json={"text": text},
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    # Read exactly the service's 4-byte sample-rate prefix, then one PCM byte.
    # requests.iter_content(chunk_size=4096) can combine transport chunks and
    # is inappropriate for a precise first-byte timestamp.
    sample_rate_header = response.raw.read(4)
    first_pcm = response.raw.read(1)
    first_pcm_at = time.perf_counter() if first_pcm else None

    pcm = bytearray(first_pcm)
    while True:
        chunk = response.raw.read(64 * 1024)
        if not chunk:
            break
        pcm.extend(chunk)
    response.close()
    ended = time.perf_counter()

    return {
        "status": response.status_code,
        "sample_rate": int.from_bytes(sample_rate_header, "little")
        if len(sample_rate_header) == 4
        else None,
        "ttfa_ms": (first_pcm_at - started) * 1000 if first_pcm_at else None,
        "total_ms": (ended - started) * 1000,
        "pcm_bytes": len(pcm),
        "pcm_md5": hashlib.md5(pcm).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8622")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--stagger-ms",
        type=float,
        default=0,
        help="Delay lane 2 submission; useful for measuring prefill contention.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rounds = []
    all_ttfas = []
    for round_idx in range(args.rounds):
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    run_one,
                    args.base_url,
                    PROMPTS[lane],
                    args.timeout,
                    args.stagger_ms if lane == 1 else 0,
                )
                for lane in range(args.concurrency)
            ]
            lanes = [future.result() for future in futures]
        rounds.append({"round": round_idx, "lanes": lanes})
        all_ttfas.extend(
            lane["ttfa_ms"] for lane in lanes if lane["ttfa_ms"] is not None
        )
        print(
            f"round={round_idx + 1}/{args.rounds} "
            f"ttfa_ms={[round(lane['ttfa_ms'], 1) for lane in lanes]} "
            f"total_ms={[round(lane['total_ms'], 1) for lane in lanes]}",
            flush=True,
        )

    report = {
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "stagger_ms": args.stagger_ms,
        "rounds": rounds,
        "ttfa_p50_ms": statistics.median(all_ttfas),
        "all_full_streams": all(
            lane["status"] == 200 and lane["pcm_bytes"] > 0
            for round_data in rounds
            for lane in round_data["lanes"]
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote={args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
