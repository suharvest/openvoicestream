#!/usr/bin/env python3
"""Strict two-lane true-streaming ASR isolation gate."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="ws://127.0.0.1:18622")
    parser.add_argument("--wav-a", type=Path, required=True)
    parser.add_argument("--wav-b", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen_asr_stream_n2_service import clean, stream_one

    rounds = []
    for index in range(1, args.rounds + 1):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(
                stream_one,
                args.base_url,
                args.wav_a,
                "Chinese",
                "非常震惊",
                250,
                True,
                barrier,
                30,
            )
            future_b = pool.submit(
                stream_one,
                args.base_url,
                args.wav_b,
                "English",
                "white smoke",
                250,
                True,
                barrier,
                30,
            )
            raw_a = future_a.result()
            raw_b = future_b.result()
        overlap_ms = max(
            0.0,
            (
                min(raw_a["final_monotonic"], raw_b["final_monotonic"])
                - max(
                    raw_a["first_send_monotonic"],
                    raw_b["first_send_monotonic"],
                )
            )
            * 1000,
        )
        a = clean(raw_a)
        b = clean(raw_b)
        lane_a_ok = bool(
            a["matches"]
            and a["text"]
            and a["partial_before_eos"]
            and a["partial_count"] > 0
            and "white smoke" not in a["text"].lower()
        )
        lane_b_ok = bool(
            b["matches"]
            and b["text"]
            and b["partial_before_eos"]
            and b["partial_count"] > 0
            and "非常震惊" not in b["text"]
        )
        passed = lane_a_ok and lane_b_ok and overlap_ms > 0 and a["text"] != b["text"]
        rounds.append(
            {
                "round": index,
                "a": a,
                "b": b,
                "overlap_ms": overlap_ms,
                "passed": passed,
            }
        )
        print(
            f"ASR_N2 {index}/{args.rounds}: pass={passed} "
            f"overlap={overlap_ms:.1f}ms "
            f"Apartial={a['first_partial_ms']:.1f}ms "
            f"Bpartial={b['first_partial_ms']:.1f}ms",
            flush=True,
        )
        if not passed:
            break

    report = {
        "test": "v091_r2_asr_isolated_n2",
        "rounds_requested": args.rounds,
        "rounds_passed": sum(row["passed"] for row in rounds),
        "rounds": rounds,
    }
    report["passed"] = (
        len(rounds) == args.rounds and report["rounds_passed"] == args.rounds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
