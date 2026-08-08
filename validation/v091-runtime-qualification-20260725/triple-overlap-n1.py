#!/usr/bin/env python3
"""Prove true client-time overlap across ASR, TTS, and GDN N=1 requests."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "p95": ordered[p95_index],
        "max": max(values),
    }


def timed_after_barrier(
    barrier: threading.Barrier, call
) -> tuple[float, float, object]:
    barrier.wait()
    started = time.perf_counter()
    value = call()
    ended = time.perf_counter()
    return started, ended, value


def gdn_stream(base_url: str, timeout: float) -> dict:
    payload = {
        "model": "engines",
        "messages": [{"role": "user", "content": "只回答：并发正常"}],
        "max_tokens": 12,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    content = ""
    event_count = 0
    first_token_at = None
    done_seen = False
    with requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        status = response.status_code
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if line.strip() == b"data: [DONE]":
                done_seen = True
                break
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                continue
            event = json.loads(raw)
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            token = delta.get("content", "")
            if not token:
                continue
            first_token_at = first_token_at or time.perf_counter()
            event_count += 1
            content += token
    ended = time.perf_counter()
    return {
        "status": status,
        "content": content,
        "event_count": event_count,
        "done_seen": done_seen,
        "ttft_ms": (
            (first_token_at - started) * 1000
            if first_token_at is not None
            else None
        ),
        "elapsed_ms": (ended - started) * 1000,
        "passed": bool(
            status == 200
            and event_count > 0
            and done_seen
            and "并发正常" in content
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-url", default="http://127.0.0.1:18621")
    parser.add_argument("--gdn-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    helper_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(helper_root))
    from qwen_asr_stream_n2_service import clean, stream_one
    from tts_http_stream_gate import stream_once

    websocket_url = args.voice_url.replace("http://", "ws://").replace(
        "https://", "wss://"
    )
    rounds = []
    for index in range(1, args.rounds + 1):
        barrier = threading.Barrier(3)
        with ThreadPoolExecutor(max_workers=3) as pool:
            asr_future = pool.submit(
                timed_after_barrier,
                barrier,
                lambda: stream_one(
                    websocket_url,
                    args.wav,
                    "Chinese",
                    "非常震惊",
                    250,
                    True,
                    threading.Barrier(1),
                    30,
                ),
            )
            tts_future = pool.submit(
                timed_after_barrier,
                barrier,
                lambda: stream_once(
                    args.voice_url,
                    "三路模型正在稳定并发运行，请完整生成这段流式语音。",
                    90,
                    24000,
                ),
            )
            gdn_future = pool.submit(
                timed_after_barrier,
                barrier,
                lambda: gdn_stream(args.gdn_url, 30),
            )
            asr_start, asr_end, asr_raw = asr_future.result()
            tts_start, tts_end, tts = tts_future.result()
            gdn_start, gdn_end, gdn = gdn_future.result()

        asr = clean(asr_raw)
        asr["passed"] = bool(
            asr["matches"]
            and asr["text"]
            and asr["partial_before_eos"]
            and asr["partial_count"] > 0
        )
        tts["passed"] = bool(tts["passed"] and tts["pcm_bytes"] > 0)
        starts = [asr_start, tts_start, gdn_start]
        ends = [asr_end, tts_end, gdn_end]
        overlap_ms = max(0.0, (min(ends) - max(starts)) * 1000)
        start_skew_ms = (max(starts) - min(starts)) * 1000
        passed = bool(
            asr["passed"]
            and tts["passed"]
            and gdn["passed"]
            and overlap_ms > 0
            and start_skew_ms <= 500
        )
        row = {
            "round": index,
            "asr": asr,
            "tts": tts,
            "gdn": gdn,
            "triple_overlap_ms": overlap_ms,
            "request_start_skew_ms": start_skew_ms,
            "passed": passed,
        }
        rounds.append(row)
        print(
            f"ROUND {index}/{args.rounds}: pass={passed} "
            f"overlap={overlap_ms:.1f}ms skew={start_skew_ms:.1f}ms "
            f"ASRpartial={asr['first_partial_ms']:.1f}ms "
            f"TTSttfa={tts['ttfa_ms']:.1f}ms GDNttft={gdn['ttft_ms']:.1f}ms",
            flush=True,
        )

    report = {
        "test": "v091_r2_base_asr_tts_gdn_true_triple_overlap_n1",
        "rounds_requested": args.rounds,
        "rounds_passed": sum(row["passed"] for row in rounds),
        "latency": {
            "triple_overlap_ms": timing_summary(
                [row["triple_overlap_ms"] for row in rounds]
            ),
            "request_start_skew_ms": timing_summary(
                [row["request_start_skew_ms"] for row in rounds]
            ),
            "asr_first_partial_ms": timing_summary(
                [row["asr"]["first_partial_ms"] for row in rounds]
            ),
            "tts_ttfa_ms": timing_summary(
                [row["tts"]["ttfa_ms"] for row in rounds]
            ),
            "gdn_ttft_ms": timing_summary(
                [row["gdn"]["ttft_ms"] for row in rounds]
            ),
        },
        "rounds": rounds,
    }
    report["passed"] = report["rounds_passed"] == args.rounds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
