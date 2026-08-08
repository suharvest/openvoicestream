#!/usr/bin/env python3
"""Product N=1 gate: sequential E2E plus valid pairwise GDN overlap."""

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


def gdn_stream(base_url: str, timeout: float = 30) -> dict:
    payload = {
        "model": "engines",
        "messages": [{"role": "user", "content": "只回答：并发正常"}],
        "max_tokens": 12,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    first_token_at = None
    event_count = 0
    content = ""
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
            if token:
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


def run_asr(stream_one, websocket_url: str, wav: Path) -> dict:
    raw = stream_one(
        websocket_url,
        wav,
        "Chinese",
        "非常震惊",
        250,
        True,
        threading.Barrier(1),
        30,
    )
    row = {
        key: value
        for key, value in raw.items()
        if not key.endswith("_monotonic")
    }
    row["passed"] = bool(
        row["matches"]
        and row["text"]
        and row["partial_before_eos"]
        and row["partial_count"] > 0
    )
    return row


def run_tts(stream_once, voice_url: str) -> dict:
    row = stream_once(
        voice_url,
        "稳定一条语音链路并记录首包延迟。",
        90,
        24000,
    )
    audio_seconds = row["pcm_bytes"] / (24000 * 2)
    row["audio_seconds"] = audio_seconds
    row["rtf"] = (
        row["wall_ms"] / 1000 / audio_seconds if audio_seconds > 0 else None
    )
    row["passed"] = bool(row["passed"] and row["pcm_bytes"] > 0)
    return row


def timed(call) -> tuple[float, float, object]:
    started = time.perf_counter()
    value = call()
    ended = time.perf_counter()
    return started, ended, value


def timed_barrier(barrier: threading.Barrier, call) -> tuple[float, float, object]:
    barrier.wait()
    return timed(call)


def pairwise(
    name: str,
    left_call,
    right_call,
    rounds: int,
) -> list[dict]:
    rows = []
    for index in range(1, rounds + 1):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            left_future = pool.submit(timed_barrier, barrier, left_call)
            right_future = pool.submit(timed_barrier, barrier, right_call)
            left_start, left_end, left = left_future.result()
            right_start, right_end, right = right_future.result()
        overlap_ms = max(
            0.0,
            (min(left_end, right_end) - max(left_start, right_start)) * 1000,
        )
        skew_ms = abs(left_start - right_start) * 1000
        passed = bool(
            left["passed"]
            and right["passed"]
            and overlap_ms > 0
            and skew_ms <= 500
        )
        row = {
            "round": index,
            "left": left,
            "gdn": right,
            "overlap_ms": overlap_ms,
            "start_skew_ms": skew_ms,
            "passed": passed,
        }
        rows.append(row)
        print(
            f"{name} {index}/{rounds}: pass={passed} "
            f"overlap={overlap_ms:.1f}ms skew={skew_ms:.1f}ms "
            f"GDNttft={right['ttft_ms']:.1f}ms",
            flush=True,
        )
        if not passed:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-url", default="http://127.0.0.1:18621")
    parser.add_argument("--gdn-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--sequential-rounds", type=int, default=10)
    parser.add_argument("--pairwise-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    helper_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(helper_root))
    from qwen_asr_stream_n2_service import stream_one
    from tts_http_stream_gate import stream_once

    websocket_url = args.voice_url.replace("http://", "ws://").replace(
        "https://", "wss://"
    )
    asr_call = lambda: run_asr(stream_one, websocket_url, args.wav)
    tts_call = lambda: run_tts(stream_once, args.voice_url)
    gdn_call = lambda: gdn_stream(args.gdn_url)

    sequential = []
    for index in range(1, args.sequential_rounds + 1):
        round_started = time.perf_counter()
        _, _, asr = timed(asr_call)
        _, _, gdn = timed(gdn_call)
        _, _, tts = timed(tts_call)
        total_ms = (time.perf_counter() - round_started) * 1000
        passed = bool(asr["passed"] and gdn["passed"] and tts["passed"])
        row = {
            "round": index,
            "asr": asr,
            "gdn": gdn,
            "tts": tts,
            "total_ms": total_ms,
            "passed": passed,
        }
        sequential.append(row)
        print(
            f"E2E {index}/{args.sequential_rounds}: pass={passed} "
            f"ASRpartial={asr['first_partial_ms']:.1f}ms "
            f"GDNttft={gdn['ttft_ms']:.1f}ms "
            f"TTSttfa={tts['ttfa_ms']:.1f}ms total={total_ms:.1f}ms",
            flush=True,
        )
        if not passed:
            break

    sequential_ok = bool(
        len(sequential) == args.sequential_rounds
        and all(row["passed"] for row in sequential)
    )
    asr_gdn = (
        pairwise(
            "ASR_GDN",
            asr_call,
            gdn_call,
            args.pairwise_rounds,
        )
        if sequential_ok
        else []
    )
    asr_gdn_ok = bool(
        len(asr_gdn) == args.pairwise_rounds
        and all(row["passed"] for row in asr_gdn)
    )
    tts_gdn = (
        pairwise(
            "TTS_GDN",
            tts_call,
            gdn_call,
            args.pairwise_rounds,
        )
        if asr_gdn_ok
        else []
    )
    report = {
        "test": "v091_r2_product_n1_co_residency",
        "sequential": {
            "rounds_requested": args.sequential_rounds,
            "rounds": sequential,
            "total_ms": timing_summary(
                [row["total_ms"] for row in sequential]
            ),
            "asr_first_partial_ms": timing_summary(
                [row["asr"]["first_partial_ms"] for row in sequential]
            ),
            "gdn_ttft_ms": timing_summary(
                [row["gdn"]["ttft_ms"] for row in sequential]
            ),
            "tts_ttfa_ms": timing_summary(
                [row["tts"]["ttfa_ms"] for row in sequential]
            ),
            "tts_rtf": timing_summary(
                [row["tts"]["rtf"] for row in sequential]
            ),
        },
        "asr_gdn_overlap": {
            "rounds_requested": args.pairwise_rounds,
            "rounds": asr_gdn,
        },
        "tts_gdn_overlap": {
            "rounds_requested": args.pairwise_rounds,
            "rounds": tts_gdn,
        },
    }
    report["passed"] = bool(
        sequential_ok
        and asr_gdn_ok
        and len(tts_gdn) == args.pairwise_rounds
        and all(row["passed"] for row in tts_gdn)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"]}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
