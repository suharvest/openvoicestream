#!/usr/bin/env python3
"""Prove useful overlap across ASR, TTS, and a streaming GDN request."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import requests


def timing_summary(values: list[float | None]) -> dict[str, float | None]:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {"min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(numbers)
    p95_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))
    return {
        "min": min(numbers),
        "p50": statistics.median(numbers),
        "p95": ordered[p95_index],
        "max": max(numbers),
    }


def timed_after_barrier(
    barrier: threading.Barrier,
    call: Callable[[], Any],
) -> tuple[float, float, Any]:
    barrier.wait()
    started = time.perf_counter()
    try:
        value = call()
    except BaseException as exc:
        value = {"__error__": f"{type(exc).__name__}: {exc}"}
    return started, time.perf_counter(), value


def gdn_stream(base_url: str, timeout: float) -> dict[str, Any]:
    payload = {
        "model": "engines",
        "messages": [{
            "role": "user",
            "content": (
                "请连续输出一段不少于一百个汉字的设备并发稳定性说明，"
                "不要使用列表，不要提前结束。"
            ),
        }],
        "max_tokens": 128,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    content = ""
    token_times_ms: list[float] = []
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
                token_times_ms.append((time.perf_counter() - started) * 1000)
                content += token
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "status": status,
        "content": content,
        "event_count": len(token_times_ms),
        "token_times_ms": token_times_ms,
        "done_seen": done_seen,
        "ttft_ms": token_times_ms[0] if token_times_ms else None,
        "elapsed_ms": elapsed_ms,
        "passed": status == 200 and done_seen and len(token_times_ms) >= 32,
    }


def _load_helpers(repo_root: Path):
    sys.path.insert(0, str(repo_root / "bench" / "perf"))
    from qwen_asr_stream_n2_service import clean, stream_one

    gate_path = Path(__file__).resolve().with_name("tts-http-stream-gate.py")
    spec = importlib.util.spec_from_file_location("v091_tts_http_stream_gate", gate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TTS helper: {gate_path}")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return clean, stream_one, gate.stream_once


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-url", default="http://127.0.0.1:8621")
    parser.add_argument("--gdn-url", default="http://127.0.0.1:8000")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tts-sample-rate", type=int, default=24000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    clean, stream_one, stream_once = _load_helpers(repo_root)
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
                    args.tts_sample_rate,
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

        if isinstance(asr_raw, dict) and asr_raw.get("__error__"):
            asr = {
                "error": asr_raw["__error__"],
                "matches": False,
                "text": "",
                "partial_before_eos": False,
                "partial_count": 0,
                "first_partial_ms": None,
                "passed": False,
            }
        else:
            asr = clean(asr_raw)
            asr["passed"] = bool(
                asr["matches"]
                and asr["text"]
                and asr["partial_before_eos"]
                and asr["partial_count"] > 0
            )
        if isinstance(tts, dict) and tts.get("__error__"):
            tts = {
                "error": tts["__error__"],
                "status": 0,
                "pcm_bytes": 0,
                "ttfa_ms": None,
                "passed": False,
            }
        else:
            tts["passed"] = bool(tts["passed"] and tts["pcm_bytes"] > 0)
        if isinstance(gdn, dict) and gdn.get("__error__"):
            gdn = {
                "error": gdn["__error__"],
                "event_count": 0,
                "token_times_ms": [],
                "ttft_ms": None,
                "passed": False,
            }

        starts = [asr_start, tts_start, gdn_start]
        ends = [asr_end, tts_end, gdn_end]
        overlap_ms = max(0.0, (min(ends) - max(starts)) * 1000)
        start_skew_ms = (max(starts) - min(starts)) * 1000
        asr_partial_ms = asr.get("first_partial_ms")
        tts_audio_ms = tts.get("ttfa_ms")
        has_voice_milestones = isinstance(asr_partial_ms, (int, float)) and isinstance(
            tts_audio_ms, (int, float)
        )
        asr_partial_at = (
            asr_start + asr_partial_ms / 1000 if has_voice_milestones else None
        )
        tts_audio_at = (
            tts_start + tts_audio_ms / 1000 if has_voice_milestones else None
        )
        voice_first_output_at = (
            max(asr_partial_at, tts_audio_at) if has_voice_milestones else None
        )
        gdn_token_times = [
            gdn_start + value / 1000 for value in gdn["token_times_ms"]
        ]
        gdn_spans_voice_outputs = bool(
            has_voice_milestones
            and any(value < voice_first_output_at for value in gdn_token_times)
            and any(value > voice_first_output_at for value in gdn_token_times)
        )
        asr_tts_useful_overlap = bool(
            has_voice_milestones
            and tts_end > asr_partial_at
            and asr_end > tts_audio_at
        )
        passed = bool(
            asr["passed"]
            and tts["passed"]
            and gdn["passed"]
            and overlap_ms > 0
            and start_skew_ms <= 500
            and gdn_spans_voice_outputs
            and asr_tts_useful_overlap
        )
        rounds.append({
            "round": index,
            "asr": asr,
            "tts": tts,
            "gdn": gdn,
            "triple_overlap_ms": overlap_ms,
            "request_start_skew_ms": start_skew_ms,
            "gdn_spans_voice_first_outputs": gdn_spans_voice_outputs,
            "asr_tts_useful_overlap": asr_tts_useful_overlap,
            "passed": passed,
        })
        print(
            f"ROUND {index}/{args.rounds}: pass={passed} "
            f"overlap={overlap_ms:.1f}ms skew={start_skew_ms:.1f}ms "
            f"ASRpartial={asr_partial_ms!r}ms "
            f"TTSttfa={tts_audio_ms!r}ms GDNttft={gdn['ttft_ms']!r}ms",
            flush=True,
        )

    report = {
        "test": "v091_strict_asr_tts_gdn_triple_overlap_n1",
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
