#!/usr/bin/env python3
"""Cancel one TTS lane while a second lane completes; recover in the freed slot."""

from __future__ import annotations

import argparse
import json
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def full_stream(
    base_url: str,
    text: str,
    timeout: float,
    speaker_id: int | None = None,
    expected_sample_rate: int = 24000,
) -> dict:
    started = time.perf_counter()
    first_pcm_at = None
    chunks = []
    retry_after = None
    with requests.post(
        base_url.rstrip("/") + "/tts/stream",
        json={
            "text": text,
            "language": "chinese",
            **({"speaker_id": speaker_id} if speaker_id is not None else {}),
        },
        stream=True,
        timeout=timeout,
    ) as response:
        status = response.status_code
        retry_after = response.headers.get("retry-after")
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
    return {
        "started_monotonic": started,
        "ended_monotonic": ended,
        "first_pcm_monotonic": first_pcm_at,
        "status": status,
        "retry_after": retry_after,
        "sample_rate": sample_rate,
        "pcm_bytes": max(0, len(payload) - 4),
        "ttfa_ms": (
            (first_pcm_at - started) * 1000
            if first_pcm_at is not None
            else None
        ),
        "wall_ms": (ended - started) * 1000,
        "passed": bool(
            status == 200
            and sample_rate == expected_sample_rate
            and len(payload) > 4
            and first_pcm_at is not None
        ),
    }


def cancel_stream(
    base_url: str,
    speaker_id: int | None = None,
    text: str | None = None,
    expected_sample_rate: int = 24000,
) -> dict:
    started = time.perf_counter()
    response = requests.post(
        base_url.rstrip("/") + "/tts/stream",
        json={
            "text": text or "取消这一条很长的语音请求并释放工作槽位。" * 12,
            "language": "chinese",
            **({"speaker_id": speaker_id} if speaker_id is not None else {}),
        },
        stream=True,
        timeout=90,
    )
    status = response.status_code
    prefix = bytearray()
    received = 0
    for chunk in response.iter_content(chunk_size=4096):
        if len(prefix) < 4:
            prefix.extend(chunk[: 4 - len(prefix)])
        received += len(chunk)
        if status == 200 and received > 4:
            break
    first_audio_at = time.perf_counter()
    response.close()
    ended = time.perf_counter()
    sample_rate = (
        struct.unpack("<I", bytes(prefix))[0]
        if status == 200 and len(prefix) == 4
        else None
    )
    return {
        "started_monotonic": started,
        "ended_monotonic": ended,
        "status": status,
        "sample_rate": sample_rate,
        "bytes_before_close": received,
        "first_audio_ms": (first_audio_at - started) * 1000,
        "passed": bool(
            status == 200
            and sample_rate == expected_sample_rate
            and received > 4
        ),
    }


def after_barrier(barrier: threading.Barrier, call, delay_seconds: float = 0):
    barrier.wait()
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return call()


def recover(
    base_url: str,
    deadline_seconds: float,
    speaker_id: int | None = None,
    text: str | None = None,
    expected_sample_rate: int = 24000,
) -> dict:
    started = time.perf_counter()
    deadline = started + deadline_seconds
    attempts = []
    result = None
    while time.perf_counter() < deadline:
        remaining = deadline - time.perf_counter()
        try:
            attempt = full_stream(
                base_url,
                text or "取消后在释放的槽位恢复正常语音。",
                max(0.1, min(90, remaining)),
                speaker_id,
                expected_sample_rate,
            )
        except requests.RequestException as exc:
            attempt = {
                "status": None,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "started_monotonic": time.perf_counter(),
                "ended_monotonic": time.perf_counter(),
            }
        attempts.append(attempt)
        if attempt["passed"]:
            result = attempt
            break
        if attempt.get("status") != 429:
            result = attempt
            break
        retry_after = attempt.get("retry_after")
        try:
            delay = float(retry_after) if retry_after is not None else 0.1
        except ValueError:
            delay = 0.1
        if time.perf_counter() + delay >= deadline:
            break
        time.sleep(min(max(delay, 0.05), 1.0))
    ended = time.perf_counter()
    result = result or (attempts[-1] if attempts else {"passed": False})
    return {
        "started_monotonic": started,
        "ended_monotonic": ended,
        "attempts": attempts,
        "result": result,
        "elapsed_ms": (ended - started) * 1000,
        "deadline_met": ended <= deadline,
        "passed": bool(result.get("passed") and ended <= deadline),
    }


def clean(value):
    if isinstance(value, dict):
        return {
            key: clean(item)
            for key, item in value.items()
            if not key.endswith("_monotonic")
        }
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18622")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--recovery-deadline", type=float, default=15)
    parser.add_argument("--speaker-cancel", type=int)
    parser.add_argument("--speaker-keep", type=int)
    parser.add_argument("--speaker-recovery", type=int)
    parser.add_argument("--cancel-text")
    parser.add_argument("--keep-text")
    parser.add_argument("--keep-repeat", type=int, default=6)
    parser.add_argument("--cancel-head-start", type=float, default=0)
    parser.add_argument("--recovery-text")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rounds = []
    for index in range(1, args.rounds + 1):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            cancel_future = pool.submit(
                after_barrier,
                barrier,
                lambda: cancel_stream(
                    args.base_url,
                    args.speaker_cancel,
                    args.cancel_text,
                    args.sample_rate,
                ),
            )
            keep_future = pool.submit(
                after_barrier,
                barrier,
                lambda: full_stream(
                    args.base_url,
                    args.keep_text
                    or "保持第二路语音连续输出，取消另一条请求不能中断这一条。"
                    * args.keep_repeat,
                    90,
                    args.speaker_keep,
                    args.sample_rate,
                ),
                args.cancel_head_start,
            )
            cancel = cancel_future.result()
            recovery = recover(
                args.base_url,
                args.recovery_deadline,
                args.speaker_recovery,
                args.recovery_text,
                args.sample_rate,
            )
            keep = keep_future.result()
        cancel_keep_overlap_ms = max(
            0.0,
            (
                min(cancel["ended_monotonic"], keep["ended_monotonic"])
                - max(cancel["started_monotonic"], keep["started_monotonic"])
            )
            * 1000,
        )
        keep_recovery_overlap_ms = max(
            0.0,
            (
                min(keep["ended_monotonic"], recovery["ended_monotonic"])
                - max(keep["started_monotonic"], recovery["started_monotonic"])
            )
            * 1000,
        )
        recovery_first_pcm = recovery["result"].get("first_pcm_monotonic")
        recovery_first_pcm_before_keep_end_ms = (
            (keep["ended_monotonic"] - recovery_first_pcm) * 1000
            if recovery_first_pcm is not None
            else None
        )
        passed = bool(
            cancel["passed"]
            and keep["passed"]
            and recovery["passed"]
            and cancel_keep_overlap_ms > 0
            and keep_recovery_overlap_ms > 0
            and recovery_first_pcm_before_keep_end_ms is not None
            and recovery_first_pcm_before_keep_end_ms > 0
        )
        rounds.append(
            {
                "round": index,
                "cancel": clean(cancel),
                "keep": clean(keep),
                "recovery": clean(recovery),
                "cancel_keep_overlap_ms": cancel_keep_overlap_ms,
                "keep_recovery_overlap_ms": keep_recovery_overlap_ms,
                "recovery_first_pcm_before_keep_end_ms":
                    recovery_first_pcm_before_keep_end_ms,
                "passed": passed,
            }
        )
        recovery_first_pcm_overlap_display = (
            f"{recovery_first_pcm_before_keep_end_ms:.1f}ms"
            if recovery_first_pcm_before_keep_end_ms is not None
            else "none"
        )
        print(
            f"CANCEL_KEEP {index}/{args.rounds}: pass={passed} "
            f"keep_pcm={keep['pcm_bytes']} "
            f"recovery={recovery['elapsed_ms']:.1f}ms "
            f"keep_recovery_overlap={keep_recovery_overlap_ms:.1f}ms "
            f"recovery_first_pcm_before_keep_end="
            f"{recovery_first_pcm_overlap_display}",
            flush=True,
        )
        if not passed:
            break
    report = {
        "test": "v091_r2_tts_n2_cancel_keep_recovery",
        "rounds_requested": args.rounds,
        "rounds_passed": sum(row["passed"] for row in rounds),
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
