#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import struct
import time
from pathlib import Path

import requests


def stream_once(base_url: str, text: str, timeout: float, expected_sample_rate: int) -> dict:
    started = time.perf_counter()
    with requests.post(
        base_url.rstrip("/") + "/tts/stream",
        json={"text": text, "language": "chinese"},
        stream=True,
        timeout=timeout,
    ) as response:
        headers_at = time.perf_counter()
        chunks = []
        first_pcm_at = None
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunks.append(chunk)
            if sum(map(len, chunks)) > 4 and first_pcm_at is None:
                first_pcm_at = time.perf_counter()
        payload = b"".join(chunks)
    sample_rate = (
        struct.unpack("<I", payload[:4])[0]
        if response.status_code == 200 and len(payload) >= 4
        else None
    )
    ended = time.perf_counter()
    return {
        "status": response.status_code,
        "retry_after": response.headers.get("retry-after"),
        "content_type": response.headers.get("content-type"),
        "headers_ms": (headers_at - started) * 1000,
        "ttfa_ms": (
            (first_pcm_at - started) * 1000 if first_pcm_at is not None else None
        ),
        "wall_ms": (ended - started) * 1000,
        "sample_rate": sample_rate,
        "pcm_bytes": max(0, len(payload) - 4),
        "passed": (
            response.status_code == 200
            and sample_rate == expected_sample_rate
            and len(payload) > 4
        ),
    }


def cancel_recovery(
    base_url: str,
    timeout: float,
    expected_sample_rate: int,
    recovery_timeout: float,
) -> dict:
    long_text = "这是一个用于断开连接和协作取消验证的长语音请求。" * 12
    started = time.perf_counter()
    response = requests.post(
        base_url.rstrip("/") + "/tts/stream",
        json={"text": long_text, "language": "chinese"},
        stream=True,
        timeout=timeout,
    )
    cancel_status = response.status_code
    received = 0
    prefix = bytearray()
    for chunk in response.iter_content(chunk_size=4096):
        if len(prefix) < 4:
            prefix.extend(chunk[: 4 - len(prefix)])
        received += len(chunk)
        if cancel_status == 200 and received > 4:
            break
    first_audio_at = time.perf_counter()
    response.close()
    closed_at = time.perf_counter()
    cancel_sample_rate = (
        struct.unpack("<I", bytes(prefix))[0]
        if cancel_status == 200 and len(prefix) == 4
        else None
    )
    cancel_stream_valid = (
        cancel_status == 200
        and cancel_sample_rate == expected_sample_rate
        and received > 4
    )
    recovery_started = time.perf_counter()
    recovery_deadline = recovery_started + recovery_timeout
    attempts = []
    recovered = None
    while time.perf_counter() < recovery_deadline:
        remaining = recovery_deadline - time.perf_counter()
        try:
            attempt = stream_once(
                base_url,
                "取消后恢复正常。",
                min(timeout, max(0.001, remaining)),
                expected_sample_rate,
            )
        except requests.RequestException as exc:
            attempt = {
                "status": None,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        completed_before_deadline = time.perf_counter() <= recovery_deadline
        attempt["completed_before_deadline"] = completed_before_deadline
        attempts.append(attempt)
        if attempt["passed"]:
            if not completed_before_deadline:
                attempt["passed"] = False
                attempt["error"] = "recovery completed after wall-clock deadline"
            recovered = attempt
            break
        if attempt["status"] != 429:
            recovered = attempt
            break
        retry_after = attempt.get("retry_after")
        try:
            delay = float(retry_after) if retry_after is not None else 0.1
        except ValueError:
            delay = 0.1
        sleep_remaining = recovery_deadline - time.perf_counter()
        if sleep_remaining <= 0:
            break
        time.sleep(min(max(delay, 0.05), 1.0, sleep_remaining))
    if recovered is None:
        recovered = attempts[-1] if attempts else {"status": None, "passed": False}
    recovery_ms = (time.perf_counter() - recovery_started) * 1000
    recovery_deadline_met = (
        recovered.get("completed_before_deadline") is True
        and recovery_ms <= recovery_timeout * 1000
    )
    return {
        "cancel_status": cancel_status,
        "cancel_sample_rate": cancel_sample_rate,
        "cancel_stream_valid": cancel_stream_valid,
        "bytes_before_close": received,
        "first_audio_ms": (first_audio_at - started) * 1000,
        "close_ms": (closed_at - started) * 1000,
        "recovery": recovered,
        "recovery_attempts": attempts,
        "recovery_429_count": sum(
            attempt["status"] == 429 for attempt in attempts
        ),
        "recovery_ms": recovery_ms,
        "recovery_deadline_met": recovery_deadline_met,
        "passed": (
            cancel_stream_valid
            and recovered["passed"]
            and recovery_deadline_met
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8621")
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--recovery-timeout", type=float, default=15)
    parser.add_argument("--sample-rate", type=int, default=24000)
    args = parser.parse_args()
    texts = [
        "第一路低延迟流式语音验证。",
        "第二路并发流式语音验证。",
    ][: args.parallel]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        streams = list(
            pool.map(
                lambda text: stream_once(
                    args.base_url, text, args.timeout, args.sample_rate
                ),
                texts,
            )
        )
    cancellation = cancel_recovery(
        args.base_url,
        args.timeout,
        args.sample_rate,
        args.recovery_timeout,
    )
    report = {
        "test": "tts_http_stream_chunk_cancel",
        "parallel": args.parallel,
        "streams": streams,
        "cancellation": cancellation,
        "passed": all(item["passed"] for item in streams) and cancellation["passed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
