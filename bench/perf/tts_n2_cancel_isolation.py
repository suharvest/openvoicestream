#!/usr/bin/env python3
"""HTTP TTS N=2 cancel-A / continue-B isolation and recovery gate."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


ERROR_RE = re.compile(
    r"(CUDA(?: runtime)? (?:error|failure)|CUDA.*illegal|"
    r"Myelin.*(?:error|fail|already loaded binary graph)|"
    r"(?:TensorRT|\[TRT\]).*(?:\[E\]|error|fail)|"
    r"illegal memory access|assert(?:ion)?(?: failed| failure| error|:)|"
    r"segmentation fault|core dumped|worker.*(?:exit|crash))",
    re.IGNORECASE,
)
SR_HEADER_BYTES = 4


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def body_digest(body: bytes) -> dict[str, Any]:
    pcm = body[SR_HEADER_BYTES:] if len(body) >= SR_HEADER_BYTES else b""
    sample_rate = int.from_bytes(body[:SR_HEADER_BYTES], "little") if len(body) >= 4 else None
    return {
        "bytes_total": len(body),
        "pcm_bytes": len(pcm),
        "sha256": hashlib.sha256(pcm).hexdigest(),
        "sample_rate": sample_rate,
    }


def full_request(
    url: str, payload: dict[str, Any], timeout: float, barrier: threading.Barrier | None
) -> tuple[dict[str, Any], bytes]:
    if barrier is not None:
        barrier.wait()
    t0 = time.perf_counter()
    started_at = utc_now()
    first_byte_at: float | None = None
    body_parts: list[bytes] = []
    status = -1
    error: str | None = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
            status = response.status_code
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                body_parts.append(chunk)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    body = b"".join(body_parts)
    return (
        {
            "started_at": started_at,
            "start_monotonic": t0,
            "first_byte_monotonic": first_byte_at,
            "end_monotonic": ended,
            "ttfb_ms": (first_byte_at - t0) * 1000 if first_byte_at else None,
            "elapsed_ms": (ended - t0) * 1000,
            "status": status,
            "error": error,
            **body_digest(body),
        },
        body,
    )


def cancel_request(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    barrier: threading.Barrier,
    cancel_after_pcm_chunks: int,
) -> dict[str, Any]:
    barrier.wait()
    t0 = time.perf_counter()
    started_at = utc_now()
    status = -1
    error: str | None = None
    first_byte_at: float | None = None
    chunks_seen = 0
    bytes_seen = 0
    header_remaining = SR_HEADER_BYTES
    try:
        response = requests.post(url, json=payload, stream=True, timeout=timeout)
        status = response.status_code
        response.raise_for_status()
        try:
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                bytes_seen += len(chunk)
                pcm_in_chunk = len(chunk)
                if header_remaining:
                    consumed = min(header_remaining, pcm_in_chunk)
                    header_remaining -= consumed
                    pcm_in_chunk -= consumed
                if pcm_in_chunk:
                    chunks_seen += 1
                if chunks_seen >= cancel_after_pcm_chunks:
                    break
        finally:
            response.raw.close()
            response.close()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    return {
        "started_at": started_at,
        "start_monotonic": t0,
        "first_byte_monotonic": first_byte_at,
        "cancel_monotonic": ended,
        "ttfb_ms": (first_byte_at - t0) * 1000 if first_byte_at else None,
        "elapsed_ms": (ended - t0) * 1000,
        "status": status,
        "bytes_seen": bytes_seen,
        "pcm_chunks_seen": chunks_seen,
        "error": error,
    }


def scan_logs(paths: list[Path]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if ERROR_RE.search(line):
                hits.append({"path": str(path), "line": line_no, "text": line[:1000]})
    return {"paths": [str(p) for p in paths], "missing": missing, "hits": hits}


def clean_timing(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if not k.endswith("_monotonic")}


def payload_from_json(raw: str | None, text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": text}
    if raw:
        extra = json.loads(raw)
        if not isinstance(extra, dict):
            raise ValueError("payload JSON must be an object")
        payload.update(extra)
    return payload


def run(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    url = args.base_url.rstrip("/") + "/" + args.endpoint.lstrip("/")
    payload_a = payload_from_json(args.payload_a_json, args.text_a)
    payload_b = payload_from_json(args.payload_b_json, args.text_b)
    baseline, baseline_body = full_request(url, payload_b, args.timeout, None)
    baseline_ok = (
        baseline["status"] == 200
        and not baseline["error"]
        and baseline["pcm_bytes"] >= args.min_pcm_bytes
    )
    rounds: list[dict[str, Any]] = []
    if args.capture_dir:
        args.capture_dir.mkdir(parents=True, exist_ok=True)
        (args.capture_dir / "baseline-b.bin").write_bytes(baseline_body)

    for index in range(1, args.rounds + 1):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(
                cancel_request,
                url,
                payload_a,
                args.timeout,
                barrier,
                args.cancel_after_pcm_chunks,
            )
            future_b = pool.submit(full_request, url, payload_b, args.timeout, barrier)
            cancel_a = future_a.result()
            complete_b, body_b = future_b.result()
        recovery, recovery_body = full_request(url, payload_b, args.timeout, None)

        requests_overlap = (
            cancel_a["start_monotonic"] < complete_b["end_monotonic"]
            and complete_b["start_monotonic"] < cancel_a["cancel_monotonic"]
        )
        cancel_during_b = (
            complete_b["start_monotonic"]
            <= cancel_a["cancel_monotonic"]
            < complete_b["end_monotonic"]
        )
        b_matches = complete_b["sha256"] == baseline["sha256"]
        recovery_matches = recovery["sha256"] == baseline["sha256"]
        round_ok = (
            cancel_a["status"] == 200
            and not cancel_a["error"]
            and cancel_a["pcm_chunks_seen"] >= args.cancel_after_pcm_chunks
            and complete_b["status"] == 200
            and not complete_b["error"]
            and complete_b["pcm_bytes"] >= args.min_pcm_bytes
            and recovery["status"] == 200
            and not recovery["error"]
            and recovery["pcm_bytes"] >= args.min_pcm_bytes
            and requests_overlap
            and cancel_during_b
            and (b_matches or not args.require_byte_equal)
            and (recovery_matches or not args.require_byte_equal)
        )
        record = {
            "round": index,
            "cancel_a": clean_timing(cancel_a),
            "complete_b": clean_timing(complete_b),
            "recovery": clean_timing(recovery),
            "timing": {
                "requests_overlap": requests_overlap,
                "cancel_a_during_b": cancel_during_b,
                "a_start_minus_b_start_ms": (
                    cancel_a["start_monotonic"] - complete_b["start_monotonic"]
                )
                * 1000,
                "b_end_minus_a_cancel_ms": (
                    complete_b["end_monotonic"] - cancel_a["cancel_monotonic"]
                )
                * 1000,
            },
            "b_matches_baseline": b_matches,
            "recovery_matches_baseline": recovery_matches,
            "ok": round_ok,
        }
        rounds.append(record)
        if args.capture_dir:
            (args.capture_dir / f"round-{index:03d}-b.bin").write_bytes(body_b)
            (args.capture_dir / f"round-{index:03d}-recovery.bin").write_bytes(
                recovery_body
            )
        print(
            f"[{index:03d}/{args.rounds}] ok={round_ok} "
            f"overlap={requests_overlap} cancel_during_b={cancel_during_b} "
            f"b_bytes={complete_b['pcm_bytes']} recovery={recovery['status']}",
            flush=True,
        )

    log_scan = scan_logs(args.server_log)
    rounds_ok = sum(bool(item["ok"]) for item in rounds)
    passed = (
        baseline_ok
        and rounds_ok == args.rounds
        and not log_scan["missing"]
        and not log_scan["hits"]
    )
    summary = {
        "schema_version": 1,
        "test": "tts_n2_cancel_a_continue_b",
        "started_at": baseline["started_at"],
        "finished_at": utc_now(),
        "config": {
            "url": url,
            "rounds": args.rounds,
            "cancel_after_pcm_chunks": args.cancel_after_pcm_chunks,
            "min_pcm_bytes": args.min_pcm_bytes,
            "require_byte_equal": args.require_byte_equal,
            "payload_a": payload_a,
            "payload_b": payload_b,
            "server_logs": [str(p) for p in args.server_log],
            "capture_dir": str(args.capture_dir) if args.capture_dir else None,
        },
        "baseline_b": clean_timing(baseline),
        "baseline_ok": baseline_ok,
        "rounds_ok": rounds_ok,
        "rounds_failed": args.rounds - rounds_ok,
        "rounds": rounds,
        "log_scan": log_scan,
        "passed": passed,
    }
    return summary, passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8621")
    parser.add_argument("--endpoint", default="/tts/stream")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument(
        "--text-a",
        default=(
            "这是一个较长的取消请求，"
            "用于验证取消一个并发请求不会影响另一个请求。"
        ),
    )
    parser.add_argument(
        "--text-b",
        default=(
            "这是并发请求乙。"
            "它必须在请求甲取消之后继续生成完整且隔离的音频。"
        ),
    )
    parser.add_argument("--payload-a-json", help="Extra/override JSON fields for A.")
    parser.add_argument("--payload-b-json", help="Extra/override JSON fields for B.")
    parser.add_argument("--cancel-after-pcm-chunks", type=int, default=1)
    parser.add_argument("--min-pcm-bytes", type=int, default=1024)
    parser.add_argument(
        "--require-byte-equal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--server-log", action="append", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.cancel_after_pcm_chunks < 1:
        parser.error("--rounds and --cancel-after-pcm-chunks must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary, passed = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
