#!/usr/bin/env python3
"""Distinct-WAV Qwen ASR N=2 service isolation/recovery gate.

Unlike the historical bare-worker harness, this script has no mel-settings or
v0.7 worker CLI dependency.  It drives the stable multipart ``POST /asr``
service contract with two different WAV/language rows, records request-window
overlap, validates each transcript independently, and checks next-request
recovery.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def transcript_from_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return transcript_from_json(value[0]) if value else ""
    if isinstance(value, dict):
        for key in ("text", "transcript", "result", "zh", "en"):
            if key in value:
                result = transcript_from_json(value[key])
                if result:
                    return result
        for item in value.values():
            result = transcript_from_json(item)
            if result:
                return result
    return ""


def post_wav(
    url: str,
    wav_path: Path,
    language: str,
    timeout: float,
    barrier: threading.Barrier | None,
) -> dict[str, Any]:
    if barrier is not None:
        barrier.wait()
    t0 = time.perf_counter()
    started_at = utc_now()
    status = -1
    text = ""
    response_excerpt = ""
    error: str | None = None
    try:
        with wav_path.open("rb") as stream:
            response = requests.post(
                url,
                files={"file": (wav_path.name, stream, "audio/wav")},
                data={"language": language},
                timeout=timeout,
            )
        status = response.status_code
        response_excerpt = response.text[:1000]
        response.raise_for_status()
        text = transcript_from_json(response.json()).strip()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    return {
        "started_at": started_at,
        "start_monotonic": t0,
        "end_monotonic": ended,
        "elapsed_ms": (ended - t0) * 1000,
        "status": status,
        "text": text,
        "response_excerpt": response_excerpt,
        "error": error,
    }


def matches(text: str, expected: str) -> bool:
    return re.search(expected, text, re.IGNORECASE) is not None


def clean_timing(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if not k.endswith("_monotonic")}


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


def scan_container_logs(container: str | None, since: str) -> dict[str, Any]:
    if not container:
        return {"container": None, "error": None, "hits": []}
    completed = subprocess.run(
        ["docker", "logs", "--since", since, container],
        text=True,
        capture_output=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    hits = [
        {"container": container, "line": line_no, "text": line[:1000]}
        for line_no, line in enumerate(combined.splitlines(), 1)
        if ERROR_RE.search(line)
    ]
    error = (
        f"docker logs exited {completed.returncode}: {completed.stderr[:1000]}"
        if completed.returncode != 0
        else None
    )
    return {"container": container, "error": error, "hits": hits}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    started_at = utc_now()
    url = args.base_url.rstrip("/") + "/" + args.endpoint.lstrip("/")
    rounds: list[dict[str, Any]] = []
    for index in range(1, args.rounds + 1):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(
                post_wav, url, args.wav_a, args.language_a, args.timeout, barrier
            )
            future_b = pool.submit(
                post_wav, url, args.wav_b, args.language_b, args.timeout, barrier
            )
            result_a = future_a.result()
            result_b = future_b.result()
        recovery = post_wav(
            url, args.wav_a, args.language_a, args.timeout, barrier=None
        )
        overlap_ms = max(
            0.0,
            (
                min(result_a["end_monotonic"], result_b["end_monotonic"])
                - max(result_a["start_monotonic"], result_b["start_monotonic"])
            )
            * 1000,
        )
        a_matches = matches(result_a["text"], args.expect_a)
        b_matches = matches(result_b["text"], args.expect_b)
        recovery_matches = matches(recovery["text"], args.expect_a)
        round_ok = (
            result_a["status"] == 200
            and result_b["status"] == 200
            and recovery["status"] == 200
            and not result_a["error"]
            and not result_b["error"]
            and not recovery["error"]
            and overlap_ms > 0
            and a_matches
            and b_matches
            and recovery_matches
        )
        rounds.append(
            {
                "round": index,
                "a": clean_timing(result_a),
                "b": clean_timing(result_b),
                "recovery_a": clean_timing(recovery),
                "request_window_overlap_ms": overlap_ms,
                "a_matches": a_matches,
                "b_matches": b_matches,
                "recovery_matches": recovery_matches,
                "ok": round_ok,
            }
        )
        print(
            f"[{index:03d}/{args.rounds}] ok={round_ok} overlap={overlap_ms:.1f}ms "
            f"a={result_a['text']!r} b={result_b['text']!r}",
            flush=True,
        )

    saturation: dict[str, Any] | None = None
    if args.check_oversubscribe:
        barrier3 = threading.Barrier(3)
        jobs = [
            (args.wav_a, args.language_a),
            (args.wav_b, args.language_b),
            (args.wav_a, args.language_a),
        ]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(post_wav, url, wav, lang, args.timeout, barrier3)
                for wav, lang in jobs
            ]
            triplet = [future.result() for future in futures]
        rejected = [
            item
            for item in triplet
            if item["status"] in (429, 4429)
            or "4429" in item["response_excerpt"]
            or "too_many_sessions" in item["response_excerpt"]
        ]
        saturation = {
            "requests": [clean_timing(item) for item in triplet],
            "rejected_count": len(rejected),
            "passed": len(rejected) >= 1,
        }

    log_scan = scan_logs(args.server_log)
    container_log_scan = scan_container_logs(args.container, started_at)
    rounds_ok = sum(bool(item["ok"]) for item in rounds)
    passed = (
        rounds_ok == args.rounds
        and (saturation is None or saturation["passed"])
        and not log_scan["missing"]
        and not log_scan["hits"]
        and not container_log_scan["error"]
        and not container_log_scan["hits"]
    )
    summary = {
        "schema_version": 1,
        "test": "qwen_asr_distinct_wav_n2_service",
        "started_at": started_at,
        "finished_at": utc_now(),
        "config": {
            "url": url,
            "rounds": args.rounds,
            "wav_a": str(args.wav_a),
            "wav_a_sha256": hashlib.sha256(args.wav_a.read_bytes()).hexdigest(),
            "wav_b": str(args.wav_b),
            "wav_b_sha256": hashlib.sha256(args.wav_b.read_bytes()).hexdigest(),
            "language_a": args.language_a,
            "language_b": args.language_b,
            "expect_a": args.expect_a,
            "expect_b": args.expect_b,
            "check_oversubscribe": args.check_oversubscribe,
            "server_logs": [str(p) for p in args.server_log],
        },
        "rounds_ok": rounds_ok,
        "rounds_failed": args.rounds - rounds_ok,
        "rounds": rounds,
        "saturation": saturation,
        "log_scan": log_scan,
        "container_log_scan": container_log_scan,
        "passed": passed,
    }
    return summary, passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8621")
    parser.add_argument("--endpoint", default="/asr")
    parser.add_argument("--wav-a", type=Path, required=True)
    parser.add_argument("--wav-b", type=Path, required=True)
    parser.add_argument("--language-a", default="Chinese")
    parser.add_argument("--language-b", default="English")
    parser.add_argument("--expect-a", required=True, help="Regex for WAV A transcript.")
    parser.add_argument("--expect-b", required=True, help="Regex for WAV B transcript.")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument(
        "--check-oversubscribe",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--server-log", action="append", type=Path, default=[])
    parser.add_argument(
        "--container",
        help="Also scan `docker logs --since <test-start>` for runtime errors.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    for field in ("wav_a", "wav_b"):
        if not getattr(args, field).is_file():
            parser.error(f"--{field.replace('_', '-')} is not a file")
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
