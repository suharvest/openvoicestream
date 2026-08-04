#!/usr/bin/env python3
"""GDN+MTP two-client SSE correctness, TTFT, and overlap gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


ERROR_RE = re.compile(
    r"(CUDA(?: runtime)? (?:error|failure)|illegal memory access|"
    r"Myelin.*(?:error|fail|already loaded binary graph)|"
    r"(?:TensorRT|\[TRT\]).*(?:\[E\]|error|fail)|"
    r"segmentation fault|core dumped|out of memory)",
    re.IGNORECASE,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def one_request(
    url: str,
    marker: str,
    max_tokens: int,
    timeout: float,
    barrier: threading.Barrier | None = None,
) -> dict[str, Any]:
    if barrier is not None:
        barrier.wait()
    payload = {
        "model": "engines",
        "messages": [
            {
                "role": "user",
                "content": f"只回答这四个字，不要解释：{marker}",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    first_content = None
    content_parts: list[str] = []
    status = -1
    error = None
    event_count = 0
    try:
        with requests.post(
            url.rstrip("/") + "/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as response:
            status = response.status_code
            response.raise_for_status()
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                event_count += 1
                event = json.loads(data)
                choices = event.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                content = delta.get("content") or ""
                if content:
                    if first_content is None:
                        first_content = time.perf_counter()
                    content_parts.append(content)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    content = "".join(content_parts)
    return {
        "marker": marker,
        "status": status,
        "content": content,
        "contains_marker": marker in content,
        "event_count": event_count,
        "started_monotonic": started,
        "first_content_monotonic": first_content,
        "ended_monotonic": ended,
        "ttft_ms": (
            (first_content - started) * 1000 if first_content is not None else None
        ),
        "total_ms": (ended - started) * 1000,
        "error": error,
    }


def clean(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.endswith("_monotonic")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--container", required=True)
    parser.add_argument("--baseline-runs", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    baselines = [
        one_request(
            args.base_url,
            "并发正常",
            args.max_tokens,
            args.timeout,
        )
        for _ in range(args.baseline_runs)
    ]
    baseline_ok = all(
        row["status"] == 200
        and row["contains_marker"]
        and row["ttft_ms"] is not None
        and not row["error"]
        for row in baselines
    )

    rounds = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index in range(1, args.rounds + 1):
            barrier = threading.Barrier(2)
            future_a = pool.submit(
                one_request,
                args.base_url,
                "并发甲路",
                args.max_tokens,
                args.timeout,
                barrier,
            )
            future_b = pool.submit(
                one_request,
                args.base_url,
                "并发乙路",
                args.max_tokens,
                args.timeout,
                barrier,
            )
            a = future_a.result()
            b = future_b.result()
            request_overlap_ms = max(
                0.0,
                (
                    min(a["ended_monotonic"], b["ended_monotonic"])
                    - max(a["started_monotonic"], b["started_monotonic"])
                )
                * 1000,
            )
            token_overlap = (
                a["first_content_monotonic"] is not None
                and b["first_content_monotonic"] is not None
                and a["first_content_monotonic"] < b["ended_monotonic"]
                and b["first_content_monotonic"] < a["ended_monotonic"]
            )
            ok = (
                a["status"] == 200
                and b["status"] == 200
                and a["contains_marker"]
                and b["contains_marker"]
                and not a["error"]
                and not b["error"]
                and request_overlap_ms > 0
            )
            rounds.append(
                {
                    "round": index,
                    "a": clean(a),
                    "b": clean(b),
                    "request_overlap_ms": request_overlap_ms,
                    "token_streams_overlap": token_overlap,
                    "ok": ok,
                }
            )
            print(
                f"[{index:03d}/{args.rounds}] ok={ok} "
                f"ttft={a['ttft_ms']}/{b['ttft_ms']}ms "
                f"total={a['total_ms']:.1f}/{b['total_ms']:.1f}ms "
                f"token_overlap={token_overlap}",
                flush=True,
            )

    logs = subprocess.run(
        ["docker", "logs", "--since", started_iso, args.container],
        text=True,
        capture_output=True,
        check=False,
    )
    error_hits = [
        line
        for line in (logs.stdout + logs.stderr).splitlines()
        if ERROR_RE.search(line)
    ]
    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            "--format={{.RestartCount}}|{{.State.OOMKilled}}",
            args.container,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    n2_lanes = [row[lane] for row in rounds for lane in ("a", "b")]
    report = {
        "test": "gdn_mtp_n2_service",
        "baseline_runs": args.baseline_runs,
        "baseline_ok": baseline_ok,
        "baseline_ttft_ms": stats(
            [float(row["ttft_ms"]) for row in baselines if row["ttft_ms"] is not None]
        ),
        "rounds_requested": args.rounds,
        "rounds_passed": sum(bool(row["ok"]) for row in rounds),
        "n2_ttft_ms": stats(
            [float(row["ttft_ms"]) for row in n2_lanes if row["ttft_ms"] is not None]
        ),
        "n2_total_ms": stats([float(row["total_ms"]) for row in n2_lanes]),
        "token_overlap_rounds": sum(
            bool(row["token_streams_overlap"]) for row in rounds
        ),
        "baselines": [clean(row) for row in baselines],
        "rounds": rounds,
        "container_state": inspect.stdout.strip(),
        "runtime_error_hits": error_hits,
    }
    report["passed"] = (
        baseline_ok
        and report["rounds_passed"] == args.rounds
        and report["token_overlap_rounds"] == args.rounds
        and not error_hits
        and inspect.stdout.strip() == "0|false"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "baseline_ok": report["baseline_ok"],
        "baseline_ttft_ms": report["baseline_ttft_ms"],
        "rounds_passed": report["rounds_passed"],
        "n2_ttft_ms": report["n2_ttft_ms"],
        "token_overlap_rounds": report["token_overlap_rounds"],
        "container_state": report["container_state"],
        "runtime_error_hits": error_hits,
        "passed": report["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
