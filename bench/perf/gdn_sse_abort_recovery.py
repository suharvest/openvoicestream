#!/usr/bin/env python3
"""Deterministic OpenAI SSE abort -> immediate-next-request GDN gate.

This is intentionally an HTTP black-box harness.  It closes the first
streaming response after a configurable number of content events, starts the
next request immediately in the same thread, checks service health after every
cycle, and writes the complete timing/event record as JSON.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests


ERROR_RE = re.compile(
    r"(CUDA(?: runtime)? (?:error|failure)|CUDA.*illegal|"
    r"Myelin.*(?:error|fail|already loaded binary graph)|"
    r"(?:TensorRT|\[TRT\]).*(?:\[E\]|error|fail)|"
    r"illegal memory access|assert(?:ion)?(?: failed| failure| error|:)|"
    r"segmentation fault|core dumped|worker.*(?:exit|crash)|"
    r"execute_async.*(?:false|fail))",
    re.IGNORECASE,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_sse_data(line: bytes) -> dict[str, Any] | None:
    if not line.startswith(b"data:"):
        return None
    raw = line[5:].strip()
    if not raw or raw == b"[DONE]":
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def content_from_event(event: dict[str, Any]) -> str:
    try:
        value = event["choices"][0]["delta"].get("content", "")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return value if isinstance(value, str) else ""


def health_check(
    session: requests.Session, base_url: str, path: str, timeout: float
) -> dict[str, Any]:
    started = time.perf_counter()
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    try:
        response = session.get(url, timeout=timeout)
        body = response.text[:512]
        return {
            "ok": 200 <= response.status_code < 300,
            "status": response.status_code,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "body_excerpt": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": -1,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "error": f"{type(exc).__name__}: {exc}",
        }


def stream_call(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    abort_after_events: int | None,
) -> dict[str, Any]:
    started_wall = utc_now()
    started = time.perf_counter()
    response: requests.Response | None = None
    event_times_ms: list[float] = []
    request_id: str | None = None
    content = ""
    done_seen = False
    error: str | None = None
    status = -1
    try:
        response = session.post(url, json=payload, stream=True, timeout=timeout)
        status = response.status_code
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if line.strip() == b"data: [DONE]":
                done_seen = True
                break
            event = parse_sse_data(line)
            if event is None:
                continue
            request_id = request_id or event.get("id")
            delta = content_from_event(event)
            if not delta:
                continue
            content += delta
            event_times_ms.append((time.perf_counter() - started) * 1000)
            if abort_after_events is not None and len(event_times_ms) >= abort_after_events:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Closing both urllib3's raw stream and Response makes the client-side
        # disconnect explicit before the immediate recovery request begins.
        if response is not None:
            try:
                response.raw.close()
            finally:
                response.close()
    ended = time.perf_counter()
    return {
        "started_at": started_wall,
        "status": status,
        "request_id": request_id,
        "event_count": len(event_times_ms),
        "event_times_ms": event_times_ms,
        "ttft_ms": event_times_ms[0] if event_times_ms else None,
        "elapsed_ms": (ended - started) * 1000,
        "ended_monotonic": ended,
        "done_seen": done_seen,
        "content": content,
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
                hits.append(
                    {"path": str(path), "line": line_no, "text": line[:1000]}
                )
    return {"paths": [str(p) for p in paths], "missing": missing, "hits": hits}


def make_payload(
    model: str, prompt: str, max_tokens: int, seed: int
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Greedy selection is deterministic without depending on whether the
        # server/runtime implements the optional OpenAI ``seed`` field.
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": seed,
        "stream": True,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    base_url = args.base_url.rstrip("/")
    url = base_url + "/v1/chat/completions"
    session = requests.Session()
    rounds: list[dict[str, Any]] = []
    initial_health = health_check(session, base_url, args.health_path, args.timeout)
    for index in range(1, args.rounds + 1):
        aborted = stream_call(
            session,
            url,
            make_payload(args.model, args.abort_prompt, args.abort_max_tokens, args.seed),
            args.timeout,
            abort_after_events=args.abort_after_events,
        )
        recovery_started = time.perf_counter()
        abort_to_next_ms = (recovery_started - aborted["ended_monotonic"]) * 1000
        recovered = stream_call(
            session,
            url,
            make_payload(
                args.model, args.recovery_prompt, args.recovery_max_tokens, args.seed
            ),
            args.timeout,
            abort_after_events=None,
        )
        health = health_check(session, base_url, args.health_path, args.timeout)
        round_ok = (
            aborted["status"] == 200
            and aborted["event_count"] >= args.abort_after_events
            and recovered["status"] == 200
            and recovered["done_seen"]
            and recovered["event_count"] > 0
            and not recovered["error"]
            and health["ok"]
            and abort_to_next_ms <= args.max_next_delay_ms
        )
        record = {
            "round": index,
            "abort": {k: v for k, v in aborted.items() if k != "ended_monotonic"},
            "abort_to_next_start_ms": abort_to_next_ms,
            "recovery": {k: v for k, v in recovered.items() if k != "ended_monotonic"},
            "health": health,
            "ok": round_ok,
        }
        rounds.append(record)
        print(
            f"[{index:03d}/{args.rounds}] ok={round_ok} "
            f"abort_events={aborted['event_count']} "
            f"next_delay={abort_to_next_ms:.2f}ms "
            f"recovery_ttft={recovered['ttft_ms']} health={health['status']}",
            flush=True,
        )
    final_health = health_check(session, base_url, args.health_path, args.timeout)
    log_scan = scan_logs(args.server_log)
    ok_rounds = sum(bool(item["ok"]) for item in rounds)
    passed = (
        initial_health["ok"]
        and final_health["ok"]
        and ok_rounds == args.rounds
        and not log_scan["missing"]
        and not log_scan["hits"]
    )
    summary = {
        "schema_version": 1,
        "test": "gdn_sse_abort_immediate_recovery",
        "started_at": rounds[0]["abort"]["started_at"] if rounds else utc_now(),
        "finished_at": utc_now(),
        "config": {
            "base_url": base_url,
            "model": args.model,
            "rounds": args.rounds,
            "abort_after_events": args.abort_after_events,
            "max_next_delay_ms": args.max_next_delay_ms,
            "health_path": args.health_path,
            "server_logs": [str(p) for p in args.server_log],
            "seed": args.seed,
        },
        "initial_health": initial_health,
        "final_health": final_health,
        "rounds_ok": ok_rounds,
        "rounds_failed": args.rounds - ok_rounds,
        "rounds": rounds,
        "log_scan": log_scan,
        "passed": passed,
    }
    return summary, passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--abort-after-events", type=int, default=1)
    parser.add_argument("--abort-max-tokens", type=int, default=256)
    parser.add_argument("--recovery-max-tokens", type=int, default=16)
    parser.add_argument("--abort-prompt", default="Count from one to one hundred.")
    parser.add_argument("--recovery-prompt", default="Reply with exactly: recovery-ok")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-next-delay-ms", type=float, default=100)
    parser.add_argument(
        "--server-log",
        action="append",
        type=Path,
        required=True,
        help="Server log captured for the test interval; repeat for multiple logs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.abort_after_events < 1:
        parser.error("--rounds and --abort-after-events must be >= 1")
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
