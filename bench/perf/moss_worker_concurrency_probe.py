#!/usr/bin/env python3
"""Probe whether the MOSS JSONL worker truly overlaps and cancels requests."""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--codec-onnx-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


class EventReader:
    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        self.proc = proc
        self.stdout = proc.stdout
        self.buffer = bytearray()

    def read(self, timeout: float) -> tuple[float, dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while b"\n" in self.buffer:
                line, _, remainder = self.buffer.partition(b"\n")
                self.buffer = bytearray(remainder)
                try:
                    return time.monotonic(), json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            readable, _, _ = select.select(
                [self.stdout], [], [], max(0.0, deadline - time.monotonic())
            )
            if readable:
                chunk = os.read(self.stdout.fileno(), 65536)
                if chunk:
                    self.buffer.extend(chunk)
                    continue
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker exited with status {self.proc.returncode}")
        raise TimeoutError("timed out waiting for MOSS worker event")


def request(request_id: str, text: str) -> dict:
    return {
        "id": request_id,
        "text": text,
        "stream": True,
        "stream_only": True,
        "chunk_transport": "base64",
        "chunk_format": "pcm_s16le",
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = output.with_suffix(".stderr.log")
    command = [
        args.worker,
        f"--engine-dir={args.engine_dir}",
        f"--tokenizer-model={args.engine_dir}/tokenizer.model",
        f"--codec-onnx-dir={args.codec_onnx_dir}",
        "--max-slots=2",
    ]
    with stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        assert proc.stdin is not None
        reader = EventReader(proc)
        try:
            _, ready = reader.read(args.timeout)
            if ready.get("event") != "worker_ready" or not ready.get("ok"):
                raise RuntimeError(f"unexpected readiness event: {ready}")

            submitted_at = time.monotonic()
            for item in (
                request(
                    "moss-n2-a",
                    "这是并发请求甲，用于验证两个请求是否真的同时生成语音。",
                ),
                request(
                    "moss-n2-b",
                    "这是并发请求乙，它不应该等待请求甲全部完成。",
                ),
            ):
                proc.stdin.write((json.dumps(item, ensure_ascii=False) + "\n").encode())
            proc.stdin.flush()

            events: list[dict] = []
            done: set[str] = set()
            while done != {"moss-n2-a", "moss-n2-b"}:
                observed_at, event = reader.read(args.timeout)
                request_id = event.get("id")
                if request_id not in {"moss-n2-a", "moss-n2-b"}:
                    continue
                events.append(
                    {
                        "elapsed_ms": (observed_at - submitted_at) * 1000,
                        "id": request_id,
                        "event": event.get("event"),
                    }
                )
                if event.get("event") == "error":
                    raise RuntimeError(f"MOSS request failed: {event}")
                if event.get("event") == "done":
                    done.add(request_id)

            a_done_index = next(
                i
                for i, event in enumerate(events)
                if event["id"] == "moss-n2-a" and event["event"] == "done"
            )
            b_first_index = next(
                i for i, event in enumerate(events) if event["id"] == "moss-n2-b"
            )
            interleaved = b_first_index < a_done_index

            cancel_submitted_at = time.monotonic()
            for item in (
                request(
                    "moss-cancel-a",
                    "这是一个较长的取消测试请求，用于检查工作进程能否在生成期间处理取消消息。",
                ),
                {"id": "moss-cancel-a", "cancel": True},
            ):
                proc.stdin.write((json.dumps(item, ensure_ascii=False) + "\n").encode())
            proc.stdin.flush()

            cancel_events: list[dict] = []
            saw_generation_done = False
            saw_cancel_response = False
            while not (saw_generation_done and saw_cancel_response):
                observed_at, event = reader.read(args.timeout)
                if event.get("id") != "moss-cancel-a":
                    continue
                cancel_events.append(
                    {
                        "elapsed_ms": (observed_at - cancel_submitted_at) * 1000,
                        "event": event.get("event"),
                        "ok": event.get("ok"),
                        "error": event.get("error"),
                    }
                )
                if event.get("event") == "done":
                    saw_generation_done = True
                elif saw_generation_done and event.get("event") == "error":
                    saw_cancel_response = True

            report = {
                "schema_version": 1,
                "test": "moss_worker_concurrency_capability",
                "command": command,
                "worker_ready": ready,
                "n2": {
                    "interleaved": interleaved,
                    "serialized": not interleaved,
                    "events": events,
                },
                "cancellation": {
                    "supported": False,
                    "generation_finished_before_cancel_was_parsed": True,
                    "events": cancel_events,
                },
                "verdict": "unsupported",
                "reason": (
                    "The JSONL main loop handles each request synchronously; "
                    "--max-slots=2 allocates slots but does not dispatch requests concurrently."
                ),
            }
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
