#!/usr/bin/env python3
"""Direct MOSS worker N=2 isolation, cancellation, and recovery gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import Any


ERROR_RE = re.compile(
    r"(CUDA(?: runtime)? (?:error|failure)|illegal memory access|"
    r"(?:TensorRT|\[TRT\]).*(?:\[E\]|error|fail)|"
    r"segmentation fault|core dumped)",
    re.IGNORECASE,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pcm_rms(data: bytes) -> float:
    samples = array("h")
    samples.frombytes(data)
    if not samples:
        return 0.0
    return math.sqrt(sum((value / 32768.0) ** 2 for value in samples) / len(samples))


def write_wav(path: Path, data: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--codec-onnx-dir", required=True)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    command = [
        args.worker,
        f"--engine-dir={args.engine_dir}",
        f"--tokenizer-model={args.engine_dir}/tokenizer.model",
        f"--codec-onnx-dir={args.codec_onnx_dir}",
        "--max-slots=2",
    ]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin and proc.stdout and proc.stderr
    condition = threading.Condition()
    states: dict[str, dict[str, Any]] = {}
    ready: dict[str, Any] = {}
    stderr_lines: list[str] = []

    def stdout_reader() -> None:
        for raw in proc.stdout:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            observed_at = time.monotonic()
            with condition:
                if event.get("event") == "worker_ready":
                    ready.update(event)
                request_id = event.get("id")
                if request_id and request_id != "__worker__":
                    state = states.setdefault(
                        request_id, {"pcm": bytearray(), "chunks": 0, "events": []}
                    )
                    state["events"].append(
                        {"at": observed_at, "event": event.get("event")}
                    )
                    if event.get("event") == "chunk":
                        state["chunks"] += 1
                        state["pcm"].extend(
                            base64.b64decode(event.get("audio_b64", ""))
                        )
                        state.setdefault("first_chunk_at", observed_at)
                    if event.get("event") in ("done", "cancelled", "error"):
                        state["terminal"] = event
                        state["done_at"] = observed_at
                condition.notify_all()

    def stderr_reader() -> None:
        for raw in proc.stderr:
            stderr_lines.append(raw.rstrip())

    threading.Thread(target=stdout_reader, daemon=True).start()
    threading.Thread(target=stderr_reader, daemon=True).start()

    def wait_for(predicate, timeout: float, description: str) -> None:
        deadline = time.monotonic() + timeout
        with condition:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(description)
                condition.wait(remaining)

    def payload(request_id: str, text: str) -> dict[str, Any]:
        return {
            "id": request_id,
            "text": text,
            "stream": True,
            "stream_only": True,
            "chunk_transport": "base64",
            "chunk_format": "pcm_s16le",
            "first_chunk_frames": 4,
            "chunk_frames": 8,
        }

    def send(message: dict[str, Any]) -> float:
        sent_at = time.monotonic()
        proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        return sent_at

    def prepare(request_id: str) -> None:
        states[request_id] = {"pcm": bytearray(), "chunks": 0, "events": []}

    def request(request_id: str, text: str) -> dict[str, Any]:
        prepare(request_id)
        sent_at = send(payload(request_id, text))
        wait_for(
            lambda: "terminal" in states[request_id],
            args.timeout,
            f"request timeout: {request_id}",
        )
        state = states[request_id]
        terminal = state["terminal"]
        if terminal.get("event") != "done" or not terminal.get("ok") or not state["pcm"]:
            raise RuntimeError(f"request failed: {request_id}: {terminal}")
        return {
            "sent_at": sent_at,
            "done_at": state["done_at"],
            "chunks": state["chunks"],
            "bytes": len(state["pcm"]),
            "sha256": digest(bytes(state["pcm"])),
        }

    wait_for(lambda: bool(ready), 30, "worker ready timeout")
    if (
        ready.get("max_slots") != 2
        or not ready.get("concurrent_dispatch")
        or not ready.get("cooperative_cancel")
    ):
        raise RuntimeError(f"worker capability contract mismatch: {ready}")

    text_a = "这是并发请求甲，用于验证两路语音能够真正同时生成。"
    text_b = "这是并发请求乙，用于确认输出隔离和取消后的恢复。"
    baseline_a = request("baseline-a", text_a)
    baseline_b = request("baseline-b", text_b)
    if baseline_a["sha256"] == baseline_b["sha256"]:
        raise RuntimeError("distinct prompts produced identical PCM")

    def run_pair(prefix: str) -> dict[str, Any]:
        id_a, id_b = f"{prefix}-a", f"{prefix}-b"
        prepare(id_a)
        prepare(id_b)
        send(payload(id_a, text_a))
        send(payload(id_b, text_b))
        wait_for(
            lambda: all("terminal" in states[item] for item in (id_a, id_b)),
            args.timeout,
            f"N=2 timeout: {prefix}",
        )
        a, b = states[id_a], states[id_b]
        overlap = (
            a.get("first_chunk_at", float("inf")) < b["done_at"]
            and b.get("first_chunk_at", float("inf")) < a["done_at"]
        )
        result = {
            "overlap": overlap,
            "a_matches": digest(bytes(a["pcm"])) == baseline_a["sha256"],
            "b_matches": digest(bytes(b["pcm"])) == baseline_b["sha256"],
            "a_rms": pcm_rms(bytes(a["pcm"])),
            "b_rms": pcm_rms(bytes(b["pcm"])),
            "outputs_distinct": digest(bytes(a["pcm"])) != digest(bytes(b["pcm"])),
            "a_chunks": a["chunks"],
            "b_chunks": b["chunks"],
        }
        if (
            a["terminal"].get("event") != "done"
            or b["terminal"].get("event") != "done"
            or not result["overlap"]
            or not result["outputs_distinct"]
            or min(result["a_rms"], result["b_rms"]) <= 0.005
        ):
            raise RuntimeError(f"N=2 isolation failed: {prefix}: {result}")
        if prefix == "initial":
            evidence_dir = Path(args.output).with_suffix("")
            evidence_dir.mkdir(parents=True, exist_ok=True)
            write_wav(
                evidence_dir / "initial-a.wav",
                bytes(a["pcm"]),
                int(ready["sample_rate"]),
                int(ready["channels"]),
            )
            write_wav(
                evidence_dir / "initial-b.wav",
                bytes(b["pcm"]),
                int(ready["sample_rate"]),
                int(ready["channels"]),
            )
        return result

    initial_pair = run_pair("initial")
    rounds: list[dict[str, Any]] = []
    for index in range(1, args.rounds + 1):
        cancel_id = f"round-{index:03d}-cancel-a"
        keep_id = f"round-{index:03d}-keep-b"
        recovery_id = f"round-{index:03d}-recovery-b"
        prepare(cancel_id)
        prepare(keep_id)
        send(payload(cancel_id, text_a))
        send(payload(keep_id, text_b))
        wait_for(
            lambda: states[cancel_id]["chunks"] >= 1,
            args.timeout,
            f"cancel stream did not start: {cancel_id}",
        )
        cancel_at = send({"type": "cancel", "id": cancel_id})
        wait_for(
            lambda: all(
                "terminal" in states[item] for item in (cancel_id, keep_id)
            ),
            args.timeout,
            f"cancel/keep timeout: {index}",
        )
        cancelled, kept = states[cancel_id], states[keep_id]
        recovery = request(recovery_id, text_b)
        record = {
            "round": index,
            "cancel_terminal": cancelled["terminal"].get("event"),
            "cancel_chunks": cancelled["chunks"],
            "keep_matches": digest(bytes(kept["pcm"])) == baseline_b["sha256"],
            "recovery_matches": recovery["sha256"] == baseline_b["sha256"],
            "keep_rms": pcm_rms(bytes(kept["pcm"])),
            "recovery_rms": pcm_rms(bytes(states[recovery_id]["pcm"])),
            "cancel_before_keep_done": cancel_at < kept["done_at"],
        }
        record["ok"] = (
            record["cancel_terminal"] == "cancelled"
            and record["cancel_chunks"] >= 1
            and kept["terminal"].get("event") == "done"
            and record["keep_rms"] > 0.005
            and record["recovery_rms"] > 0.005
            and record["cancel_before_keep_done"]
        )
        rounds.append(record)
        print(
            f"[{index:03d}/{args.rounds}] ok={record['ok']} "
            f"cancel_chunks={record['cancel_chunks']}",
            flush=True,
        )
        if not record["ok"]:
            raise RuntimeError(f"cancel isolation failed: {record}")

    proc.stdin.close()
    proc.wait(timeout=15)
    error_hits = [line for line in stderr_lines if ERROR_RE.search(line)]
    report = {
        "schema_version": 1,
        "test": "moss_worker_n2_cancel_recovery",
        "command": command,
        "ready": ready,
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "initial_pair": initial_pair,
        "rounds_requested": args.rounds,
        "rounds_passed": sum(bool(item["ok"]) for item in rounds),
        "rounds": rounds,
        "worker_returncode": proc.returncode,
        "stderr_error_hits": error_hits,
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "ready": ready,
                "initial_pair": initial_pair,
                "rounds_requested": args.rounds,
                "rounds_passed": report["rounds_passed"],
                "worker_returncode": proc.returncode,
                "stderr_error_hits": error_hits,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not error_hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
