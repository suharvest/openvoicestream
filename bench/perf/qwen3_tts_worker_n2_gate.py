#!/usr/bin/env python3
"""Direct Qwen3-TTS worker N=2 isolation, cancellation, and recovery gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


ERROR_RE = re.compile(
    r"(CUDA(?: runtime)? (?:error|failure)|illegal memory access|"
    r"(?:TensorRT|\[TRT\]).*(?:\[E\]|error|fail)|segmentation fault|core dumped)",
    re.IGNORECASE,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--talker-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--code2wav-dir", required=True)
    parser.add_argument("--cp-dir", required=True)
    parser.add_argument("--plugin-path", required=True)
    parser.add_argument("--speaker-embedding-b64-file", required=True)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    embedding = Path(args.speaker_embedding_b64_file).read_text().strip()
    decoded = base64.b64decode(embedding, validate=True)
    if len(decoded) != 4096:
        raise RuntimeError(f"expected 4096 embedding bytes, got {len(decoded)}")

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = args.plugin_path
    env["LD_PRELOAD"] = args.plugin_path + (
        f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else ""
    )
    env["EDGE_LLM_TTS_LAZY_CODE2WAV"] = "0"
    env["QWEN3_TTS_SEED"] = "42"
    cmd = [
        args.worker,
        "--talkerEngineDir",
        args.talker_dir,
        "--tokenizerDir",
        args.tokenizer_dir,
        "--code2wavEngineDir",
        args.code2wav_dir,
        "--codePredictorEngineDir",
        args.cp_dir,
        "--max_slots",
        "2",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdin and proc.stdout and proc.stderr
    condition = threading.Condition()
    states: dict[str, dict[str, Any]] = {}
    stderr_lines: list[str] = []
    ready: dict[str, Any] = {}

    def stdout_reader() -> None:
        for raw in proc.stdout:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            with condition:
                if event.get("event") == "ready":
                    ready.update(event)
                request_id = event.get("id") or event.get("request_id")
                if request_id and request_id != "__worker__":
                    state = states.setdefault(
                        request_id, {"pcm": bytearray(), "chunks": 0}
                    )
                    state["last_event"] = event
                    if event.get("event") == "chunk":
                        state["chunks"] += 1
                        state["pcm"].extend(
                            base64.b64decode(event.get("audio_b64", ""))
                        )
                        state.setdefault("first_chunk_at", time.monotonic())
                    if event.get("event") in ("done", "error", "cancelled"):
                        state["terminal"] = event
                        state["done_at"] = time.monotonic()
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

    wait_for(lambda: bool(ready), 30, "worker ready timeout")
    if ready.get("max_slots") != 2:
        raise RuntimeError(f"worker did not create two slots: {ready}")

    text_a = "这是并发验证请求甲，用于检查语音流隔离和取消恢复。"
    text_b = "这是并发验证请求乙，用于确认第二路语音能够独立完成。"

    def payload(request_id: str, text: str) -> dict[str, Any]:
        return {
            "id": request_id,
            "text": text,
            "language": "chinese",
            "speaker_embedding_b64": embedding,
            "stream": True,
            "stream_only": True,
            "chunk_transport": "base64",
            "chunk_format": "pcm_s16le",
            "first_chunk_frames": 7,
            "chunk_frames": 10,
            "max_chunk_frames": 10,
            "max_audio_length": 50,
            "min_audio_length": 10,
            "talker_top_k": 1,
            "predictor_top_k": 1,
        }

    def send(message: dict[str, Any]) -> float:
        sent_at = time.monotonic()
        proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        return sent_at

    def request(request_id: str, text: str) -> dict[str, Any]:
        states[request_id] = {"pcm": bytearray(), "chunks": 0}
        sent_at = send(payload(request_id, text))
        wait_for(
            lambda: "terminal" in states[request_id],
            args.timeout,
            f"request timeout: {request_id}",
        )
        state = states[request_id]
        terminal = state["terminal"]
        if not terminal.get("ok", False) or not state["pcm"]:
            raise RuntimeError(f"request failed: {request_id}: {terminal}")
        return {
            "id": request_id,
            "sent_at": sent_at,
            "done_at": state["done_at"],
            "chunks": state["chunks"],
            "bytes": len(state["pcm"]),
            "sha256": digest(bytes(state["pcm"])),
        }

    baseline_a = request("baseline-a", text_a)
    baseline_b = request("baseline-b", text_b)
    if baseline_a["sha256"] == baseline_b["sha256"]:
        raise RuntimeError("distinct baseline prompts produced identical PCM")

    for request_id in ("concurrent-a", "concurrent-b"):
        states[request_id] = {"pcm": bytearray(), "chunks": 0}
    concurrent_sent = time.monotonic()
    send(payload("concurrent-a", text_a))
    send(payload("concurrent-b", text_b))
    wait_for(
        lambda: all("terminal" in states[key] for key in ("concurrent-a", "concurrent-b")),
        args.timeout,
        "full N=2 timeout",
    )
    concurrent = {}
    for request_id, baseline in (
        ("concurrent-a", baseline_a),
        ("concurrent-b", baseline_b),
    ):
        state = states[request_id]
        if not state["terminal"].get("ok", False) or not state["pcm"]:
            raise RuntimeError(f"full N=2 failed: {request_id}: {state['terminal']}")
        concurrent[request_id] = {
            "chunks": state["chunks"],
            "bytes": len(state["pcm"]),
            "sha256": digest(bytes(state["pcm"])),
            "matches_baseline": digest(bytes(state["pcm"])) == baseline["sha256"],
        }
    full_overlap = all(
        states[key].get("first_chunk_at", float("inf"))
        < max(states["concurrent-a"]["done_at"], states["concurrent-b"]["done_at"])
        for key in ("concurrent-a", "concurrent-b")
    )
    if not full_overlap or not all(item["matches_baseline"] for item in concurrent.values()):
        raise RuntimeError(f"full N=2 isolation failed: {concurrent}")

    rounds = []
    for index in range(1, args.rounds + 1):
        cancel_id = f"round-{index:03d}-cancel-a"
        keep_id = f"round-{index:03d}-keep-b"
        recovery_id = f"round-{index:03d}-recovery-b"
        states[cancel_id] = {"pcm": bytearray(), "chunks": 0}
        states[keep_id] = {"pcm": bytearray(), "chunks": 0}
        send(payload(cancel_id, text_a))
        send(payload(keep_id, text_b))
        wait_for(
            lambda: states[cancel_id]["chunks"] >= 1,
            args.timeout,
            f"cancel stream did not start: {cancel_id}",
        )
        cancel_at = send({"type": "cancel", "id": cancel_id})
        wait_for(
            lambda: "terminal" in states[cancel_id] and "terminal" in states[keep_id],
            args.timeout,
            f"cancel/keep timeout: round {index}",
        )
        keep = states[keep_id]
        recovery = request(recovery_id, text_b)
        keep_digest = digest(bytes(keep["pcm"]))
        ok = (
            bool(states[cancel_id]["pcm"])
            and keep["terminal"].get("ok", False)
            and bool(keep["pcm"])
            and keep_digest == baseline_b["sha256"]
            and recovery["sha256"] == baseline_b["sha256"]
            and cancel_at < keep["done_at"]
        )
        record = {
            "round": index,
            "ok": ok,
            "cancel_chunks": states[cancel_id]["chunks"],
            "cancel_terminal": states[cancel_id]["terminal"],
            "keep_chunks": keep["chunks"],
            "keep_matches_baseline": keep_digest == baseline_b["sha256"],
            "recovery_matches_baseline": recovery["sha256"] == baseline_b["sha256"],
            "cancel_before_keep_done": cancel_at < keep["done_at"],
        }
        rounds.append(record)
        print(
            f"[{index:03d}/{args.rounds}] ok={ok} "
            f"cancel_chunks={record['cancel_chunks']} keep_chunks={record['keep_chunks']}",
            flush=True,
        )
        if not ok:
            raise RuntimeError(f"N=2 cancellation gate failed: {record}")

    proc.stdin.close()
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    error_hits = [line for line in stderr_lines if ERROR_RE.search(line)]
    report = {
        "ready": ready,
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "concurrent_sent_at": concurrent_sent,
        "concurrent": concurrent,
        "full_overlap": full_overlap,
        "rounds_requested": args.rounds,
        "rounds_passed": sum(bool(item["ok"]) for item in rounds),
        "rounds": rounds,
        "worker_returncode": proc.returncode,
        "stderr_error_hits": error_hits,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "ready", "concurrent", "full_overlap", "rounds_requested",
        "rounds_passed", "worker_returncode", "stderr_error_hits"
    )}, ensure_ascii=False, indent=2))
    return 0 if not error_hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
