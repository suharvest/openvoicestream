#!/usr/bin/env python3
"""Run deterministic Base TTS N=1 concurrently with one GDN+MTP request."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--talker-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--code2wav-dir", required=True)
    parser.add_argument("--cp-dir", required=True)
    parser.add_argument("--plugin-path", required=True)
    parser.add_argument("--speaker-embedding-b64-file", required=True)
    parser.add_argument("--llm-inference", required=True)
    parser.add_argument("--gdn-engine-dir", required=True)
    parser.add_argument("--gdn-input", required=True)
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()

    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    embedding = Path(args.speaker_embedding_b64_file).read_text().strip()
    if len(base64.b64decode(embedding, validate=True)) != 4096:
        raise RuntimeError("invalid external speaker embedding")

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = args.plugin_path
    env["LD_PRELOAD"] = args.plugin_path + (
        f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else ""
    )
    env["EDGE_LLM_TTS_LAZY_CODE2WAV"] = "0"
    env["QWEN3_TTS_SEED"] = "42"
    worker = subprocess.Popen(
        [
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
            "1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    assert worker.stdin and worker.stdout and worker.stderr
    condition = threading.Condition()
    ready: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    worker_stderr: list[str] = []

    def read_stdout() -> None:
        for raw in worker.stdout:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            with condition:
                if event.get("event") == "ready":
                    ready.update(event)
                request_id = event.get("id") or event.get("request_id")
                if request_id and request_id != "__worker__":
                    state = states.setdefault(request_id, {"pcm": bytearray()})
                    if event.get("event") == "chunk":
                        state["pcm"].extend(base64.b64decode(event.get("audio_b64", "")))
                    if event.get("event") in ("done", "error", "cancelled"):
                        state["terminal"] = event
                condition.notify_all()

    def read_stderr() -> None:
        worker_stderr.extend(line.rstrip() for line in worker.stderr)

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()

    def wait_for(predicate, timeout: float, label: str) -> None:
        deadline = time.monotonic() + timeout
        with condition:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(label)
                condition.wait(remaining)

    wait_for(lambda: bool(ready), 30, "worker ready")
    if ready.get("max_slots") != 1:
        raise RuntimeError(f"unexpected worker capacity: {ready}")

    text = "这是大语言模型和语音合成同时运行的共驻验证请求。"

    def run_tts(request_id: str) -> dict[str, Any]:
        states[request_id] = {"pcm": bytearray()}
        payload = {
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
        started = time.monotonic()
        worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        worker.stdin.flush()
        wait_for(
            lambda: "terminal" in states[request_id],
            60,
            f"TTS timeout: {request_id}",
        )
        ended = time.monotonic()
        state = states[request_id]
        if not state["terminal"].get("ok", False) or not state["pcm"]:
            raise RuntimeError(f"TTS failed: {state['terminal']}")
        return {
            "started": started,
            "ended": ended,
            "bytes": len(state["pcm"]),
            "sha256": hashlib.sha256(bytes(state["pcm"])).hexdigest(),
        }

    baseline = run_tts("baseline")
    gdn_output = evidence / "gdn-output.json"
    gdn_profile = evidence / "gdn-profile.json"
    gdn_log = evidence / "gdn.log"
    with gdn_log.open("w") as log:
        gdn_started = time.monotonic()
        gdn = subprocess.Popen(
            [
                args.llm_inference,
                f"--inputFile={args.gdn_input}",
                f"--engineDir={args.gdn_engine_dir}",
                f"--outputFile={gdn_output}",
                f"--profileOutputFile={gdn_profile}",
                "--dumpOutput",
                "--dumpProfile",
                "--specDecode",
                "--specDraftTopK=1",
                "--specDraftStep=3",
                "--specVerifySize=4",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        mixed = run_tts("mixed")
        gdn_returncode = gdn.wait(timeout=60)
        gdn_ended = time.monotonic()
    recovery = run_tts("recovery")

    worker.stdin.close()
    worker.terminate()
    try:
        worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait(timeout=5)

    output_text = gdn_output.read_text(errors="replace") if gdn_output.exists() else ""
    stderr_hits = [line for line in worker_stderr if ERROR_RE.search(line)]
    overlap = mixed["started"] < gdn_ended and gdn_started < mixed["ended"]
    report = {
        "ready": ready,
        "baseline": baseline,
        "mixed": mixed,
        "recovery": recovery,
        "mixed_matches_baseline": mixed["sha256"] == baseline["sha256"],
        "recovery_matches_baseline": recovery["sha256"] == baseline["sha256"],
        "gdn_returncode": gdn_returncode,
        "gdn_answer_ok": "2 plus 2 equals 4" in output_text,
        "overlap": overlap,
        "worker_returncode": worker.returncode,
        "worker_stderr_error_hits": stderr_hits,
    }
    (evidence / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (
        report["mixed_matches_baseline"]
        and report["recovery_matches_baseline"]
        and gdn_returncode == 0
        and report["gdn_answer_ok"]
        and overlap
        and not stderr_hits
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
