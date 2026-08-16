#!/usr/bin/env python3
"""Black-box Qwen3.5 HTTP + Qwen3-TTS Base co-residency gate."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import requests


RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
ERROR_RE = re.compile(
    r"CUDA.*(?:error|failure|illegal)|TensorRT.*(?:\[E\]|error|fail)|"
    r"segmentation fault|core dumped|assert(?:ion)? failed",
    re.IGNORECASE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--llm-requests", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    embedding = args.embedding.read_text(encoding="utf-8").strip()
    decoded = base64.b64decode(embedding, validate=True)
    if len(decoded) != 4096:
        raise SystemExit(f"speaker embedding is {len(decoded)} bytes, expected 4096")

    request = {
        "id": "co-resident-base",
        "text": "你好，这是零点一零版本的语言模型和语音共驻验证。",
        "speaker_embedding_b64": embedding,
        "stream": True,
        "first_chunk_frames": 8,
        "chunk_frames": 10,
        "chunk_format": "pcm_s16le",
        "chunk_transport": "base64",
        "max_audio_length": 128,
        "talker_top_k": 1,
        "talker_temperature": 1.0,
    }
    command = [
        str(args.worker),
        f"--talkerEngineDir={args.engine_root / 'talker'}",
        f"--codePredictorEngineDir={args.engine_root / 'code_predictor'}",
        f"--code2wavEngineDir={args.engine_root / 'code2wav'}",
        f"--cloneEncoderDir={args.engine_root / 'clone_encoders'}",
        f"--tokenizerDir={args.engine_root / 'talker'}",
        "--max_slots=1",
    ]
    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = str(args.plugin)
    env["LD_PRELOAD"] = str(args.plugin)
    env["LD_LIBRARY_PATH"] = str(args.plugin.parent) + ":" + env.get("LD_LIBRARY_PATH", "")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="v010-coresident-") as tmp:
        tmp_path = Path(tmp)
        voice_stdout = (tmp_path / "voice.stdout").open("wb")
        voice_stderr = (tmp_path / "voice.stderr").open("wb")
        tegra_log = (tmp_path / "tegrastats.log").open("wb")
        tegra = subprocess.Popen(
            ["tegrastats", "--interval", "200"], stdout=tegra_log, stderr=subprocess.STDOUT
        )
        voice = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=voice_stdout,
            stderr=voice_stderr,
            env=env,
        )
        assert voice.stdin is not None
        voice.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
        voice.stdin.close()

        llm_runs = []
        session = requests.Session()
        deadline = time.monotonic() + args.timeout
        for index in range(args.llm_requests):
            if time.monotonic() >= deadline:
                break
            call_started = time.monotonic()
            try:
                response = session.post(
                    args.base_url.rstrip("/") + "/v1/chat/completions",
                    json={
                        "model": args.model,
                        "messages": [{"role": "user", "content": f"Reply with exactly: co-{index}"}],
                        "max_tokens": 12,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": 1,
                        "stream": False,
                    },
                    timeout=args.timeout,
                )
                body = response.json()
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                llm_runs.append(
                    {
                        "index": index,
                        "status": response.status_code,
                        "elapsed_ms": round((time.monotonic() - call_started) * 1000, 3),
                        "content": content,
                        "ok": response.status_code == 200 and bool(content),
                    }
                )
            except Exception as exc:
                llm_runs.append({"index": index, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

        remaining = max(1.0, deadline - time.monotonic())
        try:
            voice_returncode = voice.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            voice.kill()
            voice_returncode = voice.wait()
        voice_stdout.close()
        voice_stderr.close()
        tegra.terminate()
        try:
            tegra.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tegra.kill()
            tegra.wait()
        tegra_log.close()

        stdout_text = (tmp_path / "voice.stdout").read_text(encoding="utf-8", errors="replace")
        stderr_text = (tmp_path / "voice.stderr").read_text(encoding="utf-8", errors="replace")
        events = []
        for line in stdout_text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        chunks = sorted(
            (item for item in events if item.get("event") == "chunk" and item.get("ok")),
            key=lambda item: item.get("chunk_index", 0),
        )
        pcm = b"".join(base64.b64decode(item["audio_b64"], validate=True) for item in chunks)
        done = [item for item in events if item.get("event") == "done" and item.get("ok")]
        worker_errors = [item for item in events if item.get("event") == "error"]
        error_lines = [line for line in stderr_text.splitlines() if ERROR_RE.search(line)]
        ram_samples = [
            int(match.group(1))
            for match in map(RAM_RE.search, (tmp_path / "tegrastats.log").read_text(errors="replace").splitlines())
            if match
        ]

    voice_ok = voice_returncode == 0 and bool(done) and bool(pcm) and not worker_errors and not error_lines
    llm_ok = len(llm_runs) == args.llm_requests and all(item.get("ok") for item in llm_runs)
    result = {
        "schema_version": 1,
        "test": "edgellm_v010_llm_tts_base_co_residency",
        "elapsed_s": round(time.monotonic() - started, 3),
        "llm": {"requested": args.llm_requests, "completed": len(llm_runs), "runs": llm_runs, "ok": llm_ok},
        "voice": {
            "returncode": voice_returncode,
            "chunks": len(chunks),
            "pcm_bytes": len(pcm),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest() if pcm else None,
            "done": done[-1] if done else None,
            "errors": worker_errors,
            "stderr_error_lines": error_lines,
            "ok": voice_ok,
        },
        "memory": {
            "samples": len(ram_samples),
            "peak_ram_used_mb": max(ram_samples) if ram_samples else None,
            "min_ram_used_mb": min(ram_samples) if ram_samples else None,
        },
        "passed": voice_ok and llm_ok,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
