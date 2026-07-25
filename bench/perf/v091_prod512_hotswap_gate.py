#!/usr/bin/env python3
"""Service-level Orin NX gate for v0.9.1 Base-512 residency swapping.

Each round overlaps a GDN request with TTS, then another GDN request with the
ASR transcription of the generated WAV. The gate fails on malformed audio,
empty/incorrect backend responses, missing overlap, a GDN restart, or an
unexpected final worker residency.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), dict(response.headers.items())


def post_wav(url: str, wav_bytes: bytes, timeout: float) -> bytes:
    boundary = f"----v091-gate-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="round.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def timed(call: Callable[[], Any]) -> dict[str, Any]:
    started = time.time()
    value = call()
    ended = time.time()
    return {"started": started, "ended": ended, "value": value}


def restart_count(container: str) -> int:
    output = subprocess.check_output(
        ["docker", "inspect", "--format={{.RestartCount}}", container],
        text=True,
    )
    return int(output.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--voice-url", default="http://127.0.0.1:8621")
    parser.add_argument("--gdn-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gdn-container", default="edge-llm-chat-service")
    parser.add_argument(
        "--output",
        default="/home/harvest/validation/v091-prod512-hotswap-gdn-10.json",
    )
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "rounds_requested": args.rounds,
        "gdn_restart_before": restart_count(args.gdn_container),
        "rounds": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tts_payload = {
        "text": "默认五百一十二长度的语音合成和大语言模型可以稳定共同运行。",
        "language": "chinese",
    }
    gdn_payload = {
        "model": "engines",
        "messages": [{"role": "user", "content": "只回答：并发正常"}],
        "max_tokens": 12,
        "temperature": 0,
    }

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for index in range(args.rounds):
                tts_future = pool.submit(
                    timed,
                    lambda: post_json(
                        f"{args.voice_url}/tts", tts_payload, args.timeout
                    ),
                )
                gdn_tts_future = pool.submit(
                    timed,
                    lambda: post_json(
                        f"{args.gdn_url}/v1/chat/completions",
                        gdn_payload,
                        args.timeout,
                    ),
                )
                tts_run = tts_future.result()
                gdn_tts_run = gdn_tts_future.result()
                wav_bytes, tts_headers = tts_run.pop("value")
                gdn_tts_body, _ = gdn_tts_run.pop("value")

                with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                    wav_info = {
                        "channels": wav.getnchannels(),
                        "sample_width": wav.getsampwidth(),
                        "sample_rate": wav.getframerate(),
                        "frames": wav.getnframes(),
                    }
                if (
                    wav_info["channels"],
                    wav_info["sample_width"],
                    wav_info["sample_rate"],
                ) != (1, 2, 24000):
                    raise RuntimeError(f"invalid WAV format: {wav_info}")

                asr_future = pool.submit(
                    timed,
                    lambda: post_wav(
                        f"{args.voice_url}/asr", wav_bytes, args.timeout
                    ),
                )
                gdn_asr_future = pool.submit(
                    timed,
                    lambda: post_json(
                        f"{args.gdn_url}/v1/chat/completions",
                        gdn_payload,
                        args.timeout,
                    ),
                )
                asr_run = asr_future.result()
                gdn_asr_run = gdn_asr_future.result()
                asr_body = asr_run.pop("value")
                gdn_asr_body, _ = gdn_asr_run.pop("value")

                asr_json = json.loads(asr_body)
                gdn_tts_json = json.loads(gdn_tts_body)
                gdn_asr_json = json.loads(gdn_asr_body)
                if asr_json.get("backend") != "trt_edgellm" or not asr_json.get("text"):
                    raise RuntimeError(f"invalid ASR response: {asr_json}")
                for phase, response in (
                    ("tts", gdn_tts_json),
                    ("asr", gdn_asr_json),
                ):
                    choices = response.get("choices") or []
                    content = (
                        choices[0].get("message", {}).get("content", "")
                        if choices
                        else ""
                    )
                    if not content:
                        raise RuntimeError(
                            f"empty GDN response during {phase}: {response}"
                        )

                tts_overlap = (
                    tts_run["started"] < gdn_tts_run["ended"]
                    and gdn_tts_run["started"] < tts_run["ended"]
                )
                asr_overlap = (
                    asr_run["started"] < gdn_asr_run["ended"]
                    and gdn_asr_run["started"] < asr_run["ended"]
                )
                if not tts_overlap or not asr_overlap:
                    raise RuntimeError(
                        f"missing request overlap: tts={tts_overlap} asr={asr_overlap}"
                    )
                current_restarts = restart_count(args.gdn_container)
                if current_restarts != result["gdn_restart_before"]:
                    raise RuntimeError(
                        f"GDN restarted: {result['gdn_restart_before']} -> "
                        f"{current_restarts}"
                    )

                result["rounds"].append(
                    {
                        "round": index + 1,
                        "wav_sha256": hashlib.sha256(wav_bytes).hexdigest(),
                        "wav": wav_info,
                        "tts_headers": tts_headers,
                        "asr_text": asr_json["text"],
                        "tts_overlap_gdn": tts_overlap,
                        "asr_overlap_gdn": asr_overlap,
                        "tts_seconds": tts_run["ended"] - tts_run["started"],
                        "asr_seconds": asr_run["ended"] - asr_run["started"],
                        "gdn_tts_seconds": (
                            gdn_tts_run["ended"] - gdn_tts_run["started"]
                        ),
                        "gdn_asr_seconds": (
                            gdn_asr_run["ended"] - gdn_asr_run["started"]
                        ),
                    }
                )

        docker_top = subprocess.check_output(
            ["docker", "top", "seeed-voice-v091"], text=True
        )
        if "qwen3_asr_worker" not in docker_top or "qwen3_tts_streaming_worker" in docker_top:
            raise RuntimeError(f"unexpected final residency:\n{docker_top}")
        result["gdn_restart_after"] = restart_count(args.gdn_container)
        result["final_residency"] = "asr_only"
        result["ok"] = True
    except BaseException as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["gdn_restart_after"] = restart_count(args.gdn_container)

    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
