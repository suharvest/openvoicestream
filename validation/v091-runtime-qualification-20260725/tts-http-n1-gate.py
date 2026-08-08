#!/usr/bin/env python3
"""Single-lane HTTP chunked TTS smoke with configurable sample rate."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--text", default="单路语音服务输出必须完整且非空。")
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    first_audio = None
    chunks: list[bytes] = []
    with requests.post(
        args.base_url.rstrip("/") + "/tts/stream",
        json={"text": args.text, "language": args.language},
        stream=True,
        timeout=90,
    ) as response:
        status = response.status_code
        content_type = response.headers.get("content-type")
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunks.append(chunk)
            if first_audio is None and sum(map(len, chunks)) > 4:
                first_audio = time.perf_counter()
    ended = time.perf_counter()
    payload = b"".join(chunks)
    sample_rate = (
        struct.unpack("<I", payload[:4])[0]
        if status == 200 and len(payload) >= 4
        else None
    )
    pcm = payload[4:] if len(payload) >= 4 else b""
    result = {
        "status": status,
        "content_type": content_type,
        "sample_rate": sample_rate,
        "pcm_bytes": len(pcm),
        "sha256": hashlib.sha256(pcm).hexdigest(),
        "ttfa_ms": (
            (first_audio - started) * 1000 if first_audio is not None else None
        ),
        "wall_ms": (ended - started) * 1000,
    }
    result["passed"] = bool(
        status == 200
        and sample_rate == args.sample_rate
        and len(pcm) > 0
        and first_audio is not None
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
