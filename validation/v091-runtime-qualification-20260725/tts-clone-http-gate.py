#!/usr/bin/env python3
"""HTTP /tts/clone smoke for prompt-audio clone backends such as MOSS."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import wave
from io import BytesIO
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--reference-wav", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    response = requests.post(
        args.base_url.rstrip("/") + "/tts/clone",
        json={
            "text": "克隆语音服务必须返回完整且非静音的音频。",
            "language": "chinese",
            "speaker_embedding_b64": base64.b64encode(
                args.reference_wav.read_bytes()
            ).decode("ascii"),
        },
        timeout=120,
    )
    payload = response.content
    sample_rate = None
    frames = b""
    channels = None
    if response.status_code == 200:
        try:
            with wave.open(BytesIO(payload), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError):
            pass
    result = {
        "status": response.status_code,
        "content_type": response.headers.get("content-type"),
        "sample_rate": sample_rate,
        "channels": channels,
        "pcm_bytes": len(frames),
        "sha256": hashlib.sha256(frames).hexdigest(),
    }
    result["passed"] = bool(
        response.status_code == 200
        and sample_rate == args.sample_rate
        and len(frames) > 0
    )
    if not result["passed"]:
        result["error_body"] = response.text[:1000]
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
