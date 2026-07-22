#!/usr/bin/env python3
"""Minimal /v2v/stream TTS-only WebSocket client.

Install dependency:
    uv run --with websockets python examples/v2v_tts_only.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import websockets


def write_wav(path: Path, sample_rate: int, pcm_parts: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(pcm_parts))


async def run(args: argparse.Namespace) -> None:
    sample_rate: int | None = None
    pcm_parts: list[bytes] = []
    async with websockets.connect(
        args.url,
        open_timeout=args.timeout_sec,
        subprotocols=["seeed.realtime.v2"],
    ) as ws:
        created = json.loads(await ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"expected session.created, got {created}")
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["audio"],
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm", "rate": args.sample_rate,
                                    "channels": 1, "endianness": "little",
                                },
                                "turn_detection": {"type": "none"},
                            },
                            "output": {
                                "format": {
                                    "type": "audio/pcm", "rate": args.sample_rate,
                                    "channels": 1, "endianness": "little",
                                },
                                "language": args.language,
                            },
                        },
                    },
                }
            )
        )
        updated = json.loads(await ws.recv())
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"expected session.updated, got {updated}")
        sample_rate = int(updated["session"]["audio"]["output"]["format"]["rate"])
        await ws.send(json.dumps({
            "type": "x_v2v.response.speak",
            "speech": {"text": args.text, "conversation": "none"},
        }, ensure_ascii=False))

        async for msg in ws:
            if isinstance(msg, bytes):
                pcm_parts.append(msg)
                continue
            event = json.loads(msg)
            if event.get("type") == "error":
                raise RuntimeError(event.get("error") or event)
            if event.get("type") == "response.done":
                break

    if sample_rate is None:
        raise RuntimeError("session.updated returned no output sample rate")
    pcm = b"".join(pcm_parts)
    if len(pcm) < 1000:
        raise RuntimeError(f"TTS stream returned too little audio: {len(pcm)} bytes")
    write_wav(args.out, sample_rate, pcm_parts)
    duration = len(pcm) / 2 / sample_rate
    print(f"Wrote {args.out} ({sample_rate} Hz, {duration:.2f} s, {len(pcm)} PCM bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8621/v2v/stream")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", type=Path, default=Path("v2v-tts.wav"))
    parser.add_argument("--language", default="zh")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--timeout-sec", type=float, default=30)
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
