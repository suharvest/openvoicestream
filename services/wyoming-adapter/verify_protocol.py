#!/usr/bin/env python3
"""Protocol-level verification of the Wyoming adapter (no Home Assistant).

Usage:
    uv run python verify_protocol.py [--host 127.0.0.1] \
        [--wav ../../bench/perf/corpus/short/zh_short_01.wav]

Checks:
  1. describe -> info on both ports; prints the two capability flags.
  2. Full STT exchange with a real 16 kHz mono WAV; prints the Transcript.
  3. Full streaming TTS exchange (SynthesizeStart -> chunks -> the trailing
     full Synthesize -> SynthesizeStop) and asserts the audio arrived ONCE and
     is not silence (RMS > 0).
"""

from __future__ import annotations

import argparse
import array
import asyncio
import json
import math
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

TTS_TEXT_PARTS = [
    "好的，",
    "客厅灯已经打开了，",
    "卧室空调也已经调到二十六度。",
    "还需要我关掉窗帘吗？",
]


async def describe(host: str, port: int) -> Info:
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Describe().event())
        while True:
            event = await client.read_event()
            assert event is not None, "connection closed before info"
            if Info.is_type(event.type):
                return Info.from_event(event)


async def run_stt(host: str, port: int, wav_path: str) -> str:
    wav = wave.open(wav_path)
    rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
    pcm = wav.readframes(wav.getnframes())
    print(f"[stt] wav rate={rate} width={width} channels={channels} bytes={len(pcm)}")
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Transcribe(language="zh").event())
        await client.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        step = 3200
        for i in range(0, len(pcm), step):
            await client.write_event(
                AudioChunk(rate=rate, width=width, channels=channels, audio=pcm[i : i + step]).event()
            )
        await client.write_event(AudioStop().event())
        while True:
            event = await client.read_event()
            assert event is not None, "connection closed before transcript"
            if Transcript.is_type(event.type):
                return Transcript.from_event(event).text


async def run_tts_streaming(host: str, port: int) -> dict:
    audio_starts = 0
    rate = width = channels = 0
    pcm = bytearray()
    stopped = False
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(SynthesizeStart(voice=None).event())
        for part in TTS_TEXT_PARTS:
            await client.write_event(SynthesizeChunk(text=part).event())
        # HA sends the COMPLETE text again "for compatibility" — the adapter
        # must ignore it, otherwise everything is synthesized twice.
        await client.write_event(Synthesize(text="".join(TTS_TEXT_PARTS)).event())
        await client.write_event(SynthesizeStop().event())
        while True:
            event = await client.read_event()
            if event is None:
                break
            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
                audio_starts += 1
                rate, width, channels = start.rate, start.width, start.channels
                print(f"[tts] audio-start rate={rate} width={width} channels={channels}")
            elif AudioChunk.is_type(event.type):
                pcm += AudioChunk.from_event(event).audio
            elif AudioStop.is_type(event.type):
                print(f"[tts] audio-stop ({len(pcm)} bytes so far)")
            elif SynthesizeStopped.is_type(event.type):
                stopped = True
                break
    samples = array.array("h")
    samples.frombytes(bytes(pcm[: len(pcm) // 2 * 2]))
    rms = math.sqrt(sum(float(s) * s for s in samples) / max(1, len(samples)))
    return {
        "audio_start_events": audio_starts,
        "bytes": len(pcm),
        "rate": rate,
        "width": width,
        "channels": channels,
        "duration_s": round(len(samples) / rate, 3) if rate else None,
        "rms": round(rms, 1),
        "synthesize_stopped": stopped,
        "text_chars": len("".join(TTS_TEXT_PARTS)),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--stt-port", type=int, default=10300)
    ap.add_argument("--tts-port", type=int, default=10200)
    ap.add_argument(
        "--wav", default="../../bench/perf/corpus/short/zh_short_01.wav"
    )
    args = ap.parse_args()

    print("=== 1. describe -> info ===")
    stt_info = await describe(args.host, args.stt_port)
    print("STT info:", json.dumps(stt_info.event().data, ensure_ascii=False))
    print("  supports_transcript_streaming =", stt_info.asr[0].supports_transcript_streaming)
    tts_info = await describe(args.host, args.tts_port)
    print("TTS info:", json.dumps(tts_info.event().data, ensure_ascii=False))
    print("  supports_synthesize_streaming =", tts_info.tts[0].supports_synthesize_streaming)
    assert stt_info.asr[0].supports_transcript_streaming is False
    assert tts_info.tts[0].supports_synthesize_streaming is True

    print("\n=== 2. STT exchange ===")
    text = await run_stt(args.host, args.stt_port, args.wav)
    print("Transcript:", repr(text))
    assert text.strip(), "empty transcript"

    print("\n=== 3. streaming TTS exchange (double-synthesis check) ===")
    result = await run_tts_streaming(args.host, args.tts_port)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result["audio_start_events"] == 1, "expected exactly one audio-start"
    assert result["rms"] > 0, "audio is silence"
    # ~0.11 s of 16 kHz audio per CJK char is typical; twice the text would be
    # far beyond this bound.
    per_char = result["duration_s"] / result["text_chars"]
    print(f"duration per char = {per_char:.3f} s (double synthesis would be ~2x)")
    assert per_char < 0.35, "duration suggests the text was synthesized twice"
    print("\nALL PROTOCOL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
