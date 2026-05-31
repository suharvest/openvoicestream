#!/usr/bin/env python3
"""Quick test CHUNK_SIZE=0.4 tuning: stream 3 wavs, show partial+final."""
import json, sys, wave, io, time, os
import numpy as np
import pytest

# Live-server tuning harness, not a unit test: ``test_wav(path, label)`` drives a
# running /asr/stream WebSocket. Skip under pytest unless explicitly opted in.
if os.environ.get("OVS_RUN_LIVE_ASR_TESTS") != "1":
    pytest.skip(
        "live-server ASR tuning harness (set OVS_RUN_LIVE_ASR_TESTS=1 to enable)",
        allow_module_level=True,
    )

import websocket

WS_URL = "ws://localhost:8621"

def test_wav(path, label):
    with open(path, "rb") as f:
        raw = f.read()
    with wave.open(io.BytesIO(raw)) as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        audio = wf.readframes(frames)
    dur = frames / sr
    samples = np.frombuffer(audio, dtype=np.int16)
    chunk_n = int(sr * 0.25)  # always send in 250ms chunks (client side)
    chunks = [samples[i:i+chunk_n].tobytes() for i in range(0, len(samples), chunk_n)]

    ws = websocket.create_connection(f"{WS_URL}/asr/stream?language=Chinese&sample_rate={sr}", timeout=60)

    partials = []
    for c in chunks:
        ws.send_binary(c)
        try:
            msg = ws.recv()
            data = json.loads(msg)
            if data.get("type") == "partial":
                partials.append(data.get("text", "").strip())
        except:
            pass

    ws.send_binary(b"")  # EOS
    final_text = ""
    while True:
        data = json.loads(ws.recv())
        if data.get("type") == "final":
            final_text = data.get("text", "").strip()
            break
    ws.close()

    print(f"\n=== {label} ({dur:.1f}s) ===")
    print(f"Partial emissions: {len(partials)}")
    # Show unique partial texts (skip empty)
    seen = set()
    for p in partials:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            print(f"  partial: {p}")
    print(f"  FINAL: {final_text}")

    # Check for repeats (bad)
    repeats = 0
    prev = ""
    for p in partials:
        p = p.strip()
        if prev and p and p.startswith(prev) and p != prev:
            pass  # progressive refinement is good
        elif prev and p == prev and p:
            repeats += 1  # exact repeat
        prev = p
    if repeats:
        print(f"  ⚠ Exact repeats: {repeats}")
    else:
        print(f"  ✓ No exact repeats")
    return final_text

if __name__ == "__main__":
    base = "/home/harvest/bench/wavs"
    finals = {}
    for wav, label in [("S0.wav", "S0"), ("S1.wav", "S1"), ("S3.wav", "S3")]:
        finals[label] = test_wav(f"{base}/{wav}", label)

    print("\n\n=== Cross-check: FINAL vs OFFLINE ===")
    for label, text in finals.items():
        print(f"{label}: {text}")
