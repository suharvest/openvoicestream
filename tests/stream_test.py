#!/usr/bin/env python3
"""Streaming ASR quality test — prints partials + final."""
import json, wave, io, sys, os
import numpy as np
import pytest

# Live-server integration harness, not a unit test: the ``test(...)`` helper
# below takes positional args and drives a running /asr/stream WebSocket on
# localhost:8621. Under pytest those args resolve as missing fixtures (collection
# error) and there is no server. Skip the whole module unless explicitly opted in
# via OVS_RUN_LIVE_ASR_TESTS=1 (intended to be run as a CLI script: __main__).
if os.environ.get("OVS_RUN_LIVE_ASR_TESTS") != "1":
    pytest.skip(
        "live-server ASR harness (set OVS_RUN_LIVE_ASR_TESTS=1 + run server to enable)",
        allow_module_level=True,
    )

import websocket

def test(path, label):
    with open(path, 'rb') as f:
        raw = f.read()
    with wave.open(io.BytesIO(raw)) as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        audio = wf.readframes(frames)
    dur = frames / sr
    samples = np.frombuffer(audio, dtype=np.int16)
    chunk_n = int(sr * 0.25)
    chunks = [samples[i:i+chunk_n].tobytes() for i in range(0, len(samples), chunk_n)]
    ws = websocket.create_connection(
        'ws://localhost:8621/asr/stream?language=Chinese&sample_rate=16000', timeout=30)
    partials = []
    ws.settimeout(0.05)
    for c in chunks:
        ws.send_binary(c)
        while True:
            try:
                msg = ws.recv()
                data = json.loads(msg)
                if data.get('type') == 'partial':
                    txt = data.get('text', '').strip()
                    if txt:
                        partials.append(txt)
            except:
                break
    ws.settimeout(None)
    ws.send_binary(b'')
    final = ''
    while True:
        data = json.loads(ws.recv())
        if data.get('type') == 'final':
            final = data.get('text', '').strip()
            break
    ws.close()
    print(f'{label} ({dur:.1f}s) partials={len(partials)} final={final}')
    seen = set()
    for p in partials:
        if p not in seen:
            seen.add(p)
            print(f'  [{p}]')
    repeats = sum(1 for i in range(1, len(partials))
                  if partials[i] == partials[i-1] and partials[i])
    if repeats:
        print(f'  REPEATS={repeats}')
    return final

if __name__ == '__main__':
    base = '/home/harvest/bench/wavs'
    for w, l in [('S0.wav', 'S0'), ('S1.wav', 'S1'), ('S3.wav', 'S3')]:
        test(f'{base}/{w}', l)
