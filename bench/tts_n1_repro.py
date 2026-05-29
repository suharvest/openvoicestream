#!/usr/bin/env python3
"""Exact N=1 reproduction: single request, preload override, text=reference.
Used to confirm the worker produces correct audio for the matching reference
text. NO_PRELOAD=1 to disable override. TEXT env overrides text.
"""
import subprocess, json, base64, struct, os, time, threading, sys

WORKER = "/home/harvest/project/v071-build/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_streaming_worker"
ENG = "/home/harvest/qwen3-tts-export-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-nx"
CWD = "/home/harvest/project/v071-build/TensorRT-Edge-LLM"
TEXT = os.environ.get("TEXT", "今天天气真不错")
OUT = os.environ.get("OUT", "/tmp/n1_repro.wav")
SLOTS = os.environ.get("SLOTS", "1")

cmd = [WORKER, f"--talkerEngineDir={ENG}/talker", f"--code2wavEngineDir={ENG}/code2wav",
       f"--codePredictorEngineDir={ENG}/code_predictor", f"--tokenizerDir={ENG}/talker",
       f"--max_slots={SLOTS}"]
env = dict(os.environ)
if os.environ.get("NO_PRELOAD") != "1":
    env["QWEN3_TTS_PRELOAD_TALKER_EMBEDS"] = "/tmp/ref_talker_embeds_15row.bin"
else:
    env.pop("QWEN3_TTS_PRELOAD_TALKER_EMBEDS", None)
env["QWEN3_TTS_SEED"] = "42"
print("PRELOAD=" + env.get("QWEN3_TTS_PRELOAD_TALKER_EMBEDS", "<none>"), "TEXT=" + TEXT, flush=True)

proc = subprocess.Popen(cmd, cwd=CWD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
pcm = bytearray(); sr = [24000]; done = threading.Event(); ready = threading.Event()

def reader():
    for line in proc.stdout:
        line = line.strip()
        if not line: continue
        try: e = json.loads(line)
        except: continue
        t = e.get("event")
        if t == "ready": ready.set()
        elif t == "chunk":
            sr[0] = e.get("sample_rate", sr[0]); b = e.get("audio_b64")
            if b: pcm.extend(base64.b64decode(b))
        elif t == "done":
            print("DONE frames", e.get("total_frames"), "elapsed_ms", e.get("elapsed_ms"), flush=True); done.set()
        elif t == "error":
            print("ERROR", json.dumps(e, ensure_ascii=False), flush=True); done.set()

threading.Thread(target=reader, daemon=True).start()
ready.wait(120); time.sleep(0.2)
proc.stdin.write(json.dumps({"id": "r1", "text": TEXT, "speaker": "vivian", "stream": True,
    "chunk_format": "pcm_s16le", "chunk_transport": "base64"}, ensure_ascii=False) + "\n")
proc.stdin.flush()
done.wait(90); time.sleep(0.3)
proc.stdin.close(); proc.terminate()
try: proc.wait(timeout=8)
except: proc.kill()
s = sr[0]
with open(OUT, "wb") as f:
    f.write(b"RIFF"); f.write(struct.pack("<I", 36+len(pcm))); f.write(b"WAVE")
    f.write(b"fmt "); f.write(struct.pack("<IHHIIHH",16,1,1,s,s*2,2,16))
    f.write(b"data"); f.write(struct.pack("<I", len(pcm))); f.write(bytes(pcm))
print(f"WROTE {OUT} bytes={len(pcm)} dur={len(pcm)/(s*2):.2f}s sr={s}", flush=True)
err = proc.stderr.read()
import re
ov = len(re.findall("Overrode talker", err))
print("OVERRIDE_APPLIED_COUNT=", ov, flush=True)
print("STDERR_TAIL:", err[-600:], flush=True)
