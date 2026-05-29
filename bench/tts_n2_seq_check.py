#!/usr/bin/env python3
"""Control check: same worker (--max_slots=2) but SEQUENTIAL requests (wait for
each done before sending next). If sequential A/B transcribe correctly but the
concurrent run produced garbage, the bug is concurrency-specific.
Writes /tmp/tts_seq_{A,B}.wav.
"""
import subprocess, json, base64, struct, os, time, threading, sys

WORKER = "/home/harvest/project/v071-build/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_streaming_worker"
ENG = "/home/harvest/qwen3-tts-export-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-nx"
CWD = "/home/harvest/project/v071-build/TensorRT-Edge-LLM"
TEXTS = {"A": "今天天气真不错", "B": "人工智能正在改变世界"}

cmd = [WORKER, f"--talkerEngineDir={ENG}/talker", f"--code2wavEngineDir={ENG}/code2wav",
       f"--codePredictorEngineDir={ENG}/code_predictor", f"--tokenizerDir={ENG}/talker",
       "--max_slots=2"]
env = dict(os.environ)
if os.environ.get("NO_PRELOAD") != "1":
    env["QWEN3_TTS_PRELOAD_TALKER_EMBEDS"] = "/tmp/ref_talker_embeds_15row.bin"
else:
    env.pop("QWEN3_TTS_PRELOAD_TALKER_EMBEDS", None)
env["QWEN3_TTS_SEED"] = "42"
print("PRELOAD=" + env.get("QWEN3_TTS_PRELOAD_TALKER_EMBEDS", "<none>"), flush=True)
proc = subprocess.Popen(cmd, cwd=CWD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
pcm = {"A": bytearray(), "B": bytearray()}
done = {}
sr = [None]
ready = threading.Event()
lock = threading.Lock()

def reader():
    for line in proc.stdout:
        line = line.strip()
        if not line: continue
        try: ev = json.loads(line)
        except: continue
        et = ev.get("event")
        with lock:
            if et == "ready":
                print("READY", flush=True); ready.set()
            elif et == "chunk":
                cid = ev.get("id"); sr[0] = ev.get("sample_rate", sr[0])
                b64 = ev.get("audio_b64")
                if b64 and cid in pcm: pcm[cid] += base64.b64decode(b64)
            elif et == "done":
                done[ev.get("id")] = ev
                print("DONE", ev.get("id"), "frames", ev.get("total_frames"), flush=True)
            elif et == "error":
                print("ERROR", json.dumps(ev, ensure_ascii=False), flush=True)

threading.Thread(target=reader, daemon=True).start()
ready.wait(120); time.sleep(0.3)

for rid in ("A", "B"):
    proc.stdin.write(json.dumps({"id": rid, "text": TEXTS[rid], "speaker": "vivian",
        "stream": True, "chunk_format": "pcm_s16le", "chunk_transport": "base64"},
        ensure_ascii=False) + "\n")
    proc.stdin.flush()
    dl = time.time() + 90
    while time.time() < dl:
        with lock:
            if rid in done: break
        time.sleep(0.05)

time.sleep(0.5)
proc.stdin.close(); proc.terminate()
try: proc.wait(timeout=8)
except: proc.kill()

s = sr[0] or 24000
def wav(p, d):
    with open(p, "wb") as f:
        f.write(b"RIFF"); f.write(struct.pack("<I", 36+len(d))); f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<IHHIIHH",16,1,1,s,s*2,2,16))
        f.write(b"data"); f.write(struct.pack("<I", len(d))); f.write(d)
for rid in ("A","B"):
    wav(f"/tmp/tts_seq_{rid}.wav", bytes(pcm[rid]))
    print(f"WROTE /tmp/tts_seq_{rid}.wav bytes={len(pcm[rid])}", flush=True)
