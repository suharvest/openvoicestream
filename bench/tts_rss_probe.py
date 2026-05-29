#!/usr/bin/env python3
"""Measure worker peak RSS for a given --max_slots. Launches worker, sends one
request per slot concurrently, samples /proc/<pid>/status VmRSS at high rate,
reports peak. SLOTS env controls slot count + number of concurrent requests.
"""
import subprocess, json, os, time, threading, sys

WORKER = "/home/harvest/project/v071-build/TensorRT-Edge-LLM/build/examples/omni/qwen3_tts_streaming_worker"
ENG = "/home/harvest/qwen3-tts-export-workspace/Qwen3-TTS-12Hz-0.6B-CustomVoice/engines-nx"
CWD = "/home/harvest/project/v071-build/TensorRT-Edge-LLM"
SLOTS = int(os.environ.get("SLOTS", "1"))
TEXT = "今天天气真不错"

cmd = [WORKER, f"--talkerEngineDir={ENG}/talker", f"--code2wavEngineDir={ENG}/code2wav",
       f"--codePredictorEngineDir={ENG}/code_predictor", f"--tokenizerDir={ENG}/talker",
       f"--max_slots={SLOTS}"]
env = dict(os.environ)
env["QWEN3_TTS_PRELOAD_TALKER_EMBEDS"] = "/tmp/ref_talker_embeds_15row.bin"
env["QWEN3_TTS_SEED"] = "42"

proc = subprocess.Popen(cmd, cwd=CWD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
pid = proc.pid
ready = threading.Event(); done_n = [0]; lock = threading.Lock()
peak = [0]

def vmrss():
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmRSS:"):
                return int(l.split()[1])  # kB
    except Exception:
        return 0
    return 0

def sampler():
    while proc.poll() is None:
        r = vmrss()
        if r > peak[0]: peak[0] = r
        time.sleep(0.05)

def reader():
    for line in proc.stdout:
        line = line.strip()
        if not line: continue
        try: e = json.loads(line)
        except: continue
        t = e.get("event")
        if t == "ready": ready.set()
        elif t in ("done", "error"):
            with lock: done_n[0] += 1

threading.Thread(target=sampler, daemon=True).start()
threading.Thread(target=reader, daemon=True).start()
ready.wait(120); time.sleep(0.2)
rss_after_init = vmrss()
for i in range(SLOTS):
    proc.stdin.write(json.dumps({"id": f"r{i}", "text": TEXT, "speaker": "vivian",
        "stream": True, "chunk_format": "pcm_s16le", "chunk_transport": "base64"},
        ensure_ascii=False) + "\n")
proc.stdin.flush()
dl = time.time() + 90
while time.time() < dl:
    with lock:
        if done_n[0] >= SLOTS: break
    time.sleep(0.05)
time.sleep(0.5)
print(f"SLOTS={SLOTS} RSS_after_init_MB={rss_after_init/1024:.1f} PEAK_RSS_MB={peak[0]/1024:.1f}", flush=True)
proc.stdin.close(); proc.terminate()
try: proc.wait(timeout=8)
except: proc.kill()
