#!/usr/bin/env python3
"""N=2 concurrency + isolation verification for qwen3_asr_worker (SlotPool<AsrSlot>).

Drives ONE worker (--max_slots=2) with the streaming begin/chunk/end protocol
(pcm_b64 cumulative hops, last=true on final), which is the path that exercises
the slot pool (acquire/release/4429), unlike the one-shot `requests` path that
only ever touches slot 0.

Phases:
  1. begin two distinct sessions A (zh) + B (en) -> expect begin_ack on distinct
     slots (verified indirectly: both reach `final` with correct, isolated text).
  2. interleave their cumulative pcm_b64 hops -> each `final` text must match the
     one-shot baseline for THAT wav (no cross-talk).
  3. with A and B still active (slots full), begin a 3rd session C -> expect a
     pool_saturated / 4429 error (RISK POINT 2: acquireOrExisting returns -1 when
     no existing mapping + no free slot).
  4. end A and B, then begin C again -> must now succeed (slot freed; RISK POINT 3:
     session reset happened before the slot was marked free).
"""
import argparse, base64, json, os, subprocess, sys, threading, time, uuid, wave
import numpy as np

SAMPLE_RATE = 16000

def wav_to_audio(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr

def resample_16k(a, sr):
    if sr == SAMPLE_RATE:
        return a
    n = int(round(len(a) * SAMPLE_RATE / sr))
    return np.interp(np.linspace(0, 1, n, endpoint=False),
                     np.linspace(0, 1, len(a), endpoint=False), a).astype(np.float32)


def strip_lang(text):
    if not text or not text.startswith("language "):
        return text
    for lang in ("Chinese", "English", "Cantonese", "Japanese", "Korean", "French",
                 "German", "Italian", "Portuguese", "Russian", "Spanish"):
        p = "language " + lang
        if text.startswith(p):
            return text[len(p):].lstrip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--multimodal-engine-dir", required=True)
    ap.add_argument("--mel-settings", required=True)
    ap.add_argument("--mel-filters", required=True)
    ap.add_argument("--wav-a", required=True)
    ap.add_argument("--wav-b", required=True)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    args = ap.parse_args()

    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = args.plugin
    env.setdefault("EDGE_LLM_ASR_CUDA_GRAPH", "0")
    cmd = [args.worker, "--engineDir", args.engine_dir,
           "--multimodalEngineDir", args.multimodal_engine_dir,
           "--melSettings", args.mel_settings, "--melFilters", args.mel_filters,
           "--max_slots", "2"]
    print("CMD:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

    events = []           # all parsed events
    finals = {}           # id -> final text
    saturations = []      # raw 4429 events
    begin_acks = {}       # id -> True
    errors = []           # non-saturation errors
    lock = threading.Lock()
    ready_evt = threading.Event()

    def reader():
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                print("NONJSON:", line[:200], flush=True)
                continue
            et = ev.get("event")
            with lock:
                events.append(ev)
                if et == "ready":
                    print("READY:", json.dumps(ev, ensure_ascii=False), flush=True)
                    ready_evt.set()
                elif et == "begin_ack":
                    begin_acks[ev.get("id")] = True
                elif et == "final":
                    cid = ev.get("id")
                    finals[cid] = strip_lang(ev.get("text", ""))
                    print(f"FINAL id={cid}: {finals[cid]!r}", flush=True)
                elif et == "error":
                    if ev.get("status") == 4429 or ev.get("error") == "pool_saturated":
                        saturations.append(ev)
                        print("SATURATED RAW:", json.dumps(ev, ensure_ascii=False), flush=True)
                    else:
                        errors.append(ev)
                        print("ERROR:", json.dumps(ev, ensure_ascii=False), flush=True)

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()

    def send(obj):
        proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    if not ready_evt.wait(timeout=180):
        print("FATAL: worker never emitted ready", flush=True)
        print(proc.stderr.read()[-2000:], flush=True)
        proc.kill(); sys.exit(1)

    aud_a = resample_16k(*wav_to_audio(args.wav_a))
    aud_b = resample_16k(*wav_to_audio(args.wav_b))
    hop = int(args.hop_sec * SAMPLE_RATE)

    id_a, id_b, id_c = "sessA", "sessB", "sessC"

    # ---- PHASE 1: begin both ----
    print("\n=== PHASE 1: begin A(zh) + B(en) ===", flush=True)
    send({"event": "begin", "id": id_a, "sample_rate": SAMPLE_RATE,
          "chunk_size_sec": args.hop_sec, "audio_format": "pcm"})
    send({"event": "begin", "id": id_b, "sample_rate": SAMPLE_RATE,
          "chunk_size_sec": args.hop_sec, "audio_format": "pcm"})
    time.sleep(0.3)

    # ---- PHASE 3 (while both active = slots full): begin C -> expect 4429 ----
    print("\n=== PHASE 3: begin C while A+B active -> expect 4429 ===", flush=True)
    sat_before = len(saturations)
    send({"event": "begin", "id": id_c, "sample_rate": SAMPLE_RATE,
          "chunk_size_sec": args.hop_sec, "audio_format": "pcm"})
    time.sleep(0.5)

    # ---- PHASE 2: interleave cumulative hops for A and B ----
    print("\n=== PHASE 2: interleave A + B cumulative pcm hops ===", flush=True)
    n_a = max(1, (len(aud_a) + hop - 1) // hop)
    n_b = max(1, (len(aud_b) + hop - 1) // hop)
    n_hops = max(n_a, n_b)
    for k in range(n_hops):
        for cid, aud, nh in ((id_a, aud_a, n_a), (id_b, aud_b, n_b)):
            if k >= nh:
                continue
            end = min((k + 1) * hop, len(aud))
            sl = aud[:end].astype(np.float32)
            b64 = base64.b64encode(sl.tobytes()).decode("ascii")
            is_last = (k == nh - 1)
            send({"event": "chunk", "id": cid, "pcm_b64": b64,
                  "audio_sec": end / SAMPLE_RATE, "last": is_last})
        time.sleep(0.02)

    # wait for both finals
    deadline = time.time() + 120
    while time.time() < deadline:
        with lock:
            if id_a in finals and id_b in finals:
                break
        if proc.poll() is not None:
            break
        time.sleep(0.05)

    # ---- PHASE 4: A+B done (slots freed) -> begin C again, expect success ----
    print("\n=== PHASE 4: begin C again after A+B freed -> expect begin_ack ===", flush=True)
    send({"event": "begin", "id": id_c, "sample_rate": SAMPLE_RATE,
          "chunk_size_sec": args.hop_sec, "audio_format": "pcm"})
    time.sleep(0.5)
    with lock:
        c_acked = begin_acks.get(id_c, False)
    send({"event": "end", "id": id_c})
    time.sleep(0.3)

    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.terminate()
    rc = proc.returncode
    err = proc.stderr.read()

    print("\n========== SUMMARY ==========", flush=True)
    print(f"BEGIN_ACKS={sorted(begin_acks.keys())}", flush=True)
    print(f"FINAL_A({id_a})={finals.get(id_a)!r}", flush=True)
    print(f"FINAL_B({id_b})={finals.get(id_b)!r}", flush=True)
    print(f"SATURATIONS_DURING_FULL={len(saturations) - sat_before}", flush=True)
    for ev in saturations:
        print("  SAT_RAW:", json.dumps(ev, ensure_ascii=False), flush=True)
    print(f"C_REACQUIRED_AFTER_FREE={c_acked}", flush=True)
    print(f"NON_SAT_ERRORS={len(errors)}", flush=True)
    for ev in errors:
        print("  ERR_RAW:", json.dumps(ev, ensure_ascii=False), flush=True)
    print(f"CRASHED={rc not in (0, None) and rc != -15}  returncode={rc}", flush=True)
    if err:
        print("\n=== STDERR TAIL ===", flush=True)
        print(err[-1500:], flush=True)


if __name__ == "__main__":
    main()
