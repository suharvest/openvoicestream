#!/usr/bin/env python3
"""Through-service CV TTS smoke: POST /tts to the running container, save WAVs.
Prints per-case HTTP status, content-type, byte length, and saved path as JSON."""
import json, sys, urllib.request, urllib.error, os

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18621"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cv_baked_smoke"
os.makedirs(OUT, exist_ok=True)

CASES = [
    {"id": "zh",   "body": {"text": "你好，很高兴见到你。", "language": "chinese", "speaker_id": 3066}},
    {"id": "en",   "body": {"text": "Hello, nice to meet you.", "language": "english", "speaker_id": 3066}},
    {"id": "base", "body": {"text": "你好，很高兴见到你。", "speaker_id": 3066}},  # no language -> Base path
]

results = {}
for c in CASES:
    rid = c["id"]
    data = json.dumps(c["body"]).encode()
    req = urllib.request.Request(f"{BASE}/tts", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    rec = {"request": c["body"]}
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            body = r.read()
            rec["http_status"] = r.status
            rec["content_type"] = r.headers.get("Content-Type")
            rec["bytes"] = len(body)
            if r.status == 200 and body[:4] == b"RIFF":
                p = os.path.join(OUT, f"{rid}.wav")
                with open(p, "wb") as f:
                    f.write(body)
                rec["wav"] = p
            else:
                rec["body_head"] = body[:300].decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        rec["http_status"] = e.code
        rec["error_body"] = e.read()[:500].decode("utf-8", "replace")
    except Exception as e:
        rec["exception"] = repr(e)
    results[rid] = rec
    print(f"[{rid}] status={rec.get('http_status')} ct={rec.get('content_type')} bytes={rec.get('bytes')}", flush=True)

print(json.dumps(results, indent=2, ensure_ascii=False))
