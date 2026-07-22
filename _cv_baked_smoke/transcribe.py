#!/usr/bin/env python3
"""Independent intelligibility check: faster-whisper roundtrip over saved WAVs.
Run via: uv run --with faster-whisper transcribe.py <dir>"""
import sys, os, json
from faster_whisper import WhisperModel

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cv_baked_smoke"
# small model is enough for short clean clips; CPU int8 keeps it light on Jetson host
model = WhisperModel("small", device="cpu", compute_type="int8")

LANGS = {"zh": "zh", "en": "en", "base": "zh"}
out = {}
for rid, lang in LANGS.items():
    p = os.path.join(d, f"{rid}.wav")
    if not os.path.exists(p):
        out[rid] = {"error": "no wav"}
        continue
    segs, info = model.transcribe(p, language=lang, beam_size=5)
    text = "".join(s.text for s in segs).strip()
    out[rid] = {"lang": info.language, "duration": round(info.duration, 2), "text": text}
    print(f"[{rid}] ({lang}) -> {text!r}", flush=True)

print(json.dumps(out, indent=2, ensure_ascii=False))
