#!/usr/bin/env bash
# ASR loopback for N=2 TTS outputs. Replicates the N=1 standalone qwen3_asr_worker
# recipe (from /tmp/asr_roundtrip.sh). Transcribes /tmp/tts_n2_{A,B}.wav.
set -uo pipefail
ASR=/home/harvest/project/asr-worker-build-verify/build_verify/workers/qwen3_asr_worker
PLUGIN=/home/harvest/seeed-local-voice-hotswap/deploy/jetson-workers/libNvInfer_edgellm_plugin_asr.so
ENGDIR=/home/harvest/qwen3-models/engines/orin-nx/highperf-v2/asr_thinker_full_fp8embed
MMDIR=/home/harvest/qwen3-models/engines/orin-nx/highperf/asr_audio_encoder
MEL_SET=/home/harvest/seeed-local-voice-hotswap/third_party/jetson-voice-engine/deploy/audio_preprocessing/whisper_feature_extractor.json
MEL_BIN=/home/harvest/seeed-local-voice-hotswap/third_party/jetson-voice-engine/deploy/audio_preprocessing/mel_filters.bin

export EDGELLM_PLUGIN_PATH=$PLUGIN

transcribe () {
  local WAV=$1
  local TAG=$2
  python3 - "$WAV" > /tmp/asr_in_$TAG.jsonl <<'PY'
import sys, wave, base64, json, audioop
wav=sys.argv[1]
w=wave.open(wav,"rb")
ch=w.getnchannels(); sw=w.getsampwidth(); sr=w.getframerate()
data=w.readframes(w.getnframes()); w.close()
if ch==2: data=audioop.tomono(data,sw,0.5,0.5)
if sw!=2: data=audioop.lin2lin(data,sw,2)
if sr!=16000:
    data,_=audioop.ratecv(data,2,1,sr,16000,None)
sec=len(data)/2/16000.0
b64=base64.b64encode(data).decode()
print(json.dumps({"event":"begin","id":"rt","sample_rate":16000,"chunk_size_sec":0.5,"unfixed_chunk_num":2,"unfixed_token_num":5}))
print(json.dumps({"event":"chunk","id":"rt","pcm_b64":b64,"audio_sec":sec,"last":True}))
print(json.dumps({"event":"end","id":"rt"}))
import sys as _s; print(f"INPUT {sys.argv[1]} audio_sec={sec:.2f}", file=_s.stderr)
PY
  "$ASR" --engineDir=$ENGDIR --multimodalEngineDir=$MMDIR \
    --melSettings=$MEL_SET --melFilters=$MEL_BIN \
    < /tmp/asr_in_$TAG.jsonl > /tmp/asr_out_$TAG.jsonl 2> /tmp/asr_err_$TAG.log
  echo "=== ASR $TAG ($WAV) EXIT $? ==="
  python3 -c '
import json,sys
for line in open(sys.argv[1]):
    line=line.strip()
    if not line.startswith("{"): continue
    try: j=json.loads(line)
    except: continue
    ev=j.get("event")
    if ev in ("final","error","prefill_failed"):
        print(ev.upper(), "->", repr(j.get("text", j)))
' /tmp/asr_out_$TAG.jsonl
}

transcribe /tmp/tts_n2_A.wav A
transcribe /tmp/tts_n2_B.wav B
