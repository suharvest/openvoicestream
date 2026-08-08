#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
artifact=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091
engine="$artifact/engines/moss"
codec="$engine/codec"
compat="$validation/state/moss-compat-root"
worker="$validation/candidates/moss-ort123/moss_tts_nano_worker"
ort=/opt/onnxruntime-linux-aarch64-1.23.2
harness="$validation/scripts/moss_worker_n2_gate.py"
result="$validation/results/moss-direct-n1-n2-cancel-3.json"
restore_log="$validation/logs/moss-direct-restore.log"

test -x "$worker"
test -s "$harness"
test -s "$engine/tokenizer.model"
test -s "$codec/codec_decode_step.plan"
test -s "$codec/codec_browser_onnx_meta.json"
test ! -e "$compat"
mkdir -p "$compat"

for source in "$engine"/*; do
  if test -f "$source"; then
    ln -s "$source" "$compat/$(basename "$source")"
  fi
done
ln -s "$codec/codec_decode_step.plan" "$compat/codec_decode_step.plan"
ln -s "$codec/codec_browser_onnx_meta.json" "$compat/codec_browser_onnx_meta.json"

restore() {
  {
    docker start seeed-voice-v091
    docker start edge-llm-chat-service
    voice_code=000
    gdn_code=000
    for _ in $(seq 1 60); do
      voice_code=$(curl -sS -o /dev/null -w '%{http_code}' \
        --max-time 3 http://127.0.0.1:8621/readyz || true)
      gdn_code=$(curl -sS -o /dev/null -w '%{http_code}' \
        --max-time 3 http://127.0.0.1:8000/health || true)
      if test "$voice_code" = 200 && test "$gdn_code" = 200; then
        break
      fi
      sleep 2
    done
    translator_code=$(curl -sS -o /dev/null -w '%{http_code}' \
      --max-time 3 http://127.0.0.1:9001/health || true)
    printf 'voice=%s gdn=%s translator=%s\n' \
      "$voice_code" "$gdn_code" "$translator_code"
    test "$voice_code" = 200
    test "$gdn_code" = 200
    test "$translator_code" = 200
  } >"$restore_log" 2>&1
}
trap restore EXIT

test "$(curl -sS -o /dev/null -w '%{http_code}' \
  --max-time 3 http://127.0.0.1:9001/health)" = 200
docker stop seeed-voice-v091 edge-llm-chat-service

env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  python3 "$harness" \
  --worker "$worker" \
  --engine-dir "$compat" \
  --codec-onnx-dir "$codec" \
  --rounds 3 \
  --timeout 90 \
  --output "$result" \
  >"$validation/logs/moss-direct-n1-n2-cancel-3.stdout.log" \
  2>"$validation/logs/moss-direct-n1-n2-cancel-3.stderr.log"

test -s "$result"
python3 -m json.tool "$result" >"$result.pretty"
