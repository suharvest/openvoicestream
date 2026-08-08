#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
artifact=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091
worker="$artifact/bin/spark_tts_worker"
llm_root="$artifact/engines"
engine_root=/home/harvest/project/v090-engines/sparktts
plugin="$artifact/libNvInfer_edgellm_plugin.so"
profile="$engine_root/voice_profile.json"
harness="$validation/scripts/spark_v091_gate.py"
output="$validation/results/spark-direct"
restore_log="$validation/logs/spark-direct-restore.log"

test -x "$worker"
test -s "$plugin"
test -s "$harness"
test -s "$llm_root/sparktts-w4a16/llm.engine"
test -s "$llm_root/sparktts-bf16/llm.engine"
test -s "$engine_root/sparktts_speaker_decoder.fp32.engine"
test -s "$engine_root/bicodec_decoder_dynT.fp16.engine"
test -s "$profile"
mkdir -p "$output"

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

run_phase() {
  variant=$1
  phase=$2
  log_name=${variant}-${phase}
  env \
    SPARK_WORKER="$worker" \
    SPARK_LLM_ROOT="$llm_root" \
    SPARK_ENGINE_ROOT="$engine_root" \
    SPARK_OUTPUT_ROOT="$output" \
    SPARK_VOICE_PROFILE="$profile" \
    EDGELLM_PLUGIN_PATH="$plugin" \
    SPARK_STRESS_ROUNDS=3 \
    python3 "$harness" "$variant" "$phase" \
    >"$validation/logs/spark-$log_name.stdout.log" \
    2>"$validation/logs/spark-$log_name.stderr.log"
}

run_phase sparktts-w4a16 basic
run_phase sparktts-w4a16 n2
run_phase sparktts-w4a16 clone
run_phase sparktts-w4a16 stress
run_phase sparktts-bf16 basic
run_phase sparktts-bf16 n2

sha256sum \
  "$worker" \
  "$plugin" \
  "$llm_root/sparktts-w4a16/llm.engine" \
  "$llm_root/sparktts-bf16/llm.engine" \
  "$engine_root/sparktts_speaker_decoder.fp32.engine" \
  "$engine_root/bicodec_decoder_dynT.fp16.engine" \
  "$profile" \
  >"$output/SHA256SUMS"
