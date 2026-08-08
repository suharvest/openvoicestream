#!/usr/bin/env bash
set -euo pipefail

validation_root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725

for container in seeed-voice-v091 edge-llm-chat-service translator; do
  printf '==== %s\n' "$container"
  cat "$validation_root/snapshot/${container}.identity-health.txt"
  cat "$validation_root/snapshot/${container}.mounts.txt"
  grep -E \
    '^(OVS_PROFILE|EDGELLM_|EDGE_LLM_|MOSS_|SPARK_|ASR_|TTS_|LANGUAGE_MODE|QWEN3_)=' \
    "$validation_root/snapshot/${container}.env.redacted.txt" || true
done

for url in \
  http://127.0.0.1:8000/health \
  http://127.0.0.1:8000/readyz \
  http://127.0.0.1:8000/v1/models \
  http://127.0.0.1:9001/health \
  http://127.0.0.1:9001/readyz; do
  body=$(mktemp)
  code=$(curl -sS -o "$body" -w '%{http_code}' "$url" || true)
  printf '%s=%s\n' "$url" "$code"
  head -c 500 "$body"
  printf '\n'
  rm -f "$body"
done
