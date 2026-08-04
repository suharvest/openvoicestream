#!/usr/bin/env bash
set -euo pipefail

url="${1:-http://127.0.0.1:8000}"
timeout_seconds="${LLM_VERIFY_TIMEOUT:-900}"
deadline=$((SECONDS + timeout_seconds))

echo "Waiting for LLM at ${url} (timeout ${timeout_seconds}s)"
until curl -fsS "${url}/v1/models" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: LLM did not become ready within ${timeout_seconds}s" >&2
    exit 1
  fi
  sleep 5
done

response="$(curl -fsS \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"Reply with exactly: migration-ok"}],"max_tokens":16,"temperature":0}' \
  "${url}/v1/chat/completions")"

python3 -c '
import json, sys
payload = json.load(sys.stdin)
choices = payload.get("choices") or []
if not choices or not (choices[0].get("message") or {}).get("content", "").strip():
    raise SystemExit("LLM smoke returned no assistant content")
print("LLM smoke: PASS")
' <<<"${response}"
