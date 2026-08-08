#!/usr/bin/env bash
set -euo pipefail

VOICE=seeed-voice-v091-r2-canary-base-n1-021112e
GDN=edge-llm-chat-service
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="$VALIDATION/results/r2-product-n1-co-residency"
INPUTS="$VALIDATION/r2-base-n1-inputs"
WAV="$VALIDATION/corpus/short/zh_short_01.wav"
TEGRA_PID=

cleanup() {
  if [[ -n "$TEGRA_PID" ]]; then
    kill "$TEGRA_PID" >/dev/null 2>&1 || true
    wait "$TEGRA_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

mkdir -p "$EVIDENCE"
[[ "$(docker inspect -f '{{.State.Health.Status}}' "$VOICE")" == healthy ]]
[[ "$(docker inspect -f '{{.State.Health.Status}}' "$GDN")" == healthy ]]
curl -fsS http://127.0.0.1:18621/readyz > "$EVIDENCE/voice-ready-before.json"
curl -fsS http://127.0.0.1:8000/health > "$EVIDENCE/gdn-health-before.json"
docker inspect -f '{{.RestartCount}}' "$VOICE" > "$EVIDENCE/voice-restarts-before.txt"
docker inspect -f '{{.RestartCount}}' "$GDN" > "$EVIDENCE/gdn-restarts-before.txt"
docker stats --no-stream "$VOICE" "$GDN" > "$EVIDENCE/docker-stats-before.txt"
start_epoch="$(date +%s)"

tegrastats --interval 500 --logfile "$EVIDENCE/tegrastats-during.log" &
TEGRA_PID=$!
python3 "$INPUTS/product-n1-co-residency-gate.py" \
  --voice-url http://127.0.0.1:18621 \
  --gdn-url http://127.0.0.1:8000 \
  --wav "$WAV" \
  --sequential-rounds 10 \
  --pairwise-rounds 3 \
  --output "$EVIDENCE/product-n1-co-residency.json" \
  | tee "$EVIDENCE/product-n1-co-residency.stdout.log"
cleanup
TEGRA_PID=

curl -fsS http://127.0.0.1:18621/readyz > "$EVIDENCE/voice-ready-after.json"
curl -fsS http://127.0.0.1:8000/health > "$EVIDENCE/gdn-health-after.json"
docker inspect -f '{{.RestartCount}}' "$VOICE" > "$EVIDENCE/voice-restarts-after.txt"
docker inspect -f '{{.RestartCount}}' "$GDN" > "$EVIDENCE/gdn-restarts-after.txt"
[[ "$(cat "$EVIDENCE/voice-restarts-before.txt")" == "$(cat "$EVIDENCE/voice-restarts-after.txt")" ]]
[[ "$(cat "$EVIDENCE/gdn-restarts-before.txt")" == "$(cat "$EVIDENCE/gdn-restarts-after.txt")" ]]
docker stats --no-stream "$VOICE" "$GDN" > "$EVIDENCE/docker-stats-after.txt"
docker top "$VOICE" > "$EVIDENCE/voice-top-after.txt"
docker top "$GDN" > "$EVIDENCE/gdn-top-after.txt"
docker logs --since "$start_epoch" "$VOICE" > "$EVIDENCE/voice-runtime.stdout.log" 2> "$EVIDENCE/voice-runtime.stderr.log"
docker logs --since "$start_epoch" "$GDN" > "$EVIDENCE/gdn-runtime.stdout.log" 2> "$EVIDENCE/gdn-runtime.stderr.log"
python3 "$INPUTS/scan-runtime-errors.py" \
  "$EVIDENCE/voice-runtime.stdout.log" \
  "$EVIDENCE/voice-runtime.stderr.log" \
  "$EVIDENCE/gdn-runtime.stdout.log" \
  "$EVIDENCE/gdn-runtime.stderr.log" \
  > "$EVIDENCE/runtime-error-scan.json"

date --iso-8601=seconds > "$EVIDENCE/gate-completed-at.txt"
echo "PRODUCT_N1_CO_RESIDENCY_PASS"
