#!/usr/bin/env bash
set -euo pipefail

NAME=seeed-voice-v091-r2-canary-base-n1-021112e
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="$VALIDATION/results/r2-base-n1"
INPUTS="$VALIDATION/r2-base-n1-inputs"
WAV="$VALIDATION/corpus/short/zh_short_01.wav"
PORT=18621

[[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == true ]]
curl -fsS "http://127.0.0.1:$PORT/readyz" > "$EVIDENCE/readyz-before-gate.json"
docker inspect -f '{{.RestartCount}}' "$NAME" > "$EVIDENCE/restarts-before.txt"
docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-before.txt"
timeout 3 tegrastats --interval 1000 > "$EVIDENCE/tegrastats-before.txt" || true

python3 "$INPUTS/base-n1-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --wav "$WAV" \
  --output "$EVIDENCE/base-n1-gate.json" \
  --asr-rounds 3 \
  --tts-rounds 3 \
  --cancel-rounds 20 \
  --recovery-timeout 15 \
  | tee "$EVIDENCE/base-n1-gate.stdout.log"

curl -fsS "http://127.0.0.1:$PORT/readyz" > "$EVIDENCE/readyz-after-gate.json"
docker inspect -f '{{.RestartCount}}' "$NAME" > "$EVIDENCE/restarts-after.txt"
[[ "$(cat "$EVIDENCE/restarts-before.txt")" == "$(cat "$EVIDENCE/restarts-after.txt")" ]]
docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-after.txt"
timeout 3 tegrastats --interval 1000 > "$EVIDENCE/tegrastats-after.txt" || true
docker logs "$NAME" > "$EVIDENCE/runtime.stdout.log" 2> "$EVIDENCE/runtime.stderr.log"

python3 "$INPUTS/scan-runtime-errors.py" \
  "$EVIDENCE/runtime.stdout.log" "$EVIDENCE/runtime.stderr.log" \
  > "$EVIDENCE/runtime-error-scan.json"

date --iso-8601=seconds > "$EVIDENCE/gate-completed-at.txt"
echo "BASE_N1_RUNTIME_GATE_PASS"
