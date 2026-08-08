#!/usr/bin/env bash
set -euo pipefail

NAME=seeed-voice-v091-r6-canary-moss-n2-b11ada3
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
INPUTS="$VALIDATION/r2-customvoice-inputs"
EVIDENCE="$VALIDATION/results/r6-moss-cancel-firstpcm-strict"
PORT=18631
mkdir -p "$EVIDENCE"
test "$(docker inspect -f '{{.State.Health.Status}}' "$NAME")" = healthy
restarts="$(docker inspect -f '{{.RestartCount}}' "$NAME")"
start_epoch="$(date +%s)"

capture() {
  rc=$?
  set +e
  docker logs --since "$start_epoch" "$NAME" >"$EVIDENCE/runtime.stdout.log" 2>"$EVIDENCE/runtime.stderr.log"
  python3 "$INPUTS/scan-runtime-errors.py" \
    "$EVIDENCE/runtime.stdout.log" "$EVIDENCE/runtime.stderr.log" \
    >"$EVIDENCE/runtime-error-scan.json"
  docker inspect -f '{{.State.Status}} {{.State.Health.Status}} {{.RestartCount}} {{.State.OOMKilled}}' \
    "$NAME" >"$EVIDENCE/state-after.txt"
  docker stats --no-stream "$NAME" >"$EVIDENCE/docker-stats-after.txt"
  docker top "$NAME" >"$EVIDENCE/top-after.txt"
  test "$restarts" = "$(docker inspect -f '{{.RestartCount}}' "$NAME")" || rc=1
  exit "$rc"
}
trap capture EXIT

python3 "$INPUTS/tts-n2-cancel-keep-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --sample-rate 48000 \
  --rounds 20 \
  --recovery-deadline 15 \
  --cancel-text "测试取消。" \
  --cancel-head-start 0.05 \
  --keep-repeat 6 \
  --output "$EVIDENCE/moss-http-n2-cancel.json" \
  | tee "$EVIDENCE/moss-http-n2-cancel.stdout.log"
