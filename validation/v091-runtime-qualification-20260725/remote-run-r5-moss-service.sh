#!/usr/bin/env bash
set -euo pipefail

NAME=seeed-voice-v091-r6-canary-moss-n2-b11ada3
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
INPUTS="$VALIDATION/r2-customvoice-inputs"
EVIDENCE="$VALIDATION/results/r6-moss-service"
PORT=18631
START_EPOCH=

capture() {
  rc=$?
  post_rc=0
  trap - EXIT
  set +e
  curl -fsS "http://127.0.0.1:$PORT/readyz" >"$EVIDENCE/readyz-after.json" || post_rc=1
  docker inspect -f '{{.RestartCount}}' "$NAME" >"$EVIDENCE/restarts-after.txt"
  test "$(cat "$EVIDENCE/restarts-before.txt")" = "$(cat "$EVIDENCE/restarts-after.txt")" || post_rc=1
  docker inspect -f '{{.State.Status}} {{.State.Health.Status}} {{.RestartCount}} {{.State.OOMKilled}}' \
    "$NAME" >"$EVIDENCE/state-after.txt"
  docker stats --no-stream "$NAME" >"$EVIDENCE/docker-stats-after.txt"
  docker top "$NAME" >"$EVIDENCE/top-after.txt"
  if test -n "$START_EPOCH"; then
    docker logs --since "$START_EPOCH" "$NAME" >"$EVIDENCE/runtime.stdout.log" 2>"$EVIDENCE/runtime.stderr.log"
    python3 "$INPUTS/scan-runtime-errors.py" \
      "$EVIDENCE/runtime.stdout.log" "$EVIDENCE/runtime.stderr.log" \
      >"$EVIDENCE/runtime-error-scan.json" || post_rc=1
  fi
  date --iso-8601=seconds >"$EVIDENCE/gate-finished-at.txt"
  test "$rc" != 0 || test "$post_rc" = 0 || rc=$post_rc
  test "$rc" != 0 || echo R5_MOSS_SERVICE_PASS
  exit "$rc"
}
trap capture EXIT

mkdir -p "$EVIDENCE"
test "$(docker inspect -f '{{.State.Health.Status}}' "$NAME")" = healthy
curl -fsS "http://127.0.0.1:$PORT/readyz" >"$EVIDENCE/readyz-before.json"
docker inspect -f '{{.RestartCount}}' "$NAME" >"$EVIDENCE/restarts-before.txt"
docker stats --no-stream "$NAME" >"$EVIDENCE/docker-stats-before.txt"
START_EPOCH="$(date +%s)"

python3 "$INPUTS/tts-http-n1-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --sample-rate 48000 \
  --output "$EVIDENCE/moss-http-n1.json" \
  | tee "$EVIDENCE/moss-http-n1.stdout.log"

python3 "$INPUTS/tts-clone-http-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --reference-wav "$VALIDATION/corpus/short/zh_short_01.wav" \
  --sample-rate 48000 \
  --output "$EVIDENCE/moss-http-clone.json" \
  | tee "$EVIDENCE/moss-http-clone.stdout.log"

python3 "$INPUTS/tts-isolated-n2-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --sample-rate 48000 \
  --rounds 20 \
  --output "$EVIDENCE/moss-http-n2.json" \
  | tee "$EVIDENCE/moss-http-n2.stdout.log"

docker top "$NAME" >"$EVIDENCE/top-after-moss-load.txt"
grep -F 'moss_tts_nano_worker' "$EVIDENCE/top-after-moss-load.txt"
grep -F -- '--max-slots=2' "$EVIDENCE/top-after-moss-load.txt"

python3 "$INPUTS/tts-n2-cancel-keep-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --sample-rate 48000 \
  --rounds 3 \
  --recovery-deadline 15 \
  --output "$EVIDENCE/moss-http-n2-cancel.json" \
  | tee "$EVIDENCE/moss-http-n2-cancel.stdout.log"
