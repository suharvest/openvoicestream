#!/usr/bin/env bash
set -euo pipefail

NAME=seeed-voice-v091-r3-canary-customvoice-n1-ef27c98
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="$VALIDATION/results/r3-customvoice-n1"
INPUTS="$VALIDATION/r2-customvoice-inputs"
TEGRA_PID=
START_EPOCH=

capture() {
  local rc=$?
  local post_rc=0
  trap - EXIT
  set +e
  if [[ -n "$TEGRA_PID" ]]; then
    kill "$TEGRA_PID" >/dev/null 2>&1
    wait "$TEGRA_PID" >/dev/null 2>&1
  fi
  curl -fsS http://127.0.0.1:18625/readyz > "$EVIDENCE/readyz-after.json" || post_rc=1
  docker inspect -f '{{.RestartCount}}' "$NAME" > "$EVIDENCE/restarts-after.txt"
  [[ "$(cat "$EVIDENCE/restarts-before.txt")" == "$(cat "$EVIDENCE/restarts-after.txt")" ]] || post_rc=1
  docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-after.txt"
  docker top "$NAME" > "$EVIDENCE/top-after.txt"
  if [[ -n "$START_EPOCH" ]]; then
    docker logs --since "$START_EPOCH" "$NAME" > "$EVIDENCE/runtime.stdout.log" 2> "$EVIDENCE/runtime.stderr.log"
    python3 "$INPUTS/scan-runtime-errors.py" \
      "$EVIDENCE/runtime.stdout.log" "$EVIDENCE/runtime.stderr.log" \
      > "$EVIDENCE/runtime-error-scan.json" || post_rc=1
  fi
  date --iso-8601=seconds > "$EVIDENCE/gate-finished-at.txt"
  [[ "$rc" != 0 || "$post_rc" == 0 ]] || rc=$post_rc
  [[ "$rc" != 0 ]] || echo "R3_CUSTOMVOICE_N1_PASS"
  exit "$rc"
}
trap capture EXIT

[[ "$(docker inspect -f '{{.State.Health.Status}}' "$NAME")" == healthy ]]
for container in seeed-voice-v091 edge-llm-chat-service translator \
  seeed-voice-v091-r2-canary-base-n1-021112e \
  seeed-voice-v091-r2-canary-base-n2-021112e \
  seeed-voice-v091-r2-canary-customvoice-n1-021112e; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$container")" == exited ]]
done
curl -fsS http://127.0.0.1:18625/readyz > "$EVIDENCE/readyz-before.json"
docker inspect -f '{{.RestartCount}}' "$NAME" > "$EVIDENCE/restarts-before.txt"
docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-before.txt"
START_EPOCH="$(date +%s)"
tegrastats --interval 500 --logfile "$EVIDENCE/tegrastats-during.log" &
TEGRA_PID=$!

python3 "$INPUTS/customvoice-n1-gate.py" \
  --base-url http://127.0.0.1:18625 \
  --rounds 10 \
  --recovery-deadline 15 \
  --output "$EVIDENCE/customvoice-n1.json" \
  | tee "$EVIDENCE/customvoice-n1.stdout.log"

docker top "$NAME" > "$EVIDENCE/top-after-tts-load.txt"
grep -F 'tts_customvoice_int4/talker' "$EVIDENCE/top-after-tts-load.txt"
grep -F 'tts_customvoice_fp16/code_predictor' "$EVIDENCE/top-after-tts-load.txt"
if grep -F -- '--max_slots 2' "$EVIDENCE/top-after-tts-load.txt"; then
  echo "N1 CustomVoice unexpectedly uses max_slots 2" >&2
  exit 1
fi
[[ "$(cat "$EVIDENCE/restarts-before.txt")" == "$(docker inspect -f '{{.RestartCount}}' "$NAME")" ]]
