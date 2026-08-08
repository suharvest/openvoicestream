#!/usr/bin/env bash
set -euo pipefail

NAME=seeed-voice-v091-r2-canary-base-n2-021112e
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="$VALIDATION/results/r2-base-isolated-n2"
INPUTS="$VALIDATION/r2-base-n1-inputs"
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
  curl -fsS http://127.0.0.1:18622/readyz > "$EVIDENCE/readyz-after.json" || post_rc=1
  docker inspect -f '{{.RestartCount}}' "$NAME" > "$EVIDENCE/restarts-after.txt"
  if [[ "$(cat "$EVIDENCE/restarts-before.txt")" != "$(cat "$EVIDENCE/restarts-after.txt")" ]]; then
    post_rc=1
  fi
  docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-after.txt"
  docker top "$NAME" > "$EVIDENCE/top-after.txt"
  if [[ -n "$START_EPOCH" ]]; then
    docker logs --since "$START_EPOCH" "$NAME" > "$EVIDENCE/runtime.stdout.log" 2> "$EVIDENCE/runtime.stderr.log"
    python3 "$INPUTS/scan-runtime-errors.py" \
      "$EVIDENCE/runtime.stdout.log" "$EVIDENCE/runtime.stderr.log" \
      > "$EVIDENCE/runtime-error-scan.json" || post_rc=1
  fi
  date --iso-8601=seconds > "$EVIDENCE/gate-finished-at.txt"
  if [[ "$rc" == 0 && "$post_rc" != 0 ]]; then
    rc=$post_rc
  fi
  if [[ "$rc" == 0 ]]; then
    echo "BASE_ISOLATED_N2_PASS"
  fi
  exit "$rc"
}
trap capture EXIT

[[ "$(docker inspect -f '{{.State.Health.Status}}' "$NAME")" == healthy ]]
for container in seeed-voice-v091 edge-llm-chat-service translator \
  seeed-voice-v091-r2-canary-base-n1-021112e; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$container")" == exited ]]
done
curl -fsS http://127.0.0.1:18622/readyz > "$EVIDENCE/readyz-before.json"
docker inspect -f '{{.RestartCount}}' "$NAME" > "$EVIDENCE/restarts-before.txt"
docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-before.txt"
START_EPOCH="$(date +%s)"
tegrastats --interval 500 --logfile "$EVIDENCE/tegrastats-during.log" &
TEGRA_PID=$!

python3 "$INPUTS/asr-isolated-n2-gate.py" \
  --base-url ws://127.0.0.1:18622 \
  --wav-a "$VALIDATION/corpus/short/zh_short_01.wav" \
  --wav-b "$VALIDATION/corpus/short/en_short_01.wav" \
  --rounds 3 \
  --output "$EVIDENCE/asr-isolated-n2.json" \
  | tee "$EVIDENCE/asr-isolated-n2.stdout.log"

python3 "$INPUTS/tts-isolated-n2-gate.py" \
  --base-url http://127.0.0.1:18622 \
  --rounds 3 \
  --output "$EVIDENCE/tts-isolated-n2.json" \
  | tee "$EVIDENCE/tts-isolated-n2.stdout.log"

docker top "$NAME" > "$EVIDENCE/top-after-tts-load.txt"
grep -F 'tts_base_talker_b2_kv1536' "$EVIDENCE/top-after-tts-load.txt"
grep -F 'tts_base_code_predictor_b2_kv1536' "$EVIDENCE/top-after-tts-load.txt"
grep -F -- '--max_slots 2' "$EVIDENCE/top-after-tts-load.txt"

python3 "$INPUTS/tts-n2-cancel-keep-gate.py" \
  --base-url http://127.0.0.1:18622 \
  --rounds 20 \
  --recovery-deadline 15 \
  --output "$EVIDENCE/tts-n2-cancel-keep.json" \
  | tee "$EVIDENCE/tts-n2-cancel-keep.stdout.log"

[[ "$(cat "$EVIDENCE/restarts-before.txt")" == "$(docker inspect -f '{{.RestartCount}}' "$NAME")" ]]
