#!/usr/bin/env bash
set -euo pipefail

NAME="${CANARY_NAME:-seeed-voice-v091-r3-canary-customvoice-n2-ef27c98}"
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="${CANARY_EVIDENCE:-$VALIDATION/results/r3-customvoice-n2}"
PORT="${CANARY_PORT:-18626}"
INPUTS="$VALIDATION/r2-customvoice-inputs"
BASE_INPUTS="$VALIDATION/r2-base-n1-inputs"
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
  curl -fsS "http://127.0.0.1:$PORT/readyz" > "$EVIDENCE/readyz-after.json" || post_rc=1
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
  [[ "$rc" != 0 ]] || echo "R3_CUSTOMVOICE_N2_PASS"
  exit "$rc"
}
trap capture EXIT

mkdir -p "$EVIDENCE"
[[ "$(docker inspect -f '{{.State.Health.Status}}' "$NAME")" == healthy ]]
for container in seeed-voice-v091 edge-llm-chat-service translator \
  seeed-voice-v091-r2-canary-base-n1-021112e \
  seeed-voice-v091-r2-canary-base-n2-021112e \
  seeed-voice-v091-r2-canary-customvoice-n1-021112e \
  seeed-voice-v091-r3-canary-customvoice-n1-ef27c98; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$container")" == exited ]]
done
curl -fsS "http://127.0.0.1:$PORT/readyz" > "$EVIDENCE/readyz-before.json"
docker inspect -f '{{.RestartCount}}' "$NAME" > "$EVIDENCE/restarts-before.txt"
docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-before.txt"
START_EPOCH="$(date +%s)"
tegrastats --interval 500 --logfile "$EVIDENCE/tegrastats-during.log" &
TEGRA_PID=$!

if [[ "${SKIP_CORE:-0}" != 1 ]]; then
python3 "$BASE_INPUTS/asr-isolated-n2-gate.py" \
  --base-url "ws://127.0.0.1:$PORT" \
  --wav-a "$VALIDATION/corpus/short/zh_short_01.wav" \
  --wav-b "$VALIDATION/corpus/short/en_short_01.wav" \
  --rounds 1 \
  --output "$EVIDENCE/asr-n2-smoke.json" \
  | tee "$EVIDENCE/asr-n2-smoke.stdout.log"

python3 "$INPUTS/customvoice-tts-n2-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --rounds 3 \
  --speaker-a 3066 \
  --speaker-b 3061 \
  --language-a chinese \
  --language-b chinese \
  --output "$EVIDENCE/customvoice-tts-n2.json" \
  | tee "$EVIDENCE/customvoice-tts-n2.stdout.log"

docker top "$NAME" > "$EVIDENCE/top-after-tts-load.txt"
grep -F 'tts_customvoice_int4/talker' "$EVIDENCE/top-after-tts-load.txt"
grep -F 'tts_customvoice_fp16/code_predictor' "$EVIDENCE/top-after-tts-load.txt"
grep -F -- '--max_slots 2' "$EVIDENCE/top-after-tts-load.txt"
fi

if [[ "${SKIP_CANCEL:-0}" != 1 ]]; then
python3 "$INPUTS/customvoice-tts-n2-cancel-keep-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --rounds 20 \
  --recovery-deadline 15 \
  --speaker-cancel 3065 \
  --speaker-keep 3061 \
  --speaker-recovery 3066 \
  --output "$EVIDENCE/customvoice-tts-n2-cancel-keep.json" \
  | tee "$EVIDENCE/customvoice-tts-n2-cancel-keep.stdout.log"
fi

[[ "$(cat "$EVIDENCE/restarts-before.txt")" == "$(docker inspect -f '{{.RestartCount}}' "$NAME")" ]]
