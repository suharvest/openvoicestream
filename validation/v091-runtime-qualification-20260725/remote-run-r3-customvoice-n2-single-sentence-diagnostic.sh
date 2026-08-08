#!/usr/bin/env bash
set -euo pipefail

NAME="${CANARY_NAME:-seeed-voice-v091-r3-canary-customvoice-n2-ef27c98}"
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="${CANARY_EVIDENCE:-$VALIDATION/results/r3-customvoice-n2-single-sentence-diagnostic}"
PORT="${CANARY_PORT:-18626}"
INPUTS="$VALIDATION/r2-customvoice-inputs"
TEGRA_PID=
START_EPOCH=

capture() {
  local rc=$?
  trap - EXIT
  set +e
  if [[ -n "$TEGRA_PID" ]]; then
    kill "$TEGRA_PID" >/dev/null 2>&1
    wait "$TEGRA_PID" >/dev/null 2>&1
  fi
  curl -fsS "http://127.0.0.1:$PORT/readyz" > "$EVIDENCE/readyz-after.json"
  docker inspect -f '{{.RestartCount}}' "$NAME" > "$EVIDENCE/restarts-after.txt"
  docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-after.txt"
  docker top "$NAME" > "$EVIDENCE/top-after.txt"
  docker logs --since "$START_EPOCH" "$NAME" > "$EVIDENCE/runtime.stdout.log" 2> "$EVIDENCE/runtime.stderr.log"
  python3 "$INPUTS/scan-runtime-errors.py" \
    "$EVIDENCE/runtime.stdout.log" "$EVIDENCE/runtime.stderr.log" \
    > "$EVIDENCE/runtime-error-scan.json"
  date --iso-8601=seconds > "$EVIDENCE/gate-finished-at.txt"
  exit "$rc"
}
trap capture EXIT

mkdir -p "$EVIDENCE"
[[ "$(docker inspect -f '{{.State.Health.Status}}' "$NAME")" == healthy ]]
[[ "$(docker inspect -f '{{.RestartCount}}' "$NAME")" == 0 ]]
curl -fsS "http://127.0.0.1:$PORT/readyz" > "$EVIDENCE/readyz-before.json"
docker stats --no-stream "$NAME" > "$EVIDENCE/docker-stats-before.txt"
START_EPOCH="$(date +%s)"
tegrastats --interval 500 --logfile "$EVIDENCE/tegrastats-during.log" &
TEGRA_PID=$!

python3 "$INPUTS/customvoice-tts-n2-cancel-single-gate.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --rounds 5 \
  --recovery-deadline 15 \
  --speaker-cancel 3065 \
  --speaker-keep 3061 \
  --speaker-recovery 3066 \
  --cancel-text '这一条语音会在收到首个音频块后立即取消并验证工作槽位能够及时释放不要继续生成后续音频' \
  --keep-text '保持第二路语音连续完整输出同时取消另一条请求不能中断这一条并继续验证不同音色之间严格隔离' \
  --recovery-text '取消后释放槽位并恢复正常语音' \
  --output "$EVIDENCE/cancel-single-sentence.json" \
  | tee "$EVIDENCE/cancel-single-sentence.stdout.log"
