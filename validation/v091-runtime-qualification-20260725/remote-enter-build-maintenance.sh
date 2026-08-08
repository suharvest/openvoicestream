#!/usr/bin/env bash
set -euo pipefail

root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/results/strict-formal-r2
before="$root/maintenance-before.txt"
after="$root/maintenance-stopped.txt"
build_pid=3257508
containers=(
  seeed-voice-v091
  edge-llm-chat-service
  translator
)

test -d "/proc/$build_pid"
{
  printf 'recorded_at=%s\n' "$(date -Is)"
  printf 'build_pid=%s\n' "$build_pid"
  docker inspect \
    -f 'name={{.Name}}|image={{.Config.Image}}|restart={{.HostConfig.RestartPolicy.Name}}|status={{.State.Status}}|restart_count={{.RestartCount}}' \
    "${containers[@]}"
} >"$before"
cat "$before"

docker stop --time 30 "${containers[@]}"

{
  printf 'stopped_at=%s\n' "$(date -Is)"
  printf 'build_pid=%s\n' "$build_pid"
  docker inspect \
    -f 'name={{.Name}}|image={{.Config.Image}}|restart={{.HostConfig.RestartPolicy.Name}}|status={{.State.Status}}|restart_count={{.RestartCount}}' \
    "${containers[@]}"
} >"$after"
cat "$after"
test -d "/proc/$build_pid"
test "$(docker inspect -f '{{.State.Status}}' seeed-voice-v091)" = exited
test "$(docker inspect -f '{{.State.Status}}' edge-llm-chat-service)" = exited
test "$(docker inspect -f '{{.State.Status}}' translator)" = exited
