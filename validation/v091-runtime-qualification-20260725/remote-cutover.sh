#!/usr/bin/env bash
set -euo pipefail

validation_root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
snapshot_dir="$validation_root/snapshot"
state_dir="$validation_root/state"
new_image=seeed-local-voice:v0.9.1-edgellm-runtime-normalized-d52d973-20260725
canonical_name=seeed-voice-v091
backup_name=seeed-voice-v091-rollback-d52d973
network_name=v091-runtime-buildctx-0b8d966_default
default_profile=jetson-edgellm-v091-qwen3ttsbase
env_file="$state_dir/voice.env"

mkdir -p "$state_dir"
chmod 700 "$state_dir"

wait_ready() {
  local port=$1
  local attempts=${2:-180}
  local index
  for ((index = 1; index <= attempts; index++)); do
    if curl -fsS "http://127.0.0.1:${port}/readyz" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

prepare() {
  python3 - "$snapshot_dir/${canonical_name}.inspect.full.json" "$env_file" <<'PY'
import json
import os
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    data = json.load(handle)
environment = data[0]["Config"]["Env"]
with open(destination, "w", encoding="utf-8") as handle:
    for item in environment:
        handle.write(item)
        handle.write("\n")
os.chmod(destination, 0o600)
PY
  docker image inspect "$new_image" >/dev/null
  docker network inspect "$network_name" >/dev/null
  test -d /home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091
  docker inspect -f '{{.Id}}' "$canonical_name" >"$state_dir/original-container-id.txt"
  docker inspect -f '{{.Image}}' "$canonical_name" >"$state_dir/original-image-id.txt"
  printf 'prepared\n'
}

create_candidate() {
  local name=$1
  local profile=${2:-$default_profile}
  local host_port=${3:-8621}
  local restart_policy=${4:-no}
  local asr_stream_mode=${5:-}
  local -a asr_stream_args=()
  if [[ -n "$asr_stream_mode" ]]; then
    asr_stream_args=(-e "EDGE_LLM_ASR_STREAM_MODE=$asr_stream_mode")
  fi

  test -s "$env_file"
  if docker container inspect "$name" >/dev/null 2>&1; then
    printf 'container already exists: %s\n' "$name" >&2
    return 1
  fi

  docker run -d \
    --name "$name" \
    --runtime nvidia \
    --ipc host \
    --restart "$restart_policy" \
    --network "$network_name" \
    --env-file "$env_file" \
    -e "OVS_PROFILE=$profile" \
    -e QWEN3_TTS_CODE2WAV_MAX_FRAMES=512 \
    "${asr_stream_args[@]}" \
    -p "${host_port}:8000" \
    --memory 8g \
    --memory-swap 14g \
    --health-cmd 'curl -fsS http://127.0.0.1:8000/readyz >/dev/null' \
    --health-interval 30s \
    --health-timeout 10s \
    --health-retries 3 \
    --health-start-period 15m \
    -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt:ro \
    -v /usr/src/tensorrt:/usr/src/tensorrt:ro \
    -v speech-models:/opt/models \
    -v /home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091:/opt/edgellm-v091:ro \
    -v /usr/local/cuda/lib64:/host-cuda:ro \
    -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
    -v /lib/aarch64-linux-gnu:/host-libs:ro \
    "$new_image"
}

stage() {
  test "$(docker inspect -f '{{.State.Running}}' "$canonical_name")" = true
  if docker container inspect "$backup_name" >/dev/null 2>&1; then
    printf 'rollback container already exists: %s\n' "$backup_name" >&2
    return 1
  fi
  docker stop "$canonical_name" >/dev/null
  docker rename "$canonical_name" "$backup_name"
  docker update --restart=no "$backup_name" >/dev/null
  printf 'staged old production as %s\n' "$backup_name"
}

rollback() {
  if docker container inspect "$canonical_name" >/dev/null 2>&1; then
    docker update --restart=no "$canonical_name" >/dev/null || true
    docker stop "$canonical_name" >/dev/null || true
    docker rm "$canonical_name" >/dev/null
  fi
  test "$(docker inspect -f '{{.State.Running}}' "$backup_name")" = false
  docker rename "$backup_name" "$canonical_name"
  docker update --restart=unless-stopped "$canonical_name" >/dev/null
  docker start "$canonical_name" >/dev/null
  wait_ready 8621
  printf 'rollback healthy\n'
}

promote() {
  create_candidate "$canonical_name" "$default_profile" 8621 unless-stopped
  wait_ready 8621
  printf 'candidate healthy\n'
}

case "${1:-}" in
  prepare)
    prepare
    ;;
  create-candidate)
    shift
    create_candidate "$@"
    ;;
  stage)
    stage
    ;;
  rollback)
    rollback
    ;;
  promote)
    promote
    ;;
  wait-ready)
    wait_ready "${2:-8621}" "${3:-180}"
    ;;
  *)
    printf 'usage: %s {prepare|create-candidate NAME [PROFILE [PORT [RESTART [ASR_STREAM_MODE]]]]|stage|rollback|promote|wait-ready [PORT [ATTEMPTS]]}\n' "$0" >&2
    exit 2
    ;;
esac
