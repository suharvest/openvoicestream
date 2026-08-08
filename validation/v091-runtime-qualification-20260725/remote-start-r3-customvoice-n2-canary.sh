#!/usr/bin/env bash
set -euo pipefail

NAME="${CANARY_NAME:-seeed-voice-v091-r3-canary-customvoice-n2-ef27c98}"
IMAGE="${CANARY_IMAGE:-seeed-local-voice:v0.9.1-edgellm-runtime-r3-cvfix-ef27c98-20260726}"
IMAGE_ID="${CANARY_IMAGE_ID:-sha256:04f3e582da5975f636105e16ff8824a664454692dd921fad958fcea1e0de2bee}"
ARTIFACT=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2/v091
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="${CANARY_EVIDENCE:-$VALIDATION/results/r3-customvoice-n2}"
ENV_FILE="$VALIDATION/state/voice.env"
NETWORK=v091-runtime-buildctx-0b8d966_default
PORT="${CANARY_PORT:-18626}"
SOURCE="${CANARY_SOURCE:-/home/harvest/project/seeed-local-voice-v091-r3-ef27c98-20260726}"
VOXEDGE_REVISION="${CANARY_VOXEDGE_REVISION:-8e043d3e676362e103d710a67b53605efc01a241}"

mkdir -p "$EVIDENCE"
for container in seeed-voice-v091 edge-llm-chat-service translator \
  seeed-voice-v091-r2-canary-base-n1-021112e \
  seeed-voice-v091-r2-canary-base-n2-021112e \
  seeed-voice-v091-r2-canary-customvoice-n1-021112e \
  seeed-voice-v091-r3-canary-customvoice-n1-ef27c98; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$container")" == exited ]]
done
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE_ID" ]]
[[ "$(docker image inspect -f '{{index .Config.Labels "com.seeed.voxedge.revision"}}' "$IMAGE")" == \
  "$VOXEDGE_REVISION" ]]
[[ -r "$ENV_FILE" ]]
[[ -f "$ARTIFACT/engines/asr_thinker_full_int4_b2/llm.engine" ]]
[[ -f "$ARTIFACT/engines/tts_customvoice_int4/talker/llm.engine" ]]
[[ -f "$ARTIFACT/engines/tts_customvoice_fp16/code_predictor/llm.engine" ]]
[[ -f "$ARTIFACT/engines/tts_customvoice_fp16/code2wav/code2wav/code2wav.engine" ]]
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "refusing to reuse existing r3 CustomVoice N2 canary: $NAME" >&2
  exit 1
fi

date --iso-8601=seconds > "$EVIDENCE/start-requested-at.txt"
docker run -d \
  --name "$NAME" \
  --runtime nvidia \
  --ipc host \
  --restart no \
  --network "$NETWORK" \
  --env-file "$ENV_FILE" \
  -e OVS_PROFILE=jetson-edgellm-v091-n2 \
  -e EDGE_LLM_ASR_STREAM_MODE=worker \
  -e QWEN3_ARTIFACT_MANIFEST=/opt/qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json \
  -e QWEN3_ARTIFACT_SET=orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2 \
  -e QWEN3_ARTIFACT_ROOT=/opt/edgellm-v091 \
  -e QWEN3_TTS_CODE2WAV_MAX_FRAMES=512 \
  -e TTS_PROVIDER=cuda \
  -e STREAMING_ASR_PROVIDER=cuda \
  -e CUDA_MODULE_LOADING=LAZY \
  -e OVS_LOG_FORMAT=text \
  -e PYTHONPATH=/opt/speech:/usr/lib/python3.10/dist-packages \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-cuda:/host-nvidia-libs:/host-libs \
  -p "$PORT:8000" \
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
  -v "$ARTIFACT:/opt/edgellm-v091:ro" \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  "$IMAGE" > "$EVIDENCE/container-id.txt"

ready=false
for _ in $(seq 1 450); do
  if curl -fsS "http://127.0.0.1:$PORT/readyz" > "$EVIDENCE/readyz.json"; then
    ready=true
    break
  fi
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == true ]] || break
  sleep 2
done
docker inspect -f '{{.State.Status}} {{.State.Health.Status}} {{.RestartCount}} {{.Image}}' \
  "$NAME" > "$EVIDENCE/state-after-ready.txt"
docker logs "$NAME" > "$EVIDENCE/startup.stdout.log" 2> "$EVIDENCE/startup.stderr.log"
[[ "$ready" == true ]] || { echo "r3 CustomVoice N2 failed to become ready" >&2; exit 1; }

grep -F '"EDGE_LLM_TTS_SHARED_ENGINE": "1"' \
  "$SOURCE/configs/profiles/jetson-edgellm-v091-n2.json" \
  > "$EVIDENCE/profile-shared-engine.txt"
grep -F '"OVS_TTS_WORKER_CONCURRENCY": "2"' \
  "$SOURCE/configs/profiles/jetson-edgellm-v091-n2.json" \
  >> "$EVIDENCE/profile-shared-engine.txt"
grep -F '"OVS_MAX_CONCURRENT_SESSIONS": "2"' \
  "$SOURCE/configs/profiles/jetson-edgellm-v091-n2.json" \
  >> "$EVIDENCE/profile-shared-engine.txt"
docker top "$NAME" > "$EVIDENCE/top-after-ready.txt"
grep -F 'asr_thinker_full_int4_b2' "$EVIDENCE/top-after-ready.txt"
grep -F -- '--max_slots 2' "$EVIDENCE/top-after-ready.txt"
date --iso-8601=seconds > "$EVIDENCE/ready-at.txt"
echo "R3_CUSTOMVOICE_N2_CANARY_READY"
