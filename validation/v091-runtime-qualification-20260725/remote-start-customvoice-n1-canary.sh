#!/usr/bin/env bash
set -euo pipefail

NAME=seeed-voice-v091-r2-canary-customvoice-n1-021112e
IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-r2-20260725
IMAGE_ID=sha256:74c34eb765223c6e0a59c72d2215a19e030dfc852f6820504e6d6fe575f14938
ARTIFACT=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2/v091
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="$VALIDATION/results/r2-customvoice-n1"
ENV_FILE="$VALIDATION/state/voice.env"
NETWORK=v091-runtime-buildctx-0b8d966_default
PORT=18623

mkdir -p "$EVIDENCE"
for container in seeed-voice-v091 edge-llm-chat-service translator \
  seeed-voice-v091-r2-canary-base-n1-021112e \
  seeed-voice-v091-r2-canary-base-n2-021112e; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$container")" == exited ]]
done
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE_ID" ]]
[[ -r "$ENV_FILE" ]]
[[ -f "$ARTIFACT/engines/asr_thinker_full_int4_b2/llm.engine" ]]
[[ -f "$ARTIFACT/engines/tts_customvoice_int4/talker/llm.engine" ]]
[[ -f "$ARTIFACT/engines/tts_customvoice_fp16/code_predictor/llm.engine" ]]
[[ -f "$ARTIFACT/engines/tts_customvoice_fp16/code2wav/code2wav/code2wav.engine" ]]
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "refusing to reuse existing CustomVoice N1 canary: $NAME" >&2
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
  -e OVS_PROFILE=jetson-edgellm-v091-customvoice \
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
[[ "$ready" == true ]] || { echo "CustomVoice N1 failed to become ready" >&2; exit 1; }

docker exec "$NAME" env > "$EVIDENCE/resolved-env.full.txt"
grep -E '^(OVS_PROFILE|EDGE_LLM_ASR_MAX_CONCURRENT|EDGE_LLM_ASR_ENGINE_DIR|EDGE_LLM_TTS_TALKER_DIR|EDGE_LLM_TTS_CP_DIR|OVS_TTS_WORKER_CONCURRENCY|OVS_MAX_CONCURRENT_SESSIONS)=' \
  "$EVIDENCE/resolved-env.full.txt" | cut -d= -f2- > "$EVIDENCE/resolved-env.txt"
grep -Fx 'jetson-edgellm-v091-customvoice' "$EVIDENCE/resolved-env.txt"
grep -F 'asr_thinker_full_int4_b2' "$EVIDENCE/resolved-env.txt"
grep -F 'tts_customvoice_int4/talker' "$EVIDENCE/resolved-env.txt"
grep -F 'tts_customvoice_fp16/code_predictor' "$EVIDENCE/resolved-env.txt"
date --iso-8601=seconds > "$EVIDENCE/ready-at.txt"
echo "CUSTOMVOICE_N1_CANARY_READY"
