#!/usr/bin/env bash
set -euo pipefail

NAME=seeed-voice-v091-r6-canary-moss-n2-b11ada3
IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-r6-moss-n2-b11ada3-20260726
IMAGE_ID=sha256:31b71218a2d87696a31b676df2913287947f76458f824e2f38ea0a2913db2ef9
ARTIFACT=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2/v091
VALIDATION=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
EVIDENCE="$VALIDATION/results/r6-moss-n2"
ENV_FILE="$VALIDATION/state/voice.env"
NETWORK=v091-runtime-buildctx-0b8d966_default
PORT=18631

mkdir -p "$EVIDENCE"
test "$(docker inspect -f '{{.State.Status}}' seeed-voice-v091-r5-canary-customvoice-n2-6e83cf0)" = exited
test "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$IMAGE_ID"
test -r "$ENV_FILE"
test -x "$ARTIFACT/bin/moss_tts_nano_worker"
test -s "$ARTIFACT/engines/moss/moss_tts_prefill.plan"
test -s "$ARTIFACT/engines/moss/codec/codec_decode_step.plan"
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "refusing to reuse existing r6 MOSS N2 canary: $NAME" >&2
  exit 1
fi

date --iso-8601=seconds >"$EVIDENCE/start-requested-at.txt"
docker run -d \
  --name "$NAME" \
  --runtime nvidia \
  --ipc host \
  --restart no \
  --network "$NETWORK" \
  --env-file "$ENV_FILE" \
  -e OVS_PROFILE=jetson-edgellm-v091-moss \
  -e EDGE_LLM_ASR_STREAM_MODE=worker \
  -e QWEN3_ARTIFACT_SET=orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2 \
  -e QWEN3_ARTIFACT_ROOT=/opt/edgellm-v091 \
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
  "$IMAGE" >"$EVIDENCE/container-id.txt"

ready=false
for _ in $(seq 1 450); do
  if curl -fsS "http://127.0.0.1:$PORT/readyz" >"$EVIDENCE/readyz.json"; then
    ready=true
    break
  fi
  test "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true || break
  sleep 2
done
docker inspect -f '{{.State.Status}} {{.State.Health.Status}} {{.RestartCount}} {{.Image}} {{.State.OOMKilled}}' \
  "$NAME" >"$EVIDENCE/state-after-ready.txt"
docker logs "$NAME" >"$EVIDENCE/startup.stdout.log" 2>"$EVIDENCE/startup.stderr.log"
test "$ready" = true
docker top "$NAME" >"$EVIDENCE/top-after-ready.txt"
grep -F 'asr_thinker_full_int4_b2' "$EVIDENCE/top-after-ready.txt"
grep -F -- '--max_slots 2' "$EVIDENCE/top-after-ready.txt"
date --iso-8601=seconds >"$EVIDENCE/ready-at.txt"
echo R5_MOSS_N2_CANARY_READY
