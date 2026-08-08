#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
root="$validation/results/r2-runtime-image"
image=seeed-local-voice:v0.9.1-edgellm-runtime-r2-20260725
artifact=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2/v091
worker="$artifact/bin/moss_tts_nano_worker"
ort=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi

test -x "$worker"
test "$(sha256sum "$worker" | cut -d' ' -f1)" = \
  9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb
test "$(docker inspect -f '{{.State.Status}}' seeed-voice-v091)" = exited
test "$(docker inspect -f '{{.State.Status}}' edge-llm-chat-service)" = exited
test "$(docker inspect -f '{{.State.Status}}' translator)" = exited
mkdir -p "$root"

docker image inspect \
  -f 'id={{.Id}}|repoTags={{json .RepoTags}}|size={{.Size}}|created={{.Created}}|cmd={{json .Config.Cmd}}|entrypoint={{json .Config.Entrypoint}}|labels={{json .Config.Labels}}' \
  "$image" >"$root/runtime-image-inspect.txt"
grep -F '/opt/speech/scripts/start_edgellm_v091_runtime.py' \
  "$root/runtime-image-inspect.txt"
grep -F 'orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2' \
  "$root/runtime-image-inspect.txt"

# Build-layer/static gate: host GPU/TRT mounts are intentionally unavailable.
docker run --rm --entrypoint python3 "$image" \
  /opt/speech/scripts/check_moss_worker_runtime.py \
  --worker /opt/edgellm-v091/bin/moss_tts_nano_worker \
  --release-lock /opt/speech/deploy/v091-release-lock.json \
  --skip-ldd \
  >"$root/image-worker-static-gate.txt" 2>&1

# Runtime semantic gate: reproduce the compose host-library and artifact mounts,
# but run only the one-shot checker, never uvicorn/the service entrypoint.
docker run --rm \
  --entrypoint python3 \
  -e "LD_LIBRARY_PATH=$ort:/host-cuda:/host-nvidia-libs:/host-libs" \
  -v "$artifact:/opt/edgellm-v091:ro" \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  "$image" \
  /opt/speech/scripts/check_moss_worker_runtime.py \
  --worker /opt/edgellm-v091/bin/moss_tts_nano_worker \
  --release-lock /opt/speech/deploy/v091-release-lock.json \
  >"$root/image-worker-mounted-semantic-gate.txt" 2>&1

docker run --rm --entrypoint python3 "$image" \
  -m py_compile \
  /opt/speech/scripts/check_moss_worker_runtime.py \
  /opt/speech/scripts/start_edgellm_v091_runtime.py \
  /opt/speech/server/main.py \
  >"$root/image-pycompile.txt" 2>&1

test "$(docker inspect -f '{{.State.Status}}' seeed-voice-v091)" = exited
test "$(docker inspect -f '{{.State.Status}}' edge-llm-chat-service)" = exited
test "$(docker inspect -f '{{.State.Status}}' translator)" = exited
sha256sum \
  "$root/BUILD-CONTEXT-PROVENANCE.txt" \
  "$root/runtime-image-inspect.txt" \
  "$root/image-worker-static-gate.txt" \
  "$root/image-worker-mounted-semantic-gate.txt" \
  "$root/image-pycompile.txt" \
  >"$root/EVIDENCE.SHA256SUMS"
sha256sum -c "$root/EVIDENCE.SHA256SUMS"
cat "$root/BUILD-CONTEXT-PROVENANCE.txt"
cat "$root/runtime-image-inspect.txt"
cat "$root/image-worker-static-gate.txt"
cat "$root/image-worker-mounted-semantic-gate.txt"
echo "PASS r2 image static + mounted semantic gate $(date -Is)"
