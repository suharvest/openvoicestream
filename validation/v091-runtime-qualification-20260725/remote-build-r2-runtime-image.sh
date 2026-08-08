#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
root="$validation/results/r2-runtime-image"
outer_bundle="$validation/input/seeed-local-voice-021112e.bundle"
inner_bundle="$validation/input/jetson-voice-engine-4b28dd2.bundle"
source=/home/harvest/project/seeed-local-voice-v091-r2-021112e-20260726
inner="$source/third_party/jetson-voice-engine"
worker=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2/v091/bin/moss_tts_nano_worker
gate_worker="$source/deploy/artifacts/v091-release-gate/moss_tts_nano_worker"
image=seeed-local-voice:v0.9.1-edgellm-runtime-r2-20260725
log="$validation/logs/r2-runtime-image-build.log"

test ! -e "$source"
test -s "$outer_bundle"
test -s "$inner_bundle"
test -x "$worker"
test "$(sha256sum "$outer_bundle" | cut -d' ' -f1)" = \
  506fec029b72d2bfa89b0409d01fec1c8bda0ccf2b2d51cd19c21d633e285043
test "$(sha256sum "$inner_bundle" | cut -d' ' -f1)" = \
  69528abc56cc751777abac6bdcd87ab812171f580669d9ded5d86ec63712e2ec
test "$(sha256sum "$worker" | cut -d' ' -f1)" = \
  9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb
test "$(docker inspect -f '{{.State.Status}}' seeed-voice-v091)" = exited
test "$(docker inspect -f '{{.State.Status}}' edge-llm-chat-service)" = exited
test "$(docker inspect -f '{{.State.Status}}' translator)" = exited

mkdir -p "$root" "$(dirname "$log")"
exec > >(tee "$log") 2>&1

echo "STAGE materialize-build-context $(date -Is)"
git clone "$outer_bundle" "$source"
test "$(git -C "$source" rev-parse HEAD)" = \
  021112eda3207a57ae91056f24d198303574b555
rmdir "$inner"
git clone "$inner_bundle" "$inner"
test "$(git -C "$inner" rev-parse HEAD)" = \
  4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f
test "$(git -C "$source" rev-parse HEAD:third_party/jetson-voice-engine)" = \
  4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f
test -z "$(git -C "$source" status --short)"
test -z "$(git -C "$inner" status --short)"

mkdir -p "$(dirname "$gate_worker")"
install -m 0755 "$worker" "$gate_worker"
test "$(sha256sum "$gate_worker" | cut -d' ' -f1)" = \
  9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb
test "$(stat -c %a "$gate_worker")" = 755
test "$(stat -c %s "$gate_worker")" = 449864
env \
  LD_LIBRARY_PATH=/opt/onnxruntime-linux-aarch64-1.23.2/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu \
  python3 "$source/scripts/check_moss_worker_runtime.py" \
  --worker "$gate_worker" \
  --release-lock "$source/deploy/artifacts/v091-release-lock.json"
{
  printf 'outer_head=%s\n' "$(git -C "$source" rev-parse HEAD)"
  printf 'inner_head=%s\n' "$(git -C "$inner" rev-parse HEAD)"
  printf 'worker_sha256=%s\n' "$(sha256sum "$gate_worker" | cut -d' ' -f1)"
  printf 'worker_size=%s\n' "$(stat -c %s "$gate_worker")"
  printf 'worker_mode=%s\n' "$(stat -c %a "$gate_worker")"
  printf 'image=%s\n' "$image"
} >"$root/BUILD-CONTEXT-PROVENANCE.txt"

echo "STAGE docker-build $(date -Is)"
docker build --network=host \
  -f "$source/deploy/docker/Dockerfile.jetson.edgellm-v091-runtime" \
  -t "$image" \
  "$source"

echo "STAGE static-semantic-image-gate $(date -Is)"
docker image inspect \
  -f 'id={{.Id}}|repoTags={{json .RepoTags}}|size={{.Size}}|created={{.Created}}|cmd={{json .Config.Cmd}}|entrypoint={{json .Config.Entrypoint}}|labels={{json .Config.Labels}}' \
  "$image" >"$root/runtime-image-inspect.txt"
grep -F '/opt/speech/scripts/start_edgellm_v091_runtime.py' \
  "$root/runtime-image-inspect.txt"
grep -F 'orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2' \
  "$root/runtime-image-inspect.txt"
docker run --rm --entrypoint python3 "$image" \
  /opt/speech/scripts/check_moss_worker_runtime.py \
  --worker /opt/edgellm-v091/bin/moss_tts_nano_worker \
  --release-lock /opt/speech/deploy/v091-release-lock.json \
  >"$root/image-worker-semantic-gate.txt" 2>&1
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
  "$root/image-worker-semantic-gate.txt" \
  "$root/image-pycompile.txt" \
  >"$root/EVIDENCE.SHA256SUMS"
sha256sum -c "$root/EVIDENCE.SHA256SUMS"
cat "$root/BUILD-CONTEXT-PROVENANCE.txt"
cat "$root/runtime-image-inspect.txt"
cat "$root/image-worker-semantic-gate.txt"
echo "PASS r2 runtime image static + semantic gate $(date -Is)"
