#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
root="${BUILD_ROOT:-$validation/results/r3-cvfix-runtime-image}"
outer_bundle="${OUTER_BUNDLE:-$validation/input/seeed-local-voice-ef27c98.bundle}"
inner_bundle="$validation/input/jetson-voice-engine-4b28dd2.bundle"
source="${BUILD_SOURCE:-/home/harvest/project/seeed-local-voice-v091-r3-ef27c98-20260726}"
inner="$source/third_party/jetson-voice-engine"
artifact=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2/v091
worker="$artifact/bin/moss_tts_nano_worker"
gate_worker="$source/deploy/artifacts/v091-release-gate/moss_tts_nano_worker"
wheel="$source/deploy/wheels/voxedge-0.0.5a0-py3-none-any.whl"
image="${BUILD_IMAGE:-seeed-local-voice:v0.9.1-edgellm-runtime-r3-cvfix-ef27c98-20260726}"
old_image="${PRESERVE_IMAGE:-seeed-local-voice:v0.9.1-edgellm-runtime-r2-20260725}"
log="${BUILD_LOG:-$validation/logs/r3-cvfix-runtime-image-build.log}"
outer_sha="${OUTER_SHA:-b740e1c76da1915fa75c1d634337fe76d604663e625487a03df49293e807f925}"
inner_sha=69528abc56cc751777abac6bdcd87ab812171f580669d9ded5d86ec63712e2ec
wheel_sha="${WHEEL_SHA:-c39a68b36e8d62c1cf74443131eca93ace0e3de9bd890858c13a2dc3cf05a037}"
worker_sha=9d114d8390e684c8876e2ef9e20e28ee6d4ec6ce18b81df5da3ba64c8f057deb
outer_head="${OUTER_HEAD:-ef27c98eedbfb8701e8433c4ff6b1feb7aee5961}"
inner_head=4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f
voxedge_head="${VOXEDGE_HEAD:-8e043d3e676362e103d710a67b53605efc01a241}"

test ! -e "$source"
test ! -e "$root"
test -s "$outer_bundle"
test -s "$inner_bundle"
test -x "$worker"
test "$(sha256sum "$outer_bundle" | cut -d' ' -f1)" = "$outer_sha"
test "$(sha256sum "$inner_bundle" | cut -d' ' -f1)" = "$inner_sha"
test "$(sha256sum "$worker" | cut -d' ' -f1)" = "$worker_sha"
test "$(docker inspect -f '{{.State.Status}}' seeed-voice-v091)" = exited
test "$(docker inspect -f '{{.State.Status}}' edge-llm-chat-service)" = exited
test "$(docker inspect -f '{{.State.Status}}' translator)" = exited
old_image_id="$(docker image inspect -f '{{.Id}}' "$old_image")"
if docker image inspect "$image" >/dev/null 2>&1; then
  echo "refusing to reuse candidate image tag: $image" >&2
  exit 1
fi

mkdir -p "$root" "$(dirname "$log")"
exec > >(tee "$log") 2>&1

echo "STAGE materialize-fresh-context $(date -Is)"
git clone "$outer_bundle" "$source"
test "$(git -C "$source" rev-parse HEAD)" = "$outer_head"
rmdir "$inner"
git clone "$inner_bundle" "$inner"
test "$(git -C "$inner" rev-parse HEAD)" = "$inner_head"
test "$(git -C "$source" rev-parse HEAD:third_party/jetson-voice-engine)" = "$inner_head"
test -z "$(git -C "$source" status --short)"
test -z "$(git -C "$inner" status --short)"
test "$(sha256sum "$wheel" | cut -d' ' -f1)" = "$wheel_sha"

mkdir -p "$(dirname "$gate_worker")"
install -m 0755 "$worker" "$gate_worker"
test "$(sha256sum "$gate_worker" | cut -d' ' -f1)" = "$worker_sha"
test "$(stat -c %a "$gate_worker")" = 755
env \
  LD_LIBRARY_PATH=/opt/onnxruntime-linux-aarch64-1.23.2/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu \
  python3 "$source/scripts/check_moss_worker_runtime.py" \
  --worker "$gate_worker" \
  --release-lock "$source/deploy/artifacts/v091-release-lock.json"

wheel_backend_sha="$(
  unzip -p "$wheel" voxedge/backends/jetson/trt_edge_llm_tts.py | sha256sum | cut -d' ' -f1
)"
{
  printf 'outer_head=%s\n' "$outer_head"
  printf 'inner_head=%s\n' "$inner_head"
  printf 'voxedge_head=%s\n' "$voxedge_head"
  printf 'outer_bundle_sha256=%s\n' "$outer_sha"
  printf 'inner_bundle_sha256=%s\n' "$inner_sha"
  printf 'wheel_sha256=%s\n' "$wheel_sha"
  printf 'wheel_backend_sha256=%s\n' "$wheel_backend_sha"
  printf 'worker_sha256=%s\n' "$worker_sha"
  printf 'image=%s\n' "$image"
  printf 'old_image=%s\n' "$old_image"
  printf 'old_image_id=%s\n' "$old_image_id"
} >"$root/BUILD-CONTEXT-PROVENANCE.txt"

echo "STAGE docker-build $(date -Is)"
docker build --network=host \
  -f "$source/deploy/docker/Dockerfile.jetson.edgellm-v091-runtime" \
  -t "$image" \
  "$source"

echo "STAGE static-and-mounted-gates $(date -Is)"
docker image inspect \
  -f 'id={{.Id}}|repoTags={{json .RepoTags}}|size={{.Size}}|created={{.Created}}|cmd={{json .Config.Cmd}}|entrypoint={{json .Config.Entrypoint}}|labels={{json .Config.Labels}}' \
  "$image" >"$root/runtime-image-inspect.txt"
grep -F '/opt/speech/scripts/start_edgellm_v091_runtime.py' \
  "$root/runtime-image-inspect.txt"
grep -F "$voxedge_head" "$root/runtime-image-inspect.txt"
grep -F 'orin-nx-edgellm-v091-jp62-trt103-sm87-20260725-r2' \
  "$root/runtime-image-inspect.txt"

docker run --rm --entrypoint python3 "$image" -c \
  "import hashlib, importlib.metadata as m; import voxedge.backends.jetson.trt_edge_llm_tts as x; print('version='+m.version('voxedge')); print('backend_sha256='+hashlib.sha256(open(x.__file__,'rb').read()).hexdigest()); print('backend_file='+x.__file__)" \
  >"$root/installed-voxedge.txt"
grep -Fx 'version=0.0.5a0' "$root/installed-voxedge.txt"
grep -Fx "backend_sha256=$wheel_backend_sha" "$root/installed-voxedge.txt"

docker run --rm --entrypoint python3 "$image" -c \
  "import inspect; from voxedge.backends.jetson.worker_io import WorkerIO; import voxedge.backends.jetson.trt_edge_llm_tts as t; assert 'cancel_event' in inspect.signature(WorkerIO.request).parameters; s=inspect.getsource(t.TRTEdgeLLMTTSBackend._generate_streaming_single); assert 'cancel_event=cancel_event' in s and 'cancel_ack' in s; print('OUT_OF_BAND_CANCEL_STATIC_PASS')" \
  >"$root/out-of-band-cancel-static.txt"

docker run --rm --entrypoint python3 "$image" -c \
  "from voxedge.backends.base import TTSCapability; from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSBackend,TRTEdgeLLMTTSConfig; c=TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(model_id='qwen3-tts-customvoice')); b=TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(model_id='qwen3-tts')); assert c.supports_voice_cloning is False and TTSCapability.VOICE_CLONE not in c.capabilities; assert b.supports_voice_cloning is True and TTSCapability.VOICE_CLONE in b.capabilities; print('CUSTOMVOICE_CAPABILITY_STATIC_PASS')" \
  >"$root/customvoice-capability-static.txt"

docker run --rm --entrypoint python3 "$image" \
  /opt/speech/scripts/check_moss_worker_runtime.py \
  --worker /opt/edgellm-v091/bin/moss_tts_nano_worker \
  --release-lock /opt/speech/deploy/v091-release-lock.json \
  --skip-ldd \
  >"$root/image-worker-static-gate.txt" 2>&1

ort=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi
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

docker run --rm --entrypoint python3 "$image" -m py_compile \
  /opt/speech/scripts/check_moss_worker_runtime.py \
  /opt/speech/scripts/start_edgellm_v091_runtime.py \
  /opt/speech/server/main.py \
  >"$root/image-pycompile.txt" 2>&1

test "$(docker image inspect -f '{{.Id}}' "$old_image")" = "$old_image_id"
test "$(docker inspect -f '{{.State.Status}}' seeed-voice-v091)" = exited
test "$(docker inspect -f '{{.State.Status}}' edge-llm-chat-service)" = exited
test "$(docker inspect -f '{{.State.Status}}' translator)" = exited
sha256sum \
  "$root/BUILD-CONTEXT-PROVENANCE.txt" \
  "$root/runtime-image-inspect.txt" \
  "$root/installed-voxedge.txt" \
  "$root/out-of-band-cancel-static.txt" \
  "$root/customvoice-capability-static.txt" \
  "$root/image-worker-static-gate.txt" \
  "$root/image-worker-mounted-semantic-gate.txt" \
  "$root/image-pycompile.txt" \
  >"$root/EVIDENCE.SHA256SUMS"
sha256sum -c "$root/EVIDENCE.SHA256SUMS"
cat "$root/BUILD-CONTEXT-PROVENANCE.txt"
cat "$root/runtime-image-inspect.txt"
cat "$root/installed-voxedge.txt"
cat "$root/out-of-band-cancel-static.txt"
cat "$root/customvoice-capability-static.txt"
cat "$root/image-worker-static-gate.txt"
cat "$root/image-worker-mounted-semantic-gate.txt"
echo "PASS r3 CustomVoice-fix image static + mounted gates $(date -Is)"
