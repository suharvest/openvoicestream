#!/usr/bin/env bash
set -euo pipefail

validation_root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
artifact_root=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091
image=seeed-local-voice:v0.9.1-edgellm-runtime-normalized-d52d973-20260725
probe_root=$(mktemp -d /tmp/moss-ort-soname-probe.XXXXXX)
probe_container=moss-ort-copy-$$
cleanup() {
  docker rm "$probe_container" >/dev/null 2>&1 || true
  rm -rf "$probe_root"
}
trap cleanup EXIT
docker create --name "$probe_container" "$image" >/dev/null
docker cp \
  "$probe_container:/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/libonnxruntime.so.1.23.2" \
  "$probe_root/libonnxruntime.so.1"
docker rm "$probe_container" >/dev/null

set +e
docker run --rm \
  --entrypoint /usr/bin/ldd \
  -e 'LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/usr/local/lib:/host-cuda:/host-nvidia-libs:/host-libs' \
  -v "$artifact_root:/opt/edgellm-v091:ro" \
  -v "$probe_root/libonnxruntime.so.1:/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/libonnxruntime.so.1:ro" \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  "$image" \
  -r /opt/edgellm-v091/bin/moss_tts_nano_worker \
  >"$validation_root/results/moss-ort-soname-1.23.2-ldd.txt" 2>&1
rc=$?
set -e

printf 'rc=%s\n' "$rc"
grep -E 'not found|undefined symbol|libonnxruntime' \
  "$validation_root/results/moss-ort-soname-1.23.2-ldd.txt" || true
exit "$rc"
