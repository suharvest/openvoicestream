#!/usr/bin/env bash
set -euo pipefail

validation_root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
artifact_root=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091
image=seeed-local-voice:v0.9.1-edgellm-runtime-normalized-d52d973-20260725

docker run --rm \
  --entrypoint python3 \
  -e OVS_PROFILE=jetson-edgellm-v091-moss \
  -e 'LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/usr/local/lib:/host-cuda:/host-nvidia-libs:/host-libs' \
  -v "$artifact_root:/opt/edgellm-v091:ro" \
  -v "$validation_root:/validation:rw" \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  "$image" \
  /validation/scripts/moss-preflight.py
