#!/usr/bin/env bash
# Baked-path CV smoke: start the baked image on orin-nano (NON-production, isolated
# name + port). NO /cv-bundle mount — relies on baked /opt paths. Host CUDA/TRT RO mounts.
set -eu
IMG=seeed-local-voice:v0.8.0-n1n2-cv-baked
NAME=cv-baked-smoke
PORT=18621

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --runtime nvidia --network host --ipc host \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt:ro \
  -v /usr/src/tensorrt:/usr/src/tensorrt:ro \
  -e LANGUAGE_MODE=multilanguage \
  -e OVS_PROFILE=jetson-edgellm-v080-customvoice \
  -e EDGE_LLM_TTS_WORKER_BIN=/opt/jv-workers/qwen3_tts_streaming_worker \
  -e EDGE_LLM_TTS_TALKER_DIR=/opt/models/qwen3-tts-customvoice/talker_assembled_dir \
  -e EDGE_LLM_TTS_CP_DIR=/opt/models/qwen3-tts-customvoice/code_predictor \
  -e EDGE_LLM_TTS_CODE2WAV_DIR=/opt/models/qwen3-tts-customvoice/code2wav \
  -e EDGE_LLM_TTS_TOKENIZER_DIR=/opt/models/qwen3-tts-customvoice/tokenizer \
  -e EDGE_LLM_TTS_STATEFUL_CODE2WAV=1 \
  -e OVS_TTS_MODEL_ID=qwen3-tts-customvoice \
  -e OVS_TTS_WORKER_CONCURRENCY=1 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-cuda:/host-nvidia-libs:/host-libs \
  -e PYTHONPATH=/opt/speech:/usr/lib/python3.10/dist-packages \
  "$IMG" \
  python3 -m uvicorn server.main:app --host 0.0.0.0 --port "$PORT"

echo "started $NAME on :$PORT (baked paths, no /cv-bundle)"
