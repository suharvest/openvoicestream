#!/usr/bin/env bash
set -euo pipefail

# Rebuild the v0.9.1 Qwen3-TTS Base Code2Wav engine with a production-sized
# dynamic profile. The upstream default maxCodeLen=2000 reserves roughly
# 2.95 GiB of activation memory on Orin NX; 512 frames still covers about
# 41 seconds of 12.5 Hz semantic codes and is retained alongside (not over)
# the full-range engine.

source_dir="${CODE2WAV_ONNX_DIR:-/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/onnx/tts-base-int4-20260725/code2wav}"
output_dir="${CODE2WAV_ENGINE_DIR:-/home/harvest/releases/edgellm/v0.9.1-jp62-trt103-sm87-20260724/engines/tts-base-int4-20260725/code2wav-prod512}"
audio_builder="${EDGELLM_AUDIO_BUILD:-/home/harvest/build/edgellm-v091-voice-cute-20260724T0925Z/examples/multimodal/audio_build}"
plugin_path="${EDGELLM_PLUGIN_PATH:-/home/harvest/build/edgellm-v091-voice-cute-20260724T0925Z/libNvInfer_edgellm_plugin.so.1.0}"
min_code_len="${CODE2WAV_MIN_CODE_LEN:-1}"
opt_code_len="${CODE2WAV_OPT_CODE_LEN:-128}"
max_code_len="${CODE2WAV_MAX_CODE_LEN:-512}"

for required_path in \
  "${source_dir}/model.onnx" \
  "${source_dir}/model.onnx.data" \
  "${source_dir}/config.json" \
  "${audio_builder}" \
  "${plugin_path}"
do
  if [[ ! -s "${required_path}" ]]; then
    echo "ERROR: required input is missing or empty: ${required_path}" >&2
    exit 2
  fi
done

if [[ -e "${output_dir}/code2wav.engine" ]]; then
  echo "ERROR: refusing to overwrite existing engine: ${output_dir}/code2wav.engine" >&2
  exit 3
fi

mkdir -p "${output_dir}"

EDGELLM_PLUGIN_PATH="${plugin_path}" \
  "${audio_builder}" \
  --onnxDir "${source_dir}" \
  --engineDir "${output_dir}" \
  --minCodeLen "${min_code_len}" \
  --optCodeLen "${opt_code_len}" \
  --maxCodeLen "${max_code_len}"

# v0.9.1 audio_build places Code2Wav outputs in a model-type subdirectory.
# Normalize them into the deploy layout consumed by the production profiles.
if [[ ! -e "${output_dir}/code2wav.engine" ]]; then
  test -s "${output_dir}/code2wav/code2wav.engine"
  test -s "${output_dir}/code2wav/config.json"
  cp "${output_dir}/code2wav/code2wav.engine" "${output_dir}/code2wav.engine"
  cp "${output_dir}/code2wav/config.json" "${output_dir}/config.json"
fi

test -s "${output_dir}/code2wav.engine"
test -s "${output_dir}/config.json"
sha256sum "${output_dir}/code2wav.engine" "${output_dir}/config.json"
