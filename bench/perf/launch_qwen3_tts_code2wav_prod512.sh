#!/usr/bin/env bash
set -euo pipefail

# Launch the long-running Orin NX build independently of the Fleet SSH
# transport. Progress and the final exit code remain available after the
# initiating connection closes.

build_script="${CODE2WAV_BUILD_SCRIPT:-/home/harvest/build_qwen3_tts_code2wav_prod512.sh}"
log_file="${CODE2WAV_BUILD_LOG:-/home/harvest/validation/code2wav-prod512-build.log}"
status_file="${CODE2WAV_BUILD_STATUS:-/home/harvest/validation/code2wav-prod512-build.status}"

if [[ ! -x "${build_script}" ]]; then
  echo "ERROR: build script is missing or not executable: ${build_script}" >&2
  exit 2
fi

mkdir -p "$(dirname "${log_file}")" "$(dirname "${status_file}")"
rm -f "${status_file}"

nohup bash -c '
  set +e
  "$1" >"$2" 2>&1
  rc=$?
  printf "%s\n" "${rc}" >"$3"
  exit "${rc}"
' _ "${build_script}" "${log_file}" "${status_file}" </dev/null >/dev/null 2>&1 &

echo "$!"
