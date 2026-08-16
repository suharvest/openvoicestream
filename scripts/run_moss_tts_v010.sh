#!/bin/sh
set -eu

# MOSS was qualified against ORT 1.20.0. Keep that ABI scoped to the MOSS
# subprocess so the server and other backends may continue using the base
# image's newer Python ONNX Runtime without symbol-version cross-contamination.
MOSS_ORT_ROOT=/opt/edgellm-v010/moss-runtime
export LD_LIBRARY_PATH="${MOSS_ORT_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec /opt/edgellm-v010/bin/moss_tts_nano_worker "$@"
