#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
export BUILD_ROOT="$validation/results/r6-moss-n2-runtime-image"
export OUTER_BUNDLE="$validation/input/seeed-local-voice-b11ada3-head.bundle"
export BUILD_SOURCE=/home/harvest/project/seeed-local-voice-v091-r6-b11ada3-20260726
export BUILD_IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-r6-moss-n2-b11ada3-20260726
export PRESERVE_IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-r5-prefetch-cancel-6e83cf0-20260726
export BUILD_LOG="$validation/logs/r6-moss-n2-runtime-image-build.log"
export OUTER_SHA=5a950f97f43fda353bbccafd8bd27e3103e9d403a9fd6cc8b4ffa53b28fb314f
export WHEEL_SHA=7cb2d067ee0796f9f4ce49437242ee56b82eaf1cbd414f55ff136d6341c6490e
export OUTER_HEAD=b11ada3e8da4f6e83b06fe6320251479bd857f68
export VOXEDGE_HEAD=f738123cdef13f774b8e6c55cc32f9dca8dba8ec

exec "$validation/input/remote-build-runtime-image-helper.sh"
