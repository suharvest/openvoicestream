#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
export BUILD_ROOT="$validation/results/r5-prefetch-cancel-runtime-image"
export OUTER_BUNDLE="$validation/input/seeed-local-voice-6e83cf0-head.bundle"
export BUILD_SOURCE=/home/harvest/project/seeed-local-voice-v091-r5b-6e83cf0-20260726
export BUILD_IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-r5-prefetch-cancel-6e83cf0-20260726
export PRESERVE_IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-r4-cancel-a327ec0-20260726
export BUILD_LOG="$validation/logs/r5-prefetch-cancel-runtime-image-build.log"
export OUTER_SHA=0da9ce644a8ccd540ad8e22c2729933f98f29f150acb269d97745f2f2cd4ac8c
export WHEEL_SHA=31f82fba9e13c5cfeaced6b8027d842e2b343d07d361c8986591b6325ed03148
export OUTER_HEAD=6e83cf0584252a4aefb6cd429eff64a85be83dc6
export VOXEDGE_HEAD=fff4e47056c829b6337d596f5bb2f52c687f9de2

exec "$validation/input/remote-build-runtime-image-helper.sh"
