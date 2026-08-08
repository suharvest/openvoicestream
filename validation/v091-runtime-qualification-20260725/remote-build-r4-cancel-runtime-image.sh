#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
export BUILD_ROOT="$validation/results/r4-cancel-runtime-image"
export OUTER_BUNDLE="$validation/input/seeed-local-voice-a327ec0.bundle"
export BUILD_SOURCE=/home/harvest/project/seeed-local-voice-v091-r4-a327ec0-20260726
export BUILD_IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-r4-cancel-a327ec0-20260726
export PRESERVE_IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-r3-cvfix-ef27c98-20260726
export BUILD_LOG="$validation/logs/r4-cancel-runtime-image-build.log"
export OUTER_SHA=5e624065481aa816b410be0f605c5b00cfe768c6058854c0fe3b2183e74be5a2
export WHEEL_SHA=50f41b21a43bf8079a3969ff9be72c960fe8c8f0fbe1106553139af1efb96d63
export OUTER_HEAD=a327ec028d605ffe6b4e8494e49b372fca874aaa
export VOXEDGE_HEAD=cfbaad8134d64eb012dfec43ffa6b1773df34b0b

exec "$validation/input/remote-build-runtime-image-helper.sh"
