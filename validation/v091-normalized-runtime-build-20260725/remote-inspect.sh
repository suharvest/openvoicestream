#!/usr/bin/env bash
set -euo pipefail

find /home/harvest/validation/edgellm-v091-voice-candidate-20260724T0925Z/input/cutedsl-sm87-cuda126 \
  -maxdepth 3 -printf '%y %p %s\n' | sort | head -200
echo META
find /home/harvest -path '*cutedsl-sm87*' -type f 2>/dev/null | head -100
echo BUILD_STATUS
if [ -f /home/harvest/validation/upstream-six-pr-20260725/build.log ]; then
  tail -80 /home/harvest/validation/upstream-six-pr-20260725/build.log
fi
