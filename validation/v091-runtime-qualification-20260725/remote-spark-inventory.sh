#!/usr/bin/env bash
set -euo pipefail

for root in \
  /home/harvest/releases \
  /home/harvest/edgellm-artifacts \
  /home/harvest/project/v090-engines \
  /home/harvest/edgellm-workspace; do
  [[ -d "$root" ]] || continue
  find "$root" -type f \
    \( -name 'spark_tts_worker' -o -name '*spark*.engine' -o -name '*spark*.plan' -o -name 'DRIVER_REVISION' \) \
    -printf '%p|%s\n'
done
