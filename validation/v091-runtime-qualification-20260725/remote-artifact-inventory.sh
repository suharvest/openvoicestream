#!/usr/bin/env bash
set -euo pipefail

artifact_root=/home/harvest/edgellm-artifacts/orin-nx-edgellm-v091-jp62-trt103-sm87-20260725/v091
for relative in \
  engines/moss \
  engines/moss/codec \
  engines/spark \
  engines/sparktts-w4a16 \
  engines/sparktts-bf16 \
  engines/tts_customvoice_int4 \
  engines/tts_customvoice_fp16; do
  printf '==== %s\n' "$relative"
  if [[ -d "$artifact_root/$relative" ]]; then
    find "$artifact_root/$relative" -maxdepth 2 -type f -printf '%P|%s\n' | sort
  else
    printf 'MISSING\n'
  fi
done
