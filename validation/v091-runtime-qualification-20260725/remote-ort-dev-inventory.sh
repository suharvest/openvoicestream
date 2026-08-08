#!/usr/bin/env bash
set -euo pipefail

for root in /opt /home/harvest /usr/local; do
  [[ -d "$root" ]] || continue
  find "$root" \
    \( -path '*/.git' -o -path '*/node_modules' -o -path '*/models' \) -prune -o \
    \( -name onnxruntime_cxx_api.h -o -name 'libonnxruntime.so*' \) \
    -type f -printf '%p|%s\n' 2>/dev/null || true
done

ort_root=/opt/onnxruntime-linux-aarch64-1.23.2
if [[ -f "$ort_root/lib/libonnxruntime.so.1.23.2" ]]; then
  sha256sum \
    "$ort_root/include/onnxruntime_cxx_api.h" \
    "$ort_root/lib/libonnxruntime.so.1.23.2"
  readelf -d "$ort_root/lib/libonnxruntime.so.1.23.2" |
    grep -E 'SONAME|NEEDED'
  readelf --version-info "$ort_root/lib/libonnxruntime.so.1.23.2" |
    grep -E 'VERS_1\\.(20|23)' | head -20 || true
fi

for cache in \
  /home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-B-no0039/CMakeCache.txt \
  /home/harvest/build/TensorRT-Edge-LLM-v091-formal-d52d973-20260725/CMakeCache.txt; do
  [[ -f "$cache" ]] || continue
  printf '==== %s\n' "$cache"
  grep -E 'ORT|ONNX|onnxruntime|CMAKE_HOME_DIRECTORY' "$cache" || true
done

for path in \
  /home/harvest/project/TensorRT-Edge-LLM-v091-formal-d52d973-20260725 \
  /home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-no0039-20260725 \
  /home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-B-no0039; do
  if [[ -e "$path" ]]; then
    printf 'EXISTS|%s\n' "$path"
  fi
done
