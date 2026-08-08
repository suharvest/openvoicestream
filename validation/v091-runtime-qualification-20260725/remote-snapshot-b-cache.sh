#!/usr/bin/env bash
set -euo pipefail

build=/home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-B-no0039
evidence=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/results/moss-b-cache-fallback
worker="$build/examples/omni/moss_tts_nano_worker"
worker_object="$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/moss_tts_nano_worker.cpp.o"
device_link="$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/cmake_device_link.o"
link_txt="$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/link.txt"

test ! -e "$evidence/before.CMakeCache.txt"
test -x "$worker"
test -s "$worker_object"
test -s "$device_link"
test -s "$link_txt"
mkdir -p "$evidence"

cp "$build/CMakeCache.txt" "$evidence/before.CMakeCache.txt"
cp "$link_txt" "$evidence/before.link.txt"
cp "$worker" "$evidence/before.moss_tts_nano_worker"
chmod 0755 "$evidence/before.moss_tts_nano_worker"

sha256sum \
  "$build/CMakeCache.txt" \
  "$worker" \
  "$worker_object" \
  "$device_link" \
  "$link_txt" \
  "$build/cpp/libedgellmCore.a" \
  >"$evidence/before.SHA256SUMS"

grep -E \
  '^(AARCH64_BUILD|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CUDA_CTK_VERSION|CUTE_DSL_ARTIFACT_TAG|EMBEDDED_TARGET|ENABLE_CUTE_DSL|MOSS_ORT_INCLUDE_DIR|MOSS_ORT_LIBRARY|MOSS_ORT_ROOT|TRT_PACKAGE_DIR|CMAKE_HOME_DIRECTORY):' \
  "$build/CMakeCache.txt" >"$evidence/before.intent-cache.txt"

cat "$evidence/before.intent-cache.txt"
cat "$evidence/before.SHA256SUMS"
