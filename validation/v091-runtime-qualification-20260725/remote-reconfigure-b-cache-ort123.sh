#!/usr/bin/env bash
set -euo pipefail

source_root=/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-no0039-20260725
build=/home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-B-no0039
ort=/opt/onnxruntime-linux-aarch64-1.23.2
evidence=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/results/moss-b-cache-fallback

test -s "$evidence/before.CMakeCache.txt"
test -s "$source_root/3rdParty/nlohmannJson/include/nlohmann/json.hpp"
test -s "$ort/include/onnxruntime_cxx_api.h"
test -s "$ort/lib/libonnxruntime.so.1.23.2"

env \
  ORT_ROOT="$ort" \
  PATH="/usr/local/cuda-12.6/bin:$PATH" \
  CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
  CUDA_HOME=/usr/local/cuda-12.6 \
  cmake \
  -S "$source_root" \
  -B "$build" \
  -DCUDA_CTK_VERSION=12.6 \
  -DCMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++ \
  -DTRT_PACKAGE_DIR=/usr \
  -DAARCH64_BUILD=ON \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCUTE_DSL_ARTIFACT_TAG=sm_87 \
  -DENABLE_CUTE_DSL=ALL \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOSS_ORT_ROOT="$ort" \
  -DMOSS_ORT_INCLUDE_DIR="$ort/include" \
  -DMOSS_ORT_LIBRARY="$ort/lib/libonnxruntime.so.1.23.2" \
  >"$evidence/reconfigure.stdout.txt" 2>"$evidence/reconfigure.stderr.txt"

cp "$build/CMakeCache.txt" "$evidence/after-configure.CMakeCache.txt"
cp \
  "$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/link.txt" \
  "$evidence/after-configure.link.txt"

grep -E '^[A-Za-z0-9_].*:' "$evidence/before.CMakeCache.txt" \
  | grep -v '^MOSS_ORT_' \
  >"$evidence/before.non-moss.CMakeCache.txt"
grep -E '^[A-Za-z0-9_].*:' "$evidence/after-configure.CMakeCache.txt" \
  | grep -v '^MOSS_ORT_' \
  >"$evidence/after.non-moss.CMakeCache.txt"
cmp \
  "$evidence/before.non-moss.CMakeCache.txt" \
  "$evidence/after.non-moss.CMakeCache.txt"

test "$(realpath "$(sed -n 's/^MOSS_ORT_ROOT:PATH=//p' "$build/CMakeCache.txt")")" = \
  "$(realpath "$ort")"
test "$(realpath "$(sed -n 's/^MOSS_ORT_INCLUDE_DIR:PATH=//p' "$build/CMakeCache.txt")")" = \
  "$(realpath "$ort/include")"
test "$(realpath "$(sed -n 's/^MOSS_ORT_LIBRARY:FILEPATH=//p' "$build/CMakeCache.txt")")" = \
  "$(realpath "$ort/lib/libonnxruntime.so.1.23.2")"
grep -F 'CUTE_DSL_GDN_ENABLED' "$build/cpp/CMakeFiles/edgellmCore.dir/flags.make"
grep -F 'CUTE_DSL_GEMM_ENABLED' "$build/cpp/CMakeFiles/edgellmCore.dir/flags.make"
grep -F 'CUTE_DSL_SSD_ENABLED' "$build/cpp/CMakeFiles/edgellmCore.dir/flags.make"
grep -F '/aarch64/sm_87/include' "$build/cpp/CMakeFiles/edgellmCore.dir/flags.make"

cmake --build "$build" --target moss_tts_nano_worker -- -n \
  >"$evidence/build-dry-run.txt" 2>&1
if grep -E \
  'gen_cubins\.py|nvcc .*generated/xqa|(^| )/usr/bin/c\+\+ .*cpp/CMakeFiles/edgellmCore\.dir/.* -c ' \
  "$evidence/build-dry-run.txt"; then
  printf 'dry run would rebuild core/CuTe; refusing\n' >&2
  exit 1
fi
if grep -Ei 'cuda.?13|extract[^ ]*13' \
  "$evidence/reconfigure.stdout.txt" \
  "$evidence/reconfigure.stderr.txt" \
  "$evidence/build-dry-run.txt"; then
  printf 'unexpected CUDA 13 extraction path; refusing\n' >&2
  exit 1
fi

diff -u \
  "$evidence/before.intent-cache.txt" \
  <(grep -E \
    '^(AARCH64_BUILD|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CUDA_CTK_VERSION|CUTE_DSL_ARTIFACT_TAG|EMBEDDED_TARGET|ENABLE_CUTE_DSL|MOSS_ORT_INCLUDE_DIR|MOSS_ORT_LIBRARY|MOSS_ORT_ROOT|TRT_PACKAGE_DIR|CMAKE_HOME_DIRECTORY):' \
    "$build/CMakeCache.txt") \
  >"$evidence/intent-cache.diff" || true

cat "$evidence/intent-cache.diff"
tail -n 40 "$evidence/build-dry-run.txt"
