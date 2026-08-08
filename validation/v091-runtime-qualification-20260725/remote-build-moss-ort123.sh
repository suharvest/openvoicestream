#!/usr/bin/env bash
set -euo pipefail

source_root=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-d52d973-20260725
build_root=/home/harvest/build/TensorRT-Edge-LLM-v091-formal-d52d973-moss-ort123-r2-20260726
ort_root=/opt/onnxruntime-linux-aarch64-1.23.2
validation_root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
candidate_root="$validation_root/candidates/moss-ort123"
log="$validation_root/logs/moss-ort123-build.log"

test ! -e "$build_root"
test -s "$ort_root/include/onnxruntime_cxx_api.h"
test -s "$ort_root/lib/libonnxruntime.so.1.23.2"
mkdir -p "$candidate_root"

{
  printf 'source=%s\nbuild=%s\nort=%s\n' \
    "$source_root" "$build_root" "$ort_root"
  env \
    ORT_ROOT="$ort_root" \
    PATH="/usr/local/cuda-12.6/bin:$PATH" \
    CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
    CUDA_HOME=/usr/local/cuda-12.6 \
    cmake \
    -S "$source_root" \
    -B "$build_root" \
    -DCUDA_CTK_VERSION=12.6 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
    -DTRT_PACKAGE_DIR=/usr \
    -DAARCH64_BUILD=ON \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCUTE_DSL_ARTIFACT_TAG=sm_87 \
    -DENABLE_CUTE_DSL=ALL \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DMOSS_ORT_ROOT="$ort_root"
  grep -E '^MOSS_ORT_(ROOT|INCLUDE_DIR|LIBRARY):' "$build_root/CMakeCache.txt"
  test "$(realpath "$(sed -n 's/^MOSS_ORT_ROOT:PATH=//p' "$build_root/CMakeCache.txt")")" = \
    "$(realpath "$ort_root")"
  test "$(realpath "$(sed -n 's/^MOSS_ORT_LIBRARY:FILEPATH=//p' "$build_root/CMakeCache.txt")")" = \
    "$(realpath "$ort_root/lib/libonnxruntime.so.1.23.2")"
  cmake --build "$build_root" -j2 --target moss_tts_nano_worker
} 2>&1 | tee "$log"

worker="$build_root/examples/omni/moss_tts_nano_worker"
test -x "$worker"
cp --preserve=mode,timestamps "$worker" "$candidate_root/moss_tts_nano_worker"
sha256sum \
  "$candidate_root/moss_tts_nano_worker" \
  "$ort_root/include/onnxruntime_cxx_api.h" \
  "$ort_root/lib/libonnxruntime.so.1.23.2" \
  >"$candidate_root/SHA256SUMS"

env \
  LD_LIBRARY_PATH="$ort_root/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  ldd -r "$candidate_root/moss_tts_nano_worker" \
  >"$candidate_root/ldd-r.txt" 2>&1
if grep -E 'not found|undefined symbol|version .* not found' \
  "$candidate_root/ldd-r.txt"; then
  printf 'semantic ldd gate failed\n' >&2
  exit 1
fi
nm -D --with-symbol-versions "$candidate_root/moss_tts_nano_worker" \
  >"$candidate_root/nm-dynamic.txt"
grep 'OrtGetApiBase@VERS_1.23.2' "$candidate_root/nm-dynamic.txt"
if grep 'OrtGetApiBase@VERS_1.20' "$candidate_root/nm-dynamic.txt"; then
  printf 'stale ORT 1.20 symbol remains\n' >&2
  exit 1
fi
env \
  LD_LIBRARY_PATH="$ort_root/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  "$candidate_root/moss_tts_nano_worker" --help \
  >"$candidate_root/help.txt" 2>&1
printf 'candidate=%s\n' "$candidate_root/moss_tts_nano_worker"
cat "$candidate_root/SHA256SUMS"
