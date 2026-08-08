#!/usr/bin/env bash
set -euo pipefail

source_root=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-d52d973-20260725
build_root=/home/harvest/build/TensorRT-Edge-LLM-v091-formal-d52d973-moss-ort123-r2-20260726
ort_root=/opt/onnxruntime-linux-aarch64-1.23.2
validation_root=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
candidate_root="$validation_root/candidates/moss-ort123"
log="$validation_root/logs/moss-ort123-resume.log"

test -s "$source_root/3rdParty/nlohmannJson/include/nlohmann/json.hpp"
test -s "$source_root/3rdParty/NVTX/include/nvtx3/nvToolsExt.h"
test -s "$source_root/3rdParty/googletest/googletest/include/gtest/gtest.h"
test "$(realpath "$(sed -n 's/^MOSS_ORT_ROOT:PATH=//p' "$build_root/CMakeCache.txt")")" = \
  "$(realpath "$ort_root")"
test "$(realpath "$(sed -n 's/^MOSS_ORT_LIBRARY:FILEPATH=//p' "$build_root/CMakeCache.txt")")" = \
  "$(realpath "$ort_root/lib/libonnxruntime.so.1.23.2")"

{
  printf 'resume_source=%s\nresume_build=%s\nort=%s\n' \
    "$source_root" "$build_root" "$ort_root"
  cmake --build "$build_root" -j2 --target moss_tts_nano_worker
} 2>&1 | tee "$log"

worker="$build_root/examples/omni/moss_tts_nano_worker"
test -x "$worker"
install -m 0755 "$worker" "$candidate_root/moss_tts_nano_worker"
sha256sum \
  "$candidate_root/moss_tts_nano_worker" \
  "$source_root/examples/omni/moss_tts_nano_worker.cpp" \
  "$source_root/cpp/runtime/mossTtsNanoRuntime.cpp" \
  "$build_root/cpp/libedgellmCore.a" \
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
