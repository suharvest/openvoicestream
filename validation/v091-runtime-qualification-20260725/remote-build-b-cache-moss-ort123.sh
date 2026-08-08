#!/usr/bin/env bash
set -euo pipefail

source_root=/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-no0039-20260725
formal_root=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-d52d973-20260725
build=/home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-B-no0039
ort=/opt/onnxruntime-linux-aarch64-1.23.2
validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
evidence="$validation/results/moss-b-cache-fallback"
candidate="$validation/candidates/moss-ort123"
worker_object="$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/moss_tts_nano_worker.cpp.o"
worker_dep="$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/moss_tts_nano_worker.cpp.o.d"
worker="$build/examples/omni/moss_tts_nano_worker"

test -s "$evidence/before.SHA256SUMS"
test -s "$evidence/build-dry-run.txt"
test "$(sha256sum "$source_root/examples/omni/moss_tts_nano_worker.cpp" | cut -d' ' -f1)" = \
  "$(sha256sum "$formal_root/examples/omni/moss_tts_nano_worker.cpp" | cut -d' ' -f1)"
test "$(sha256sum "$source_root/cpp/runtime/mossTtsNanoRuntime.cpp" | cut -d' ' -f1)" = \
  "$(sha256sum "$formal_root/cpp/runtime/mossTtsNanoRuntime.cpp" | cut -d' ' -f1)"

# The original object is preserved by SHA in before.SHA256SUMS. Remove only
# this exact derived object/dependency pair so CMake must rebuild it with the
# new ORT 1.23.2 include path.
if test -e "$worker_object"; then
  rm "$worker_object"
fi
if test -e "$worker_dep"; then
  rm "$worker_dep"
fi

cmake --build "$build" -j2 --target moss_tts_nano_worker \
  >"$evidence/build.stdout.txt" 2>"$evidence/build.stderr.txt"

if grep -E \
  'gen_cubins\.py|nvcc .*generated/xqa|Building (CXX|CUDA) object cpp/CMakeFiles/edgellmCore\.dir/' \
  "$evidence/build.stdout.txt" "$evidence/build.stderr.txt"; then
  printf 'unexpected core/CuTe rebuild occurred\n' >&2
  exit 1
fi

test -x "$worker"
mkdir -p "$candidate"
install -m 0755 "$worker" "$candidate/moss_tts_nano_worker"
cp \
  "$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/link.txt" \
  "$evidence/after-build.link.txt"

sha256sum \
  "$build/CMakeCache.txt" \
  "$worker" \
  "$worker_object" \
  "$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/cmake_device_link.o" \
  "$build/examples/omni/CMakeFiles/moss_tts_nano_worker.dir/link.txt" \
  "$build/cpp/libedgellmCore.a" \
  "$source_root/examples/omni/moss_tts_nano_worker.cpp" \
  "$formal_root/examples/omni/moss_tts_nano_worker.cpp" \
  "$source_root/cpp/runtime/mossTtsNanoRuntime.cpp" \
  "$formal_root/cpp/runtime/mossTtsNanoRuntime.cpp" \
  "$ort/include/onnxruntime_cxx_api.h" \
  "$ort/lib/libonnxruntime.so.1.23.2" \
  >"$evidence/after.SHA256SUMS"

before_core=$(grep '/cpp/libedgellmCore.a$' "$evidence/before.SHA256SUMS" | cut -d' ' -f1)
after_core=$(grep '/cpp/libedgellmCore.a$' "$evidence/after.SHA256SUMS" | cut -d' ' -f1)
test "$before_core" = "$after_core"
grep -F "$ort/lib/libonnxruntime.so.1.23.2" "$evidence/after-build.link.txt"

env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  ldd -r "$candidate/moss_tts_nano_worker" \
  >"$candidate/ldd-r.txt" 2>&1
if grep -E 'not found|undefined symbol|version .* not found' \
  "$candidate/ldd-r.txt"; then
  printf 'semantic ldd gate failed\n' >&2
  exit 1
fi

nm -D --with-symbol-versions "$candidate/moss_tts_nano_worker" \
  >"$candidate/nm-dynamic.txt"
grep 'OrtGetApiBase@VERS_1.23.2' "$candidate/nm-dynamic.txt"
if grep 'OrtGetApiBase@VERS_1.20' "$candidate/nm-dynamic.txt"; then
  printf 'stale ORT 1.20 symbol remains\n' >&2
  exit 1
fi

env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  "$candidate/moss_tts_nano_worker" --help \
  >"$candidate/help.txt" 2>&1

cp "$evidence/after.SHA256SUMS" "$candidate/SHA256SUMS"
printf 'candidate=%s\n' "$candidate/moss_tts_nano_worker"
cat "$candidate/SHA256SUMS"
