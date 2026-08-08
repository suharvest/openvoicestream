#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
root="$validation/results/strict-formal-r2"
overlay_source=/home/harvest/project/jetson-voice-engine-v091-r2-4b28dd2
overlay="$overlay_source/engine-overlay"
formal=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-r2-4b28dd2-20260726
build=/home/harvest/build/TensorRT-Edge-LLM-v091-formal-r2-4b28dd2-clean-20260726
trusted=/home/harvest/project/edgellm-v091-upstream-six-pr-20260725/cpp/kernels/cuteDSLArtifact/aarch64/sm_87
artifact="$formal/cpp/kernels/cuteDSLArtifact/aarch64/sm_87"
rejected="$root/rejected-cuda13-sm87"
ort=/opt/onnxruntime-linux-aarch64-1.23.2
log="$validation/logs/strict-formal-r2-build.log"

test ! -e "$build"
test -d "$overlay_source/.git"
test -d "$formal/.git"
test -s "$ort/include/onnxruntime_cxx_api.h"
test -s "$ort/lib/libonnxruntime.so.1.23.2"
mkdir -p "$root" "$(dirname "$log")"
exec > >(tee "$log") 2>&1

echo "STAGE verify-passed-replay $(date -Is)"
test "$(git -C "$overlay_source" rev-parse HEAD)" = \
  4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f
test -z "$(git -C "$overlay_source" status --short)"
test "$(grep -vE '^[[:space:]]*(#|$)' "$overlay/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')" = \
  7f061f21f0a581ba234a1e233c9315b89d8e47d6
test "$(grep -cvE '^[[:space:]]*(#|$)' "$overlay/patches/upstream-v091-prs/series")" = 7
test "$(grep -cvE '^[[:space:]]*(#|$)' "$overlay/patches/v091-candidate/series")" = 35
! grep -Fq '0039-' "$overlay/patches/v091-candidate/series"

test "$(git -C "$formal" rev-parse HEAD)" = \
  7f061f21f0a581ba234a1e233c9315b89d8e47d6
git -C "$formal" diff --check
test -s "$formal/3rdParty/nlohmannJson/include/nlohmann/json.hpp"
test -s "$formal/3rdParty/NVTX/include/nvtx3/nvToolsExt.h"
test -s "$formal/3rdParty/googletest/googletest/include/gtest/gtest.h"

echo "STAGE lock-compatible-cute-artifact $(date -Is)"
test -s "$trusted/metadata.json"
test -s "$trusted/libcutedsl_aarch64.a"
test "$(sha256sum "$trusted/metadata.json" | cut -d' ' -f1)" = \
  5fd23c06136225b26ee51c0f2a8a3bdf5383a12b6f2fbad676cc15ef5f411dbf
test "$(sha256sum "$trusted/libcutedsl_aarch64.a" | cut -d' ' -f1)" = \
  2252e293801f1fd505a7a2c40ed9e58987c3c000559485147b2adb5a23f41722
python3 - "$trusted/metadata.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
assert d["cuda_version"] == "12.6.68", d
assert d["cutlass_dsl_version"] == "4.5.1", d
assert d["groups"] == ["gdn", "gemm", "ssd"], d
PY
test ! -e "$artifact"
test ! -e "$rejected"
printf 'apply_pass_tree_artifact_state=absent\n' >"$root/prelock-artifact-state.txt"
mkdir -p "$artifact"
cp -a "$trusted/." "$artifact/"
diff -qr "$trusted" "$artifact"
(
  cd "$trusted"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$root/trusted-cute-files.sha256"
(
  cd "$artifact"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$root/formal-cute-files.sha256"
cmp "$root/trusted-cute-files.sha256" "$root/formal-cute-files.sha256"
cp "$artifact/metadata.json" "$root/locked-cute-metadata.json"
{
  printf 'trusted_source=%s\n' "$trusted"
  printf 'metadata_sha256=%s\n' "$(sha256sum "$artifact/metadata.json" | cut -d' ' -f1)"
  printf 'archive_sha256=%s\n' "$(sha256sum "$artifact/libcutedsl_aarch64.a" | cut -d' ' -f1)"
  printf 'file_manifest_sha256=%s\n' \
    "$(sha256sum "$root/formal-cute-files.sha256" | cut -d' ' -f1)"
} >"$root/CUTE-LOCK.txt"
cat "$root/locked-cute-metadata.json"
cat "$root/CUTE-LOCK.txt"

echo "STAGE clean-configure $(date -Is)"
env \
  ORT_ROOT="$ort" \
  PATH="/usr/local/cuda-12.6/bin:$PATH" \
  CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
  CUDA_HOME=/usr/local/cuda-12.6 \
  cmake \
  -S "$formal" \
  -B "$build" \
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
  -DMOSS_ORT_ROOT="$ort"

flags="$build/cpp/CMakeFiles/edgellmCore.dir/flags.make"
grep '^CXX_DEFINES' "$flags" >"$root/core-defines.txt"
grep -F 'CUTE_DSL_GDN_ENABLED' "$flags"
grep -F 'CUTE_DSL_GEMM_ENABLED' "$flags"
grep -F 'CUTE_DSL_SSD_ENABLED' "$flags"
if grep -E 'CUTE_DSL_(F16_MOE|FFPA|INT4_FP16_GEMM)' "$flags"; then
  echo "ERROR: incompatible CUDA13/CuTe4.6 groups leaked into clean configure" >&2
  exit 31
fi
grep -E \
  '^(AARCH64_BUILD|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CUDA_CTK_VERSION|CUTE_DSL_ARTIFACT_TAG|EMBEDDED_TARGET|ENABLE_CUTE_DSL|MOSS_ORT_INCLUDE_DIR|MOSS_ORT_LIBRARY|MOSS_ORT_ROOT|TRT_PACKAGE_DIR|CMAKE_HOME_DIRECTORY):' \
  "$build/CMakeCache.txt" >"$root/CMake-intent.txt"

echo "STAGE clean-core-build $(date -Is)"
cmake --build "$build" -j2 --target edgellmCore

echo "STAGE clean-moss-worker-build $(date -Is)"
cmake --build "$build" -j2 --target moss_tts_nano_worker

worker="$build/examples/omni/moss_tts_nano_worker"
test -x "$worker"
sha256sum "$worker" >"$root/moss-worker.sha256"
env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  ldd -r "$worker" >"$root/moss-worker.ldd-r.txt" 2>&1
if grep -E 'not found|undefined symbol|version .* not found' "$root/moss-worker.ldd-r.txt"; then
  exit 32
fi
nm -D --with-symbol-versions "$worker" >"$root/moss-worker.nm-dynamic.txt"
grep 'OrtGetApiBase@VERS_1.23.2' "$root/moss-worker.nm-dynamic.txt"
! grep -q 'OrtGetApiBase@VERS_1.20' "$root/moss-worker.nm-dynamic.txt"
env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  "$worker" --help >"$root/moss-worker.help.txt" 2>&1

{
  printf 'overlay_head=%s\n' "$(git -C "$overlay_source" rev-parse HEAD)"
  printf 'upstream_head=%s\n' "$(git -C "$formal" rev-parse HEAD)"
  printf 'upstream_tree=%s\n' "$(git -C "$formal" show -s --format=%T HEAD)"
  printf 'formal_diff_sha256=%s\n' \
    "$(git -C "$formal" diff --binary HEAD | sha256sum | cut -d' ' -f1)"
  printf 'worker=%s\n' "$worker"
  printf 'worker_sha256=%s\n' "$(sha256sum "$worker" | cut -d' ' -f1)"
} >"$root/PROVENANCE.txt"
sha256sum \
  "$root/CUTE-LOCK.txt" \
  "$root/prelock-artifact-state.txt" \
  "$root/locked-cute-metadata.json" \
  "$root/trusted-cute-files.sha256" \
  "$root/formal-cute-files.sha256" \
  "$root/CMake-intent.txt" \
  "$root/core-defines.txt" \
  "$root/moss-worker.sha256" \
  "$root/moss-worker.ldd-r.txt" \
  "$root/moss-worker.nm-dynamic.txt" \
  "$root/moss-worker.help.txt" \
  "$root/PROVENANCE.txt" \
  >"$root/EVIDENCE.SHA256SUMS"
sha256sum -c "$root/EVIDENCE.SHA256SUMS"
cat "$root/PROVENANCE.txt"
echo "PASS strict formal r2 clean core + MOSS worker build $(date -Is)"
