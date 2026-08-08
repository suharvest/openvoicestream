#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
out="$validation/results/fresh-formal-cute-comparison"
formal_source=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-d52d973-20260725
formal_build=/home/harvest/build/TensorRT-Edge-LLM-v091-formal-d52d973-moss-ort123-r2-20260726
six_source=/home/harvest/project/edgellm-v091-upstream-six-pr-20260725
six_build=/home/harvest/build/edgellm-v091-upstream-six-pr-20260725
formal_artifact="$formal_source/cpp/kernels/cuteDSLArtifact/aarch64/sm_87"
six_artifact="$six_source/cpp/kernels/cuteDSLArtifact/aarch64/sm_87"
failure_log="$validation/logs/moss-ort123-resume.log"

mkdir -p "$out"

{
  printf 'formal_head=%s\n' "$(git -C "$formal_source" rev-parse HEAD)"
  printf 'formal_tree=%s\n' "$(git -C "$formal_source" show -s --format=%T HEAD)"
  printf 'formal_diff_sha=%s\n' \
    "$(git -C "$formal_source" diff --binary HEAD | sha256sum | cut -d' ' -f1)"
  printf 'six_head=%s\n' "$(git -C "$six_source" rev-parse HEAD)"
  printf 'six_tree=%s\n' "$(git -C "$six_source" show -s --format=%T HEAD)"
  printf 'formal_source=%s\nformal_build=%s\n' "$formal_source" "$formal_build"
  printf 'six_source=%s\nsix_build=%s\n' "$six_source" "$six_build"
} >"$out/source-heads.txt"

git -C "$formal_source" status --short >"$out/formal-status.txt"
git -C "$formal_source" diff --name-status HEAD >"$out/formal-name-status.txt"
git -C "$six_source" status --short >"$out/six-status.txt"
test ! -s "$out/six-status.txt"

cp "$formal_artifact/metadata.json" "$out/formal-sm87-metadata.json"
cp "$six_artifact/metadata.json" "$out/six-sm87-metadata.json"
sha256sum \
  "$formal_artifact/metadata.json" \
  "$formal_artifact/libcutedsl_aarch64.a" \
  "$formal_artifact/include/f16_moe_ampere_grouped_fp16.h" \
  "$formal_source/cmake/CuteDsl.cmake" \
  >"$out/formal-artifact-SHA256SUMS"
sha256sum \
  "$six_artifact/metadata.json" \
  "$six_artifact/libcutedsl_aarch64.a" \
  "$six_source/cmake/CuteDsl.cmake" \
  >"$out/six-artifact-SHA256SUMS"

grep -E \
  '^(AARCH64_BUILD|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CUDA_CTK_VERSION|CUTE_DSL_ARTIFACT_TAG|EMBEDDED_TARGET|ENABLE_CUTE_DSL|TRT_PACKAGE_DIR|CMAKE_HOME_DIRECTORY):' \
  "$formal_build/CMakeCache.txt" >"$out/formal-intent-cache.txt"
grep -E \
  '^(AARCH64_BUILD|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CUDA_CTK_VERSION|CUTE_DSL_ARTIFACT_TAG|EMBEDDED_TARGET|ENABLE_CUTE_DSL|TRT_PACKAGE_DIR|CMAKE_HOME_DIRECTORY):' \
  "$six_build/CMakeCache.txt" >"$out/six-intent-cache.txt"
grep '^CXX_DEFINES' \
  "$formal_build/cpp/CMakeFiles/edgellmCore.dir/flags.make" \
  >"$out/formal-core-defines.txt"
grep '^CXX_INCLUDES' \
  "$formal_build/cpp/CMakeFiles/edgellmCore.dir/flags.make" \
  >"$out/formal-core-includes.txt"
grep '^CXX_DEFINES' \
  "$six_build/cpp/CMakeFiles/edgellmCore.dir/flags.make" \
  >"$out/six-core-defines.txt"
grep '^CXX_INCLUDES' \
  "$six_build/cpp/CMakeFiles/edgellmCore.dir/flags.make" \
  >"$out/six-core-includes.txt"

grep -F 'cuteDslF16MoeRunner.cpp.o -c ' "$failure_log" \
  | tail -1 >"$out/failing-compile-command.txt"
grep -n -m1 'error:' "$failure_log" >"$out/first-error.txt"
test -s "$out/failing-compile-command.txt"
test -s "$out/first-error.txt"

{
  printf 'formal artifact: '
  python3 -c \
    'import json; d=json.load(open("'"$formal_artifact"'/metadata.json")); print(d["cuda_version"], d["cutlass_dsl_version"], ",".join(d["groups"]))'
  printf 'six artifact: '
  python3 -c \
    'import json; d=json.load(open("'"$six_artifact"'/metadata.json")); print(d["cuda_version"], d["cutlass_dsl_version"], ",".join(d["groups"]))'
  printf 'first error: '
  cat "$out/first-error.txt"
} >"$out/SUMMARY.txt"

sha256sum "$out"/*.txt "$out"/*.json >"$out/EVIDENCE.SHA256SUMS"
cat "$out/SUMMARY.txt"
cat "$out/source-heads.txt"
cat "$out/formal-artifact-SHA256SUMS"
cat "$out/six-artifact-SHA256SUMS"
