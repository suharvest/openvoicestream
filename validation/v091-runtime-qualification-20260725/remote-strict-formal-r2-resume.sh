#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
root="$validation/results/strict-formal-r2"
formal=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-r2-4b28dd2-20260726
build=/home/harvest/build/TensorRT-Edge-LLM-v091-formal-r2-4b28dd2-clean-20260726
artifact="$formal/cpp/kernels/cuteDSLArtifact/aarch64/sm_87"
ort=/opt/onnxruntime-linux-aarch64-1.23.2
status="$root/RESUME.status"

finish() {
  rc=$?
  printf 'rc=%s\nfinished=%s\n' "$rc" "$(date -Is)" >"$status"
}
trap finish EXIT

rm -f "$status"
test -d "$build"
test -s "$build/CMakeCache.txt"
test "$(git -C "$formal" rev-parse HEAD)" = \
  7f061f21f0a581ba234a1e233c9315b89d8e47d6
git -C "$formal" diff --check

python3 - "$artifact/metadata.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
assert d["cuda_version"] == "12.6.68", d
assert d["cutlass_dsl_version"] == "4.5.1", d
assert d["groups"] == ["gdn", "gemm", "ssd"], d
PY
test "$(sha256sum "$artifact/metadata.json" | cut -d' ' -f1)" = \
  5fd23c06136225b26ee51c0f2a8a3bdf5383a12b6f2fbad676cc15ef5f411dbf
test "$(sha256sum "$artifact/libcutedsl_aarch64.a" | cut -d' ' -f1)" = \
  2252e293801f1fd505a7a2c40ed9e58987c3c000559485147b2adb5a23f41722
(
  cd "$artifact"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$root/formal-cute-files.resume.sha256"
cmp "$root/formal-cute-files.sha256" "$root/formal-cute-files.resume.sha256"

flags="$build/cpp/CMakeFiles/edgellmCore.dir/flags.make"
grep -F 'CUTE_DSL_GDN_ENABLED' "$flags"
grep -F 'CUTE_DSL_GEMM_ENABLED' "$flags"
grep -F 'CUTE_DSL_SSD_ENABLED' "$flags"
if grep -E 'CUTE_DSL_(F16_MOE|FFPA|INT4_FP16_GEMM)' "$flags"; then
  exit 41
fi
test "$(realpath "$(sed -n 's/^MOSS_ORT_LIBRARY:FILEPATH=//p' "$build/CMakeCache.txt")")" = \
  "$(realpath "$ort/lib/libonnxruntime.so.1.23.2")"

echo "STAGE resume-clean-core-build $(date -Is)"
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
  exit 42
fi
nm -D --with-symbol-versions "$worker" >"$root/moss-worker.nm-dynamic.txt"
grep 'OrtGetApiBase@VERS_1.23.2' "$root/moss-worker.nm-dynamic.txt"
! grep -q 'OrtGetApiBase@VERS_1.20' "$root/moss-worker.nm-dynamic.txt"
env \
  LD_LIBRARY_PATH="$ort/lib:/usr/local/cuda-12.6/lib64:/usr/lib/aarch64-linux-gnu" \
  "$worker" --help >"$root/moss-worker.help.txt" 2>&1

{
  printf 'outer_head=%s\n' 6353848a646d9971c03e975ee3642ad916c0a0f8
  printf 'inner_head=%s\n' 4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f
  printf 'upstream_head=%s\n' "$(git -C "$formal" rev-parse HEAD)"
  printf 'upstream_tree=%s\n' "$(git -C "$formal" show -s --format=%T HEAD)"
  printf 'formal_diff_sha256=%s\n' \
    "$(git -C "$formal" diff --binary HEAD | sha256sum | cut -d' ' -f1)"
  printf 'worker=%s\n' "$worker"
  printf 'worker_sha256=%s\n' "$(sha256sum "$worker" | cut -d' ' -f1)"
} >"$root/PROVENANCE.txt"
sha256sum \
  "$root/APPLY-PROVENANCE.txt" \
  "$root/CUTE-LOCK.txt" \
  "$root/prelock-artifact-state.txt" \
  "$root/locked-cute-metadata.json" \
  "$root/trusted-cute-files.sha256" \
  "$root/formal-cute-files.sha256" \
  "$root/formal-cute-files.resume.sha256" \
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
