#!/usr/bin/env bash
set -euo pipefail

validation=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725
root="$validation/results/strict-formal-r2"
input="$validation/input/jetson-voice-engine-4b28dd2.bundle"
overlay_source=/home/harvest/project/jetson-voice-engine-v091-r2-4b28dd2
overlay="$overlay_source/engine-overlay"
base=/home/harvest/project/edgellm-v091-official-base-b364b06-20260725
formal=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-r2-4b28dd2-20260726
submodule_store=/home/harvest/project/edgellm-v091-voice-candidate-20260724T0925Z/.git/modules/3rdParty
log="$validation/logs/strict-formal-r2-apply.log"

test ! -e "$overlay_source"
test ! -e "$formal"
test -s "$input"
mkdir -p "$root" "$(dirname "$log")"
exec > >(tee "$log") 2>&1

test "$(sha256sum "$input" | cut -d' ' -f1)" = \
  69528abc56cc751777abac6bdcd87ab812171f580669d9ded5d86ec63712e2ec
git -C "$base" bundle verify "$input"
git clone "$input" "$overlay_source"
test "$(git -C "$overlay_source" rev-parse HEAD)" = \
  4b28dd24bcf5c5240f604b7d6be78ad39d9c5e2f
test -z "$(git -C "$overlay_source" status --short)"
test "$(grep -vE '^[[:space:]]*(#|$)' "$overlay/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')" = \
  7f061f21f0a581ba234a1e233c9315b89d8e47d6
test "$(grep -cvE '^[[:space:]]*(#|$)' "$overlay/patches/upstream-v091-prs/series")" = 7
test "$(grep -cvE '^[[:space:]]*(#|$)' "$overlay/patches/v091-candidate/series")" = 35
! grep -Fq '0039-' "$overlay/patches/v091-candidate/series"

VOXEDGE_WORKDIR="$formal" \
EDGELLM_UPSTREAM_REMOTE="$base" \
  bash "$overlay/build.sh" --apply-only
test "$(git -C "$formal" rev-parse HEAD)" = \
  7f061f21f0a581ba234a1e233c9315b89d8e47d6
git -C "$formal" diff --check

restore_submodule() {
  name=$1
  pin=$2
  destination="$formal/3rdParty/$name"
  test -d "$destination"
  test -z "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)"
  git --git-dir="$submodule_store/$name" archive "$pin" |
    tar -xf - -C "$destination"
}
restore_submodule nlohmannJson 22db828de4e24818599931dca17e0f111e1e895f
restore_submodule NVTX f71a0342a464b8580ac8573e4349086a631c3992
restore_submodule googletest 7917641ff965959afae189afb5f052524395525c
test -s "$formal/3rdParty/nlohmannJson/include/nlohmann/json.hpp"
test -s "$formal/3rdParty/NVTX/include/nvtx3/nvToolsExt.h"
test -s "$formal/3rdParty/googletest/googletest/include/gtest/gtest.h"

{
  printf 'outer_head=%s\n' 6353848a646d9971c03e975ee3642ad916c0a0f8
  printf 'inner_head=%s\n' "$(git -C "$overlay_source" rev-parse HEAD)"
  printf 'upstream_head=%s\n' "$(git -C "$formal" rev-parse HEAD)"
  printf 'upstream_tree=%s\n' "$(git -C "$formal" show -s --format=%T HEAD)"
  printf 'formal_diff_sha256=%s\n' \
    "$(git -C "$formal" diff --binary HEAD | sha256sum | cut -d' ' -f1)"
  printf 'upstream_patch_count=7\nlocal_patch_count=35\n'
} >"$root/APPLY-PROVENANCE.txt"
sha256sum "$root/APPLY-PROVENANCE.txt" >"$root/APPLY-EVIDENCE.SHA256SUMS"
sha256sum -c "$root/APPLY-EVIDENCE.SHA256SUMS"
cat "$root/APPLY-PROVENANCE.txt"
echo "PASS strict formal r2 7+35 apply gate $(date -Is)"
