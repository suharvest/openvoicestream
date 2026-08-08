#!/usr/bin/env bash
set -euo pipefail

formal=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-d52d973-20260725
store=/home/harvest/project/edgellm-v091-voice-candidate-20260724T0925Z/.git/modules/3rdParty
evidence=/home/harvest/validation/v091-runtime-qualification-d52d973-20260725/results/formal-submodule-restore

mkdir -p "$evidence"

restore_one() {
  name=$1
  pin=$2
  git_dir="$store/$name"
  destination="$formal/3rdParty/$name"
  archive="$evidence/$name-$pin.tar"

  test "$(git --git-dir="$git_dir" cat-file -t "$pin")" = commit
  test -d "$destination"
  test -z "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)"

  git --git-dir="$git_dir" archive --format=tar -o "$archive" "$pin"
  test -s "$archive"
  tar -tf "$archive" >"$evidence/$name-$pin.files.txt"
  tar -xf "$archive" -C "$destination"
  sha256sum "$archive" >"$evidence/$name-$pin.tar.sha256"
  printf '%s %s %s files\n' \
    "$name" "$pin" "$(wc -l <"$evidence/$name-$pin.files.txt")"
}

restore_one nlohmannJson 22db828de4e24818599931dca17e0f111e1e895f
restore_one NVTX f71a0342a464b8580ac8573e4349086a631c3992
restore_one googletest 7917641ff965959afae189afb5f052524395525c

test -s "$formal/3rdParty/nlohmannJson/include/nlohmann/json.hpp"
test -s "$formal/3rdParty/NVTX/include/nvtx3/nvToolsExt.h"
test -s "$formal/3rdParty/googletest/googletest/include/gtest/gtest.h"

git -C "$formal" submodule status >"$evidence/submodule-status.txt"
git -C "$formal" status --short -- \
  3rdParty/nlohmannJson 3rdParty/NVTX 3rdParty/googletest \
  >"$evidence/git-status-short.txt"
test ! -s "$evidence/git-status-short.txt"

cat "$evidence/submodule-status.txt"
cat "$evidence"/*.tar.sha256
