#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/harvest/validation/v091-normalized-runtime-build-20260725
SOURCE=/home/harvest/project/edgellm-v091-upstream-six-pr-20260725
OVERLAY=/home/harvest/project/seeed-local-voice-v091-normalized-20260725/third_party/jetson-voice-engine/engine-overlay
EXACT=/home/harvest/project/edgellm-v091-exact-object-source-20260725
PROBE=/tmp/edgellm-v091-exact-object-source-probe-20260725
LOG="${ROOT}/exact-object-source.log"

exec > >(tee "${LOG}") 2>&1
if [ -e "${EXACT}" ]; then
  echo "ERROR: refusing to replace ${EXACT}" >&2
  exit 30
fi
if [ -e "${PROBE}" ]; then
  echo "ERROR: refusing to replace ${PROBE}" >&2
  exit 31
fi

git init "${EXACT}"
git -C "${EXACT}" remote add source "${SOURCE}"
PIN="$(grep -vE '^[[:space:]]*#' "${OVERLAY}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
git -C "${EXACT}" fetch --no-tags --depth=1 source "${PIN}"
git -C "${EXACT}" update-ref refs/heads/base FETCH_HEAD

index=0
while IFS='|' read -r file pr commit parent tree patch_id expected_sha; do
  case "${file}" in ""|\#*) continue ;; esac
  index=$((index + 1))
  git -C "${EXACT}" fetch --no-tags --depth=1 source "${commit}"
  git -C "${EXACT}" update-ref "refs/heads/lock-${index}" FETCH_HEAD
done < "${OVERLAY}/patches/upstream-v091-prs/LOCK"
test "${index}" -eq 7
git -C "${EXACT}" symbolic-ref HEAD refs/heads/base

git clone --no-local --no-checkout "${EXACT}" "${PROBE}"
git -C "${PROBE}" cat-file -e "${PIN}^{commit}"
while IFS='|' read -r file pr commit parent tree patch_id expected_sha; do
  case "${file}" in ""|\#*) continue ;; esac
  git -C "${PROBE}" cat-file -e "${commit}^{commit}"
  test "$(git -C "${PROBE}" show -s --format=%P "${commit}")" = "${parent}"
  test "$(git -C "${PROBE}" show -s --format=%T "${commit}")" = "${tree}"
done < "${OVERLAY}/patches/upstream-v091-prs/LOCK"

git -C "${EXACT}" show-ref | sort
git -C "${EXACT}" count-objects -v
sha256sum "${LOG}" > "${ROOT}/exact-object-source.log.sha256"
echo "PASS exact minimal object source"
