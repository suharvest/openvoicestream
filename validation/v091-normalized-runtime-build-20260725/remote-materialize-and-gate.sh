#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/harvest/validation/v091-normalized-runtime-build-20260725
INPUTS="${ROOT}/inputs"
OUTER_BUNDLE="${INPUTS}/seeed-local-voice-outer-2e91b32.bundle"
INNER_BUNDLE="${INPUTS}/jetson-voice-engine-inner-6361606.bundle"
SOURCE=/home/harvest/project/seeed-local-voice-v091-normalized-20260725
INNER="${SOURCE}/third_party/jetson-voice-engine"
OVERLAY="${INNER}/engine-overlay"
MATERIALIZED=/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-20260725
OFFICIAL_WORKTREE=/home/harvest/project/edgellm-v091-upstream-six-pr-20260725
OFFICIAL=/home/harvest/project/edgellm-v091-official-object-source-20260725
LOG="${ROOT}/materialize-and-gate.log"

exec > >(tee "${LOG}") 2>&1
echo "START=$(date -Is)"
echo "HOST=$(hostname)"
echo "USER=$(id -un)"
echo "ARCH=$(uname -m)"
sha256sum "${OUTER_BUNDLE}" "${INNER_BUNDLE}"
git -C "${OFFICIAL_WORKTREE}" bundle verify "${OUTER_BUNDLE}"
git -C "${OFFICIAL_WORKTREE}" bundle verify "${INNER_BUNDLE}"

if [ -e "${MATERIALIZED}" ]; then
  echo "ERROR: refusing to replace existing ${MATERIALIZED}" >&2
  exit 21
fi

if [ ! -e "${SOURCE}" ]; then
  git clone "${OUTER_BUNDLE}" "${SOURCE}"
  if [ -d "${INNER}" ]; then
    rmdir "${INNER}"
  fi
  git clone "${INNER_BUNDLE}" "${INNER}"
else
  echo "resuming exact already-cloned source ${SOURCE}"
fi

test "$(git -C "${SOURCE}" rev-parse HEAD)" = 2e91b3245363532a7b5cf9b53fd85b334d2dec24
test "$(git -C "${INNER}" rev-parse HEAD)" = 6361606dcf590038052e20390629964d1d50a78d
test "$(git -C "${SOURCE}" rev-parse HEAD:third_party/jetson-voice-engine)" = 6361606dcf590038052e20390629964d1d50a78d
test -z "$(git -C "${SOURCE}" status --short)"
test -z "$(git -C "${INNER}" status --short)"

echo "OUTER_HEAD=$(git -C "${SOURCE}" rev-parse HEAD)"
echo "INNER_HEAD=$(git -C "${INNER}" rev-parse HEAD)"
echo "SUBMODULE_PIN=$(git -C "${SOURCE}" rev-parse HEAD:third_party/jetson-voice-engine)"
echo "UPSTREAM_PIN=$(grep -vE '^[[:space:]]*#' "${OVERLAY}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
echo "UPSTREAM_PATCH_COUNT=$(grep -cvE '^[[:space:]]*(#|$)' "${OVERLAY}/patches/upstream-v091-prs/series")"
echo "LOCAL_PATCH_COUNT=$(grep -cvE '^[[:space:]]*(#|$)' "${OVERLAY}/patches/v091-candidate/series")"

if [ ! -e "${OFFICIAL}" ]; then
  git clone --no-checkout "${OFFICIAL_WORKTREE}" "${OFFICIAL}"
fi
test -d "${OFFICIAL}/.git"
bash "${OVERLAY}/tests/verify-patch-stack.sh" "${OFFICIAL}"
bash "${OVERLAY}/tests/test-provenance-negative.sh" "${OFFICIAL}"

VOXEDGE_WORKDIR="${MATERIALIZED}" \
EDGELLM_UPSTREAM_REMOTE="${OFFICIAL}" \
  bash "${OVERLAY}/build.sh" --apply-only

git -C "${MATERIALIZED}" diff --check
test "$(git -C "${MATERIALIZED}" rev-parse HEAD)" = 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git -C "${MATERIALIZED}" diff --binary | sha256sum | tee "${ROOT}/materialized-tracked-diff.sha256"
(
  cd "${MATERIALIZED}"
  find . -type f -not -path './.git/*' -print0 |
    sort -z |
    xargs -0 sha256sum
) > "${ROOT}/materialized-files.SHA256SUMS"

{
  echo "source=${SOURCE}"
  echo "outer_head=$(git -C "${SOURCE}" rev-parse HEAD)"
  echo "inner_head=$(git -C "${INNER}" rev-parse HEAD)"
  echo "materialized=${MATERIALIZED}"
  echo "materialized_base=$(git -C "${MATERIALIZED}" rev-parse HEAD)"
  echo "materialized_status_begin"
  git -C "${MATERIALIZED}" status --short
  echo "materialized_status_end"
  echo "END=$(date -Is)"
} | tee "${ROOT}/SOURCE-PROVENANCE.txt"

sha256sum \
  "${LOG}" \
  "${ROOT}/materialized-tracked-diff.sha256" \
  "${ROOT}/materialized-files.SHA256SUMS" \
  "${ROOT}/SOURCE-PROVENANCE.txt" \
  > "${ROOT}/gate-evidence.SHA256SUMS"
sha256sum -c "${ROOT}/gate-evidence.SHA256SUMS"
echo "PASS normalized materialize-and-gate"
