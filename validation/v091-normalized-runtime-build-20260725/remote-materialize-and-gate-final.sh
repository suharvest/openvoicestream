#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/harvest/validation/v091-normalized-runtime-build-b364b06-20260725
INPUTS="${ROOT}/inputs"
OUTER_BUNDLE="${INPUTS}/seeed-local-voice-outer-d6ae52b.bundle"
INNER_BUNDLE="${INPUTS}/jetson-voice-engine-inner-b364b06.bundle"
SOURCE=/home/harvest/project/seeed-local-voice-v091-normalized-b364b06-20260725
INNER="${SOURCE}/third_party/jetson-voice-engine"
OVERLAY="${INNER}/engine-overlay"
MATERIALIZED=/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-20260725
OBJECT_SOURCE=/home/harvest/project/edgellm-v091-upstream-six-pr-20260725
BASE_SOURCE=/home/harvest/project/edgellm-v091-official-base-b364b06-20260725
LOG="${ROOT}/materialize-and-gate.log"

exec > >(tee "${LOG}") 2>&1
echo "START=$(date -Is)"
echo "HOST=$(hostname)"
echo "USER=$(id -un)"
echo "ARCH=$(uname -m)"
sha256sum "${OUTER_BUNDLE}" "${INNER_BUNDLE}"
test "$(sha256sum "${OUTER_BUNDLE}" | awk '{print $1}')" = 2e1515682bc7fe20f202e8c139cbeade54b31ab9b3e0e7f6fe2d8f304bcba899
test "$(sha256sum "${INNER_BUNDLE}" | awk '{print $1}')" = c603cfc5e1d8d2c05cce0ae678e5937371cf9861515c0a5f75955d1a9fb1bc8b
git -C "${OBJECT_SOURCE}" bundle verify "${OUTER_BUNDLE}"
git -C "${OBJECT_SOURCE}" bundle verify "${INNER_BUNDLE}"

for path in "${SOURCE}" "${MATERIALIZED}" "${BASE_SOURCE}"; do
  if [ -e "${path}" ]; then
    echo "ERROR: refusing to replace existing ${path}" >&2
    exit 20
  fi
done

git clone "${OUTER_BUNDLE}" "${SOURCE}"
if [ -d "${INNER}" ]; then
  rmdir "${INNER}"
fi
git clone "${INNER_BUNDLE}" "${INNER}"

test "$(git -C "${SOURCE}" rev-parse HEAD)" = d6ae52b8dee9b6d8c714c153a02187deaaec6ba5
test "$(git -C "${INNER}" rev-parse HEAD)" = b364b0687dce9c72fba93192c1ef807c5881ebb9
test "$(git -C "${SOURCE}" rev-parse HEAD:third_party/jetson-voice-engine)" = b364b0687dce9c72fba93192c1ef807c5881ebb9
test -z "$(git -C "${SOURCE}" status --short)"
test -z "$(git -C "${INNER}" status --short)"

PIN="$(grep -vE '^[[:space:]]*#' "${OVERLAY}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
git init "${BASE_SOURCE}"
git -C "${BASE_SOURCE}" fetch --no-tags --depth=1 "${OBJECT_SOURCE}" "${PIN}"
git -C "${BASE_SOURCE}" update-ref refs/heads/main FETCH_HEAD
git -C "${BASE_SOURCE}" symbolic-ref HEAD refs/heads/main
test "$(git -C "${BASE_SOURCE}" rev-parse refs/heads/main)" = "${PIN}"
EXPECTED_TREE="$(git -C "${OBJECT_SOURCE}" show -s --format=%T "${PIN}")"
test "$(git -C "${BASE_SOURCE}" show -s --format=%T "${PIN}")" = "${EXPECTED_TREE}"

echo "OUTER_HEAD=$(git -C "${SOURCE}" rev-parse HEAD)"
echo "INNER_HEAD=$(git -C "${INNER}" rev-parse HEAD)"
echo "SUBMODULE_PIN=$(git -C "${SOURCE}" rev-parse HEAD:third_party/jetson-voice-engine)"
echo "UPSTREAM_PIN=${PIN}"
echo "UPSTREAM_TREE=${EXPECTED_TREE}"
echo "UPSTREAM_PATCH_COUNT=$(grep -cvE '^[[:space:]]*(#|$)' "${OVERLAY}/patches/upstream-v091-prs/series")"
echo "LOCAL_PATCH_COUNT=$(grep -cvE '^[[:space:]]*(#|$)' "${OVERLAY}/patches/v091-candidate/series")"

bash "${OVERLAY}/tests/verify-patch-stack.sh" "${OBJECT_SOURCE}"
bash "${OVERLAY}/tests/test-provenance-negative.sh" "${OBJECT_SOURCE}"

VOXEDGE_WORKDIR="${MATERIALIZED}" \
EDGELLM_UPSTREAM_REMOTE="${BASE_SOURCE}" \
  bash "${OVERLAY}/build.sh" --apply-only

git -C "${MATERIALIZED}" diff --check
test "$(git -C "${MATERIALIZED}" rev-parse HEAD)" = "${PIN}"
test "$(git -C "${MATERIALIZED}" show -s --format=%T HEAD)" = "${EXPECTED_TREE}"
git -C "${MATERIALIZED}" diff --binary | sha256sum | tee "${ROOT}/materialized-tracked-diff.sha256"

{
  echo "source=${SOURCE}"
  echo "outer_head=$(git -C "${SOURCE}" rev-parse HEAD)"
  echo "inner_head=$(git -C "${INNER}" rev-parse HEAD)"
  echo "submodule_pin=$(git -C "${SOURCE}" rev-parse HEAD:third_party/jetson-voice-engine)"
  echo "upstream_pin=${PIN}"
  echo "upstream_tree=${EXPECTED_TREE}"
  echo "materialized=${MATERIALIZED}"
  echo "materialized_base=$(git -C "${MATERIALIZED}" rev-parse HEAD)"
  echo "materialized_base_tree=$(git -C "${MATERIALIZED}" show -s --format=%T HEAD)"
  echo "materialized_status_begin"
  git -C "${MATERIALIZED}" status --short
  echo "materialized_status_end"
  echo "END=$(date -Is)"
} | tee "${ROOT}/SOURCE-PROVENANCE.txt"

sha256sum \
  "${LOG}" \
  "${ROOT}/materialized-tracked-diff.sha256" \
  "${ROOT}/SOURCE-PROVENANCE.txt" \
  > "${ROOT}/gate-evidence.SHA256SUMS"
sha256sum -c "${ROOT}/gate-evidence.SHA256SUMS"
echo "PASS normalized final materialize-and-gate"
