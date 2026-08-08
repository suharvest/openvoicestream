#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/harvest/validation/v091-normalized-runtime-build-b364b06-20260725
SOURCE=/home/harvest/project/seeed-local-voice-v091-normalized-b364b06-20260725
INNER="${SOURCE}/third_party/jetson-voice-engine"
OVERLAY="${INNER}/engine-overlay"
MATERIALIZED=/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-20260725
BASE_SOURCE=/home/harvest/project/edgellm-v091-official-base-b364b06-20260725
LOG="${ROOT}/apply-only.log"

exec > >(tee "${LOG}") 2>&1
echo "START=$(date -Is)"
test -s "${ROOT}/DEVICE-NEGATIVE-FIXTURE.txt"

test "$(git -C "${SOURCE}" rev-parse HEAD)" = d6ae52b8dee9b6d8c714c153a02187deaaec6ba5
test "$(git -C "${INNER}" rev-parse HEAD)" = b364b0687dce9c72fba93192c1ef807c5881ebb9
test "$(git -C "${SOURCE}" rev-parse HEAD:third_party/jetson-voice-engine)" = b364b0687dce9c72fba93192c1ef807c5881ebb9
test ! -e "${MATERIALIZED}"

PIN="$(grep -vE '^[[:space:]]*#' "${OVERLAY}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
EXPECTED_TREE="$(git -C "${BASE_SOURCE}" show -s --format=%T "${PIN}")"
test "${EXPECTED_TREE}" = 3c3550839468342d36d57c22f09f38841b01c256

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
  "${ROOT}/materialize-and-gate.log" \
  "${ROOT}/DEVICE-NEGATIVE-FIXTURE.txt" \
  "${LOG}" \
  "${ROOT}/materialized-tracked-diff.sha256" \
  "${ROOT}/SOURCE-PROVENANCE.txt" \
  > "${ROOT}/gate-evidence.SHA256SUMS"
sha256sum -c "${ROOT}/gate-evidence.SHA256SUMS"
echo "PASS normalized final apply-only"
