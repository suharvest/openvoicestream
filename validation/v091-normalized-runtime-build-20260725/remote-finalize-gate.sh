#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/harvest/validation/v091-normalized-runtime-build-b364b06-20260725
LOG="${ROOT}/apply-only.log"

grep -F '==> patched source tree ready at /home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-20260725' "${LOG}"
grep -F '==> --apply-only: stopping before compile (no CUDA/TRT needed).' "${LOG}"
if grep -F 'ERROR:' "${LOG}"; then
  echo "ERROR: apply-only log contains an error" >&2
  exit 40
fi

sha256sum \
  "${ROOT}/materialize-and-gate.log" \
  "${ROOT}/DEVICE-NEGATIVE-FIXTURE.txt" \
  "${ROOT}/apply-only.log" \
  "${ROOT}/materialized-tracked-diff.sha256" \
  "${ROOT}/SOURCE-PROVENANCE.txt" \
  > "${ROOT}/gate-evidence.SHA256SUMS"
sha256sum -c "${ROOT}/gate-evidence.SHA256SUMS"
echo "PASS normalized final device provenance + apply-only"
