#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/harvest/validation/v091-formal-35patch-d52d973-20260725
INPUT="${ROOT}/input"
OUTER_BUNDLE="${INPUT}/seeed-local-voice-outer-3d0ce7e.bundle"
INNER_BUNDLE="${INPUT}/jetson-voice-engine-inner-d52d973.bundle"
SOURCE=/home/harvest/project/seeed-local-voice-v091-formal-3d0ce7e-20260725
INNER="${SOURCE}/third_party/jetson-voice-engine"
OVERLAY="${INNER}/engine-overlay"
FORMAL=/home/harvest/project/TensorRT-Edge-LLM-v091-formal-d52d973-20260725
B_SOURCE=/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-no0039-20260725
B_BUILD=/home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-B-no0039
BASE_SOURCE=/home/harvest/project/edgellm-v091-official-base-b364b06-20260725
OBJECT_SOURCE=/home/harvest/project/edgellm-v091-upstream-six-pr-20260725
PRIOR_ROOT=/home/harvest/validation/v091-normalized-runtime-build-b364b06-20260725
IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-normalized-d52d973-20260725
LOG="${ROOT}/formal-gate-image.log"

mkdir -p "${ROOT}"
exec > >(tee "${LOG}") 2>&1

service_state() {
  local output=$1
  docker inspect \
    -f '{{.Name}}|status={{.State.Status}}|health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|restart={{.RestartCount}}|image={{.Config.Image}}' \
    seeed-voice-v091 edge-llm-chat-service translator | tee "${output}"
  curl -fsS http://127.0.0.1:8621/readyz
  echo
  curl -fsS http://127.0.0.1:8000/health
  echo
  curl -fsS http://127.0.0.1:9001/health
  echo
}

remove_if_empty() {
  local path=$1
  if [ -d "${path}" ]; then
    if find "${path}" -mindepth 1 -print -quit | grep -q .; then
      echo "ERROR: refusing to replace non-empty ${path}" >&2
      exit 20
    fi
    rmdir "${path}"
  elif [ -e "${path}" ]; then
    echo "ERROR: refusing to replace non-directory ${path}" >&2
    exit 21
  fi
}

tracked_archive_sha() {
  local repo=$1
  (
    cd "${repo}"
    git ls-files -s -z |
      while IFS= read -r -d '' entry; do
        mode="${entry%% *}"
        path="${entry#*$'\t'}"
        if [ "${mode}" != 160000 ]; then
          printf '%s\0' "${path}"
        fi
      done |
      tar --null --verbatim-files-from --files-from=- --sort=name \
        --mtime=@0 --owner=0 --group=0 --numeric-owner --format=gnu -cf - |
      sha256sum |
      awk '{print $1}'
  )
}

echo "STAGE preflight $(date -Is)"
echo "HOST=$(hostname)"
echo "ARCH=$(uname -m)"
service_state "${ROOT}/services-before.txt"
sha256sum "${OUTER_BUNDLE}" "${INNER_BUNDLE}" | tee "${ROOT}/bundle-sha256.txt"
test "$(sha256sum "${OUTER_BUNDLE}" | awk '{print $1}')" = b14661b98d123fff0ac2fb68ac2eaf78c27256887af7c4294ef9a6b14b26d372
test "$(sha256sum "${INNER_BUNDLE}" | awk '{print $1}')" = b894c609d1391a56b6afc6c47d3537eb3fc4eba3c0ac27711f19b8a23b5fe9bf
git -C "${BASE_SOURCE}" bundle verify "${OUTER_BUNDLE}"
git -C "${BASE_SOURCE}" bundle verify "${INNER_BUNDLE}"
test -d "${B_SOURCE}/.git"
test -d "${B_BUILD}"
test -s "${PRIOR_ROOT}/B-ldd-r.txt"
if grep -F 'undefined symbol:' "${PRIOR_ROOT}/B-ldd-r.txt"; then
  echo "ERROR: preserved B dynamic-link evidence contains undefined symbols" >&2
  exit 22
fi
if [ ! -d "${FORMAL}/.git" ]; then
  remove_if_empty "${FORMAL}"
fi

echo "STAGE clone-formal-heads $(date -Is)"
if [ ! -d "${SOURCE}/.git" ]; then
  remove_if_empty "${SOURCE}"
  git clone "${OUTER_BUNDLE}" "${SOURCE}"
  rmdir "${INNER}"
  git clone "${INNER_BUNDLE}" "${INNER}"
fi
test "$(git -C "${SOURCE}" rev-parse HEAD)" = 3d0ce7e5f995cc569f86d6d2354ee0a675dcd3d3
test "$(git -C "${INNER}" rev-parse HEAD)" = d52d973f4a69951831ce7de0b5eee2b5ecf81006
test "$(git -C "${SOURCE}" rev-parse HEAD:third_party/jetson-voice-engine)" = d52d973f4a69951831ce7de0b5eee2b5ecf81006
test -z "$(git -C "${SOURCE}" status --short)"
test -z "$(git -C "${INNER}" status --short)"

PIN="$(grep -vE '^[[:space:]]*#' "${OVERLAY}/UPSTREAM_PIN" | head -1 | tr -d '[:space:]')"
EXPECTED_TREE="$(git -C "${BASE_SOURCE}" show -s --format=%T "${PIN}")"
test "${PIN}" = 7f061f21f0a581ba234a1e233c9315b89d8e47d6
test "${EXPECTED_TREE}" = 3c3550839468342d36d57c22f09f38841b01c256
test "$(grep -cvE '^[[:space:]]*(#|$)' "${OVERLAY}/patches/upstream-v091-prs/series")" = 7
test "$(grep -cvE '^[[:space:]]*(#|$)' "${OVERLAY}/patches/v091-candidate/series")" = 35
if grep -Fq '0039-' "${OVERLAY}/patches/v091-candidate/series"; then
  echo "ERROR: retired 0039 patch remains in formal series" >&2
  exit 23
fi

echo "STAGE formal-device-gate $(date -Is)"
bash "${OVERLAY}/tests/verify-patch-stack.sh" "${OBJECT_SOURCE}"

echo "STAGE formal-apply-only $(date -Is)"
if [ ! -d "${FORMAL}/.git" ]; then
  VOXEDGE_WORKDIR="${FORMAL}" \
  EDGELLM_UPSTREAM_REMOTE="${BASE_SOURCE}" \
    bash "${OVERLAY}/build.sh" --apply-only
else
  echo "Reusing exact formal apply-only tree from the preceding gate attempt."
fi
git -C "${FORMAL}" diff --check
test "$(git -C "${FORMAL}" rev-parse HEAD)" = "${PIN}"
test "$(git -C "${FORMAL}" show -s --format=%T HEAD)" = "${EXPECTED_TREE}"
test "$(git -C "${B_SOURCE}" rev-parse HEAD)" = "${PIN}"
test "$(git -C "${B_SOURCE}" show -s --format=%T HEAD)" = "${EXPECTED_TREE}"

echo "STAGE formal-vs-preserved-B $(date -Is)"
git -C "${FORMAL}" diff --binary HEAD > "${ROOT}/formal.diff"
git -C "${B_SOURCE}" diff --binary HEAD > "${ROOT}/preserved-B.diff"
cmp "${ROOT}/formal.diff" "${ROOT}/preserved-B.diff"
git -C "${FORMAL}" diff --name-status HEAD > "${ROOT}/formal.name-status"
git -C "${B_SOURCE}" diff --name-status HEAD > "${ROOT}/preserved-B.name-status"
cmp "${ROOT}/formal.name-status" "${ROOT}/preserved-B.name-status"
git -C "${FORMAL}" ls-files -s > "${ROOT}/formal.index"
git -C "${B_SOURCE}" ls-files -s > "${ROOT}/preserved-B.index"
cmp "${ROOT}/formal.index" "${ROOT}/preserved-B.index"
FORMAL_TRACKED_SHA="$(tracked_archive_sha "${FORMAL}")"
B_TRACKED_SHA="$(tracked_archive_sha "${B_SOURCE}")"
test "${FORMAL_TRACKED_SHA}" = "${B_TRACKED_SHA}"
{
  echo "formal_diff_sha=$(sha256sum "${ROOT}/formal.diff" | awk '{print $1}')"
  echo "preserved_B_diff_sha=$(sha256sum "${ROOT}/preserved-B.diff" | awk '{print $1}')"
  echo "formal_tracked_archive_sha=${FORMAL_TRACKED_SHA}"
  echo "preserved_B_tracked_archive_sha=${B_TRACKED_SHA}"
  echo "formal_index_sha=$(sha256sum "${ROOT}/formal.index" | awk '{print $1}')"
  echo "preserved_B_index_sha=$(sha256sum "${ROOT}/preserved-B.index" | awk '{print $1}')"
} | tee "${ROOT}/formal-vs-B.txt"

echo "STAGE collect-preserved-B-artifacts $(date -Is)"
ARTIFACTS="${ROOT}/artifacts-b-no0039"
mkdir -p "${ARTIFACTS}/bin" "${ARTIFACTS}/lib"
install -m 0755 "${B_BUILD}/examples/omni/qwen3_tts_streaming_worker" "${ARTIFACTS}/bin/"
install -m 0755 "${B_BUILD}/examples/omni/moss_tts_nano_worker" "${ARTIFACTS}/bin/"
install -m 0755 "${B_BUILD}/voice-workers/workers/qwen3_asr_worker" "${ARTIFACTS}/bin/"
install -m 0755 "${B_BUILD}/voice-workers/workers/spark_tts_worker" "${ARTIFACTS}/bin/"
install -m 0755 "${B_BUILD}/libNvInfer_edgellm_plugin.so.1.0" "${ARTIFACTS}/lib/"
(
  cd "${ARTIFACTS}"
  find ./bin ./lib -type f -print0 | sort -z | xargs -0 sha256sum
) > "${ARTIFACTS}/.SHA256SUMS.tmp"
mv "${ARTIFACTS}/.SHA256SUMS.tmp" "${ARTIFACTS}/SHA256SUMS"
(
  cd "${ARTIFACTS}"
  sha256sum -c SHA256SUMS
)

echo "STAGE build-new-runtime-image $(date -Is)"
docker build --network=host \
  -f "${SOURCE}/deploy/docker/Dockerfile.jetson.edgellm-v091-runtime" \
  -t "${IMAGE}" \
  "${SOURCE}"
docker image inspect \
  -f 'id={{.Id}}|repoTags={{json .RepoTags}}|repoDigests={{json .RepoDigests}}|size={{.Size}}|created={{.Created}}|labels={{json .Config.Labels}}' \
  "${IMAGE}" | tee "${ROOT}/runtime-image.txt"
docker run --rm --entrypoint python3 "${IMAGE}" \
  -m py_compile /opt/speech/server/core/capability_resolver.py \
  /opt/speech/server/core/worker_io.py /opt/speech/server/main.py

echo "STAGE final-health-and-evidence $(date -Is)"
service_state "${ROOT}/services-after.txt"
{
  echo "outer_head=3d0ce7e5f995cc569f86d6d2354ee0a675dcd3d3"
  echo "inner_head=d52d973f4a69951831ce7de0b5eee2b5ecf81006"
  echo "upstream_pin=${PIN}"
  echo "upstream_tree=${EXPECTED_TREE}"
  echo "formal_source=${FORMAL}"
  echo "preserved_B_source=${B_SOURCE}"
  echo "preserved_B_build=${B_BUILD}"
  echo "runtime_image=${IMAGE}"
} > "${ROOT}/SOURCE-PROVENANCE.txt"
sha256sum \
  "${ROOT}/bundle-sha256.txt" \
  "${ROOT}/formal-vs-B.txt" \
  "${ROOT}/formal.diff" \
  "${ROOT}/formal.name-status" \
  "${ROOT}/formal.index" \
  "${ROOT}/runtime-image.txt" \
  "${ROOT}/services-before.txt" \
  "${ROOT}/services-after.txt" \
  "${ROOT}/SOURCE-PROVENANCE.txt" \
  "${ARTIFACTS}/SHA256SUMS" \
  > "${ROOT}/.EVIDENCE.SHA256SUMS.tmp"
mv "${ROOT}/.EVIDENCE.SHA256SUMS.tmp" "${ROOT}/EVIDENCE.SHA256SUMS"
sha256sum -c "${ROOT}/EVIDENCE.SHA256SUMS"
echo "PASS formal 35-patch gate + B reuse + new runtime image $(date -Is)"
