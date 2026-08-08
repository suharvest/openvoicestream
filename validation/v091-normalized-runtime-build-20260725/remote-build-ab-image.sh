#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/harvest/validation/v091-normalized-runtime-build-b364b06-20260725
OUTER=/home/harvest/project/seeed-local-voice-v091-normalized-b364b06-20260725
OVERLAY="${OUTER}/third_party/jetson-voice-engine/engine-overlay"
SRC=/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-20260725
BSRC=/home/harvest/project/TensorRT-Edge-LLM-v091-normalized-b364b06-no0039-20260725
ABUILD=/home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-A
BBUILD=/home/harvest/build/TensorRT-Edge-LLM-v091-normalized-b364b06-B-no0039
OBJECT_SOURCE=/home/harvest/project/edgellm-v091-upstream-six-pr-20260725
MODULE_GIT_ROOT=/home/harvest/project/edgellm-v091-voice-candidate-20260724T0925Z/.git/modules
CUTE_SOURCE=/home/harvest/validation/edgellm-v091-voice-candidate-20260724T0925Z/input/cutedsl-sm87-cuda126
CUTE_DEST="${SRC}/cpp/kernels/cuteDSLArtifact/aarch64/sm_87"
VOICE_SRC="${OUTER}/third_party/jetson-voice-engine/native/edgellm_voice_worker"
IMAGE=seeed-local-voice:v0.9.1-edgellm-runtime-normalized-b364b06-20260725
LOG="${ROOT}/build-ab-image.log"
SERVICES=(seeed-voice-v091 edge-llm-chat-service translator)
RESTORED=0

restore_services() {
  if [ "${RESTORED}" -eq 1 ]; then
    return
  fi
  set +e
  echo "STAGE restore-services $(date -Is)"
  for service in "${SERVICES[@]}"; do
    docker start "${service}"
  done
  for attempt in $(seq 1 180); do
    all_ready=1
    for service in "${SERVICES[@]}"; do
      state="$(docker inspect -f '{{.State.Status}}' "${service}" 2>/dev/null)"
      health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${service}" 2>/dev/null)"
      if [ "${state}" != "running" ]; then
        all_ready=0
      fi
      if [ "${health}" != "none" ] && [ "${health}" != "healthy" ]; then
        all_ready=0
      fi
    done
    if [ "${all_ready}" -eq 1 ]; then
      break
    fi
    sleep 5
  done
  docker inspect -f '{{.Name}}|status={{.State.Status}}|health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|restart={{.RestartCount}}|image={{.Config.Image}}' "${SERVICES[@]}" |
    tee "${ROOT}/services-after.txt"
  curl -fsS http://127.0.0.1:8621/readyz
  curl -fsS http://127.0.0.1:8000/health
  curl -fsS http://127.0.0.1:9001/health
  RESTORED=1
  set -e
}
trap 'rc=$?; restore_services; exit "${rc}"' EXIT
trap 'exit 143' TERM INT

exec > >(tee "${LOG}") 2>&1
echo "START=$(date -Is)"
echo "STAGE preflight"
test "$(git -C "${OUTER}" rev-parse HEAD)" = d6ae52b8dee9b6d8c714c153a02187deaaec6ba5
test "$(git -C "${OUTER}/third_party/jetson-voice-engine" rev-parse HEAD)" = b364b0687dce9c72fba93192c1ef807c5881ebb9
test "$(git -C "${SRC}" rev-parse HEAD)" = 7f061f21f0a581ba234a1e233c9315b89d8e47d6
test -s "${ROOT}/gate-evidence.SHA256SUMS"
(
  cd "${ROOT}"
  sha256sum -c gate-evidence.SHA256SUMS
)
for path in "${BSRC}" "${ABUILD}" "${BBUILD}"; do
  if [ -e "${path}" ]; then
    echo "ERROR: refusing to replace existing ${path}" >&2
    exit 50
  fi
done
test -s "${CUTE_SOURCE}/metadata.json"
test -s "${CUTE_SOURCE}/libcutedsl_aarch64.a"
grep -F '"cuda_version": "12.6.68"' "${CUTE_SOURCE}/metadata.json"
grep -F '"cutlass_dsl_version": "4.5.1"' "${CUTE_SOURCE}/metadata.json"
test -d "${VOICE_SRC}"
test -d /home/harvest/ort-from-container
export PATH="/usr/local/cuda-12.6/bin:${PATH}"
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
export CUDA_HOME=/usr/local/cuda-12.6
export ORT_ROOT=/home/harvest/ort-from-container
nvcc --version
dpkg-query -W -f='${Package}|${Version}\n' libnvinfer10 libnvinfer-plugin10
df -h /
free -h
docker inspect -f '{{.Name}}|status={{.State.Status}}|health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|restart={{.RestartCount}}|image={{.Config.Image}}' "${SERVICES[@]}" |
  tee "${ROOT}/services-before.txt"
curl -fsS http://127.0.0.1:8621/readyz
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:9001/health

echo "STAGE stage-exact-submodules"
while read -r relative module_git; do
  expected="$(git -C "${SRC}" rev-parse "HEAD:${relative}")"
  destination="${SRC}/${relative}"
  test "$(git --git-dir="${MODULE_GIT_ROOT}/${module_git}" rev-parse "${expected}^{commit}")" = "${expected}"
  mkdir -p "${destination}"
  git --git-dir="${MODULE_GIT_ROOT}/${module_git}" archive "${expected}" | tar -x -C "${destination}"
  test -n "$(find "${destination}" -mindepth 1 -maxdepth 1 -print -quit)"
  test ! -e "${destination}/.git"
  echo "${relative}|${expected}"
done <<'SUBMODULES'
3rdParty/NVTX 3rdParty/NVTX
3rdParty/googletest 3rdParty/googletest
3rdParty/nlohmannJson 3rdParty/nlohmannJson
SUBMODULES
git -C "${SRC}" ls-tree HEAD \
  3rdParty/NVTX 3rdParty/googletest 3rdParty/nlohmannJson \
  | tee "${ROOT}/submodule-gitlinks.txt"
git -C "${SRC}" status --short | tee "${ROOT}/source-status-after-submodule-stage.txt"

echo "STAGE stage-cuda126-cute451-sm87"
if [ ! -e "${CUTE_DEST}" ]; then
  mkdir -p "${CUTE_DEST}"
  cp -a "${CUTE_SOURCE}/." "${CUTE_DEST}/"
fi
(
  cd "${CUTE_SOURCE}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${ROOT}/cute-source.SHA256SUMS"
(
  cd "${CUTE_DEST}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${ROOT}/cute-dest.SHA256SUMS"
diff -u "${ROOT}/cute-source.SHA256SUMS" "${ROOT}/cute-dest.SHA256SUMS"
grep -F '"cuda_version": "12.6.68"' "${CUTE_DEST}/metadata.json"
grep -F '"cutlass_dsl_version": "4.5.1"' "${CUTE_DEST}/metadata.json"

echo "STAGE online-static-target-preflight"
grep -R -F 'qwen3_tts_streaming_worker' "${SRC}/examples/omni/CMakeLists.txt"
grep -R -F 'moss_tts_nano_worker' "${SRC}/examples/omni/CMakeLists.txt"
grep -R -F 'qwen3_asr_worker' "${VOICE_SRC}/CMakeLists.txt"
grep -R -F 'spark_tts_worker' "${VOICE_SRC}/CMakeLists.txt"
grep -R -F 'NvInfer_edgellm_plugin' "${SRC}/cpp/CMakeLists.txt"
grep -R -F 'add_executable(llm_inference' "${SRC}/examples/llm/CMakeLists.txt"
grep -R -F 'add_executable(audio_build' "${SRC}/examples/multimodal/CMakeLists.txt"

echo "STAGE stop-services $(date -Is)"
docker stop "${SERVICES[@]}"
docker inspect -f '{{.Name}}|status={{.State.Status}}|restart={{.RestartCount}}' "${SERVICES[@]}" |
  tee "${ROOT}/services-stopped.txt"
free -h

echo "STAGE configure-A-with-0039 $(date -Is)"
cmake -S "${SRC}" -B "${ABUILD}" \
  -DCUDA_CTK_VERSION=12.6 \
  -DTRT_PACKAGE_DIR=/usr \
  -DAARCH64_BUILD=ON \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCUTE_DSL_ARTIFACT_TAG=sm_87 \
  -DENABLE_CUTE_DSL=ALL \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_VERBOSE_MAKEFILE=ON
grep -F 'CuTe DSL: using artifact' "${ABUILD}/CMakeFiles/CMakeOutput.log" 2>/dev/null || true
if grep -R -F 'extracting prebuilt' "${ABUILD}/CMakeFiles" "${ABUILD}/CMakeCache.txt"; then
  echo "ERROR: A configure attempted prebuilt extraction" >&2
  exit 51
fi

echo "STAGE build-A-main-targets $(date -Is)"
cmake --build "${ABUILD}" -j2 --target \
  NvInfer_edgellm_plugin llm_build llm_inference audio_build \
  qwen3_tts_streaming_worker moss_tts_nano_worker

echo "STAGE build-A-external-workers $(date -Is)"
cmake -S "${VOICE_SRC}" -B "${ABUILD}/voice-workers" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_CTK_VERSION=12.6 \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DEDGE_LLM_SOURCE_DIR="${SRC}" \
  -DEDGE_LLM_BUILD_DIR="${ABUILD}"
cmake --build "${ABUILD}/voice-workers" -j2 --target qwen3_asr_worker spark_tts_worker

echo "STAGE inspect-A-links"
find "${ABUILD}" -path '*/CMakeFiles/*/link.txt' -type f \
  \( -path '*NvInfer_edgellm_plugin*' -o -path '*qwen3_tts_streaming_worker*' \
     -o -path '*moss_tts_nano_worker*' -o -path '*qwen3_asr_worker*' \
     -o -path '*spark_tts_worker*' -o -path '*llm_inference*' \) \
  -print -exec sed -n '1p' {} \; > "${ROOT}/A-link-lines.txt"
grep -F -- '--wrap=_cudaLaunchKernelEx' "${ROOT}/A-link-lines.txt"
grep -F 'libcuda.so' "${ROOT}/A-link-lines.txt"

echo "STAGE prepare-B-remove-0039-temp-tree"
cp -a --reflink=auto "${SRC}" "${BSRC}"
git -C "${BSRC}" apply --reverse \
  "${OVERLAY}/patches/v091-candidate/0039-fix-cmake-propagate-CuTe-shim-driver-and-wrap-requir.patch"
if grep -F 'target_link_libraries(${_tgt} PUBLIC "${CUDA_DRIVER_LIB}")' "${BSRC}/cmake/CuteDsl.cmake"; then
  echo "ERROR: temporary B tree still contains 0039 PUBLIC driver edge" >&2
  exit 52
fi
grep -F 'target_link_libraries(${_tgt} PRIVATE "${CUDA_DRIVER_LIB}")' "${BSRC}/cmake/CuteDsl.cmake"

echo "STAGE configure-B-without-0039 $(date -Is)"
cmake -S "${BSRC}" -B "${BBUILD}" \
  -DCUDA_CTK_VERSION=12.6 \
  -DTRT_PACKAGE_DIR=/usr \
  -DAARCH64_BUILD=ON \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCUTE_DSL_ARTIFACT_TAG=sm_87 \
  -DENABLE_CUTE_DSL=ALL \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_VERBOSE_MAKEFILE=ON
if grep -R -F 'extracting prebuilt' "${BBUILD}/CMakeFiles" "${BBUILD}/CMakeCache.txt"; then
  echo "ERROR: B configure attempted prebuilt extraction" >&2
  exit 53
fi

echo "STAGE build-B-product-final-links $(date -Is)"
cmake --build "${BBUILD}" -j2 --target \
  NvInfer_edgellm_plugin llm_inference qwen3_tts_streaming_worker moss_tts_nano_worker
cmake -S "${VOICE_SRC}" -B "${BBUILD}/voice-workers" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_CTK_VERSION=12.6 \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DEDGE_LLM_SOURCE_DIR="${BSRC}" \
  -DEDGE_LLM_BUILD_DIR="${BBUILD}"
cmake --build "${BBUILD}/voice-workers" -j2 --target qwen3_asr_worker spark_tts_worker

echo "STAGE inspect-B-links"
find "${BBUILD}" -path '*/CMakeFiles/*/link.txt' -type f \
  \( -path '*NvInfer_edgellm_plugin*' -o -path '*qwen3_tts_streaming_worker*' \
     -o -path '*moss_tts_nano_worker*' -o -path '*qwen3_asr_worker*' \
     -o -path '*spark_tts_worker*' -o -path '*llm_inference*' \) \
  -print -exec sed -n '1p' {} \; > "${ROOT}/B-link-lines.txt"
grep -F -- '--wrap=_cudaLaunchKernelEx' "${ROOT}/B-link-lines.txt"
diff -u "${ROOT}/A-link-lines.txt" "${ROOT}/B-link-lines.txt" \
  > "${ROOT}/A-vs-B-link-lines.diff" || true

echo "STAGE B-dynamic-link-smoke"
find "${BBUILD}" -type f \
  \( -name qwen3_tts_streaming_worker -o -name moss_tts_nano_worker \
     -o -name qwen3_asr_worker -o -name spark_tts_worker -o -name llm_inference \
     -o -name 'libNvInfer_edgellm_plugin.so.1.0' \) \
  -print -exec ldd -r {} \; > "${ROOT}/B-ldd-r.txt"
if grep -F 'undefined symbol:' "${ROOT}/B-ldd-r.txt"; then
  echo "ERROR: B final-linked product artifact has undefined dynamic symbols" >&2
  exit 54
fi

echo "STAGE collect-A-artifacts"
mkdir -p "${ROOT}/artifacts/bin" "${ROOT}/artifacts/lib"
install -m 0755 "${ABUILD}/examples/omni/qwen3_tts_streaming_worker" "${ROOT}/artifacts/bin/"
install -m 0755 "${ABUILD}/examples/omni/moss_tts_nano_worker" "${ROOT}/artifacts/bin/"
install -m 0755 "${ABUILD}/voice-workers/workers/qwen3_asr_worker" "${ROOT}/artifacts/bin/"
install -m 0755 "${ABUILD}/voice-workers/workers/spark_tts_worker" "${ROOT}/artifacts/bin/"
install -m 0755 "${ABUILD}/libNvInfer_edgellm_plugin.so.1.0" "${ROOT}/artifacts/lib/"
(
  cd "${ROOT}/artifacts"
  find ./bin ./lib -type f -print0 | sort -z | xargs -0 sha256sum
) > "${ROOT}/artifacts/.SHA256SUMS.tmp"
mv "${ROOT}/artifacts/.SHA256SUMS.tmp" "${ROOT}/artifacts/SHA256SUMS"
(
  cd "${ROOT}/artifacts"
  sha256sum -c SHA256SUMS
)

echo "STAGE build-runtime-image $(date -Is)"
docker build --network=host \
  -f "${OUTER}/deploy/docker/Dockerfile.jetson.edgellm-v091-runtime" \
  -t "${IMAGE}" \
  "${OUTER}"
docker image inspect \
  -f 'id={{.Id}}|repoDigests={{json .RepoDigests}}|size={{.Size}}|created={{.Created}}|labels={{json .Config.Labels}}' \
  "${IMAGE}" | tee "${ROOT}/runtime-image.txt"
docker run --rm --entrypoint python3 "${IMAGE}" \
  -m py_compile /opt/speech/server/core/capability_resolver.py \
  /opt/speech/server/core/worker_io.py /opt/speech/server/main.py

echo "STAGE finalize-evidence"
df -h /
free -h
sha256sum \
  "${ROOT}/services-before.txt" \
  "${ROOT}/services-stopped.txt" \
  "${ROOT}/submodule-gitlinks.txt" \
  "${ROOT}/source-status-after-submodule-stage.txt" \
  "${ROOT}/cute-source.SHA256SUMS" \
  "${ROOT}/cute-dest.SHA256SUMS" \
  "${ROOT}/A-link-lines.txt" \
  "${ROOT}/B-link-lines.txt" \
  "${ROOT}/A-vs-B-link-lines.diff" \
  "${ROOT}/B-ldd-r.txt" \
  "${ROOT}/runtime-image.txt" \
  "${ROOT}/artifacts/SHA256SUMS" \
  > "${ROOT}/build-evidence.SHA256SUMS"
sha256sum -c "${ROOT}/build-evidence.SHA256SUMS"

restore_services
trap - EXIT
echo "PASS normalized A/B + runtime image $(date -Is)"
