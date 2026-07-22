#!/usr/bin/env bash
# entrypoint-spark.sh — bring up the two-service voice stack on DGX Spark (GB10)
# in ONE container: edge-llm OpenAI server (:8100) in the background, then the
# voxedge voice server (:8621) in the foreground. Single-node, so the voice
# server reaches the LLM at localhost:8100 (EDGE_LLM_BASE_URL in the profile).
set -euo pipefail

UP=/work/build/upstream
LLM_LOG=/tmp/llm_server.log

echo "[entrypoint] starting edge-llm server on :8100 ..."
EDGELLM_PLUGIN_PATH="${UP}/build-gdn/libNvInfer_edgellm_plugin.so" \
EDGELLM_PYBIND_DIR="${UP}/build/pybind" \
LD_LIBRARY_PATH="${UP}/build/pybind:${UP}/build-gdn:/usr/lib/aarch64-linux-gnu" \
PYTHONPATH="${UP}" \
  python3 -u /work/build/llm_launch.py > "${LLM_LOG}" 2>&1 &

echo "[entrypoint] waiting for edge-llm readiness ..."
for i in $(seq 1 90); do
  grep -q "Uvicorn running" "${LLM_LOG}" 2>/dev/null && { echo "[entrypoint] edge-llm UP"; break; }
  if ! kill -0 %1 2>/dev/null; then echo "[entrypoint] edge-llm died:"; tail -20 "${LLM_LOG}"; exit 1; fi
  sleep 2
done

echo "[entrypoint] starting voice server on :8621 ..."
export OVS_PROFILE_JSON=/work/build/spark-customvoice.json
export LD_LIBRARY_PATH="${UP}/build:${UP}/build-gdn:/usr/lib/aarch64-linux-gnu"
export PYTHONPATH=/work/seeed
export HF_ENDPOINT=https://huggingface.co
cd /work/seeed
exec python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8621
