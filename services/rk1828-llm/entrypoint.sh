#!/usr/bin/env bash
# RK1828 LLM service entrypoint: pull model artifacts if absent, then serve.
#
# Artifacts are deliberately NOT baked into the image (3.2 GB), and are pulled
# per configuration on first start into RK1828_MODEL_DIR — mount that as a
# volume so a container replacement does not re-download.
#
# Uses plain curl rather than huggingface_hub: the files are public, and this
# keeps the image free of a dependency whose only job is one download.
set -euo pipefail

MODEL_DIR="${RK1828_MODEL_DIR:-/opt/llm/models}"
MANIFEST="${RK1828_ARTIFACT_MANIFEST:-/opt/rk1828-llm/artifacts.json}"
REPO_ID="${RK1828_ARTIFACT_REPO_ID:-harvestsu/seeed-local-voice-rk-artifacts}"
PREFIX="${RK1828_ARTIFACT_PREFIX:-rk1828/opt/llm/qwen3-4b}"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
REVISION="${RK1828_ARTIFACT_REVISION:-main}"

log() { printf '[rk1828-llm] %s\n' "$*" >&2; }

# ── preflight: the card must already be initialised by the HOST ────────────
# Failing loudly here beats a confusing MODEL_SETUP failure several minutes in.
if ! compgen -G '/dev/pcie-rkep-*' >/dev/null; then
  log "FATAL: no /dev/pcie-rkep-* device visible in this container."
  log "  The RK1828 driver and firmware live on the HOST, not in this image."
  log "  On the host, check: lspci | grep 182a ; systemctl is-active rknn3.service"
  log "  and make sure the container gets the device (privileged + /dev mount)."
  exit 1
fi

# ── artifact pull ─────────────────────────────────────────────────────────
if [ "${RK1828_ARTIFACT_AUTO_DOWNLOAD:-1}" = "1" ]; then
  if [ ! -f "${MANIFEST}" ]; then
    log "FATAL: artifact manifest ${MANIFEST} missing"
    exit 1
  fi
  mkdir -p "${MODEL_DIR}"
  # Read (name, size) pairs from the manifest without pulling in jq.
  python3 - "$MANIFEST" <<'PY' > /tmp/_artifacts.tsv
import json, sys
m = json.load(open(sys.argv[1]))
for f in m["files"]:
    print(f"{f['name']}\t{f.get('size_bytes', 0)}\t{f.get('sha256', '')}")
PY
  while IFS=$'\t' read -r name size sha; do
    dest="${MODEL_DIR}/${name}"
    # Size check, not just existence: a download interrupted halfway leaves a
    # short file that would otherwise be treated as present and then fail model
    # init with something far less obvious.
    if [ -f "${dest}" ]; then
      have=$(stat -c %s "${dest}")
      if [ "${size}" = "0" ] || [ "${have}" = "${size}" ]; then
        log "have ${name} (${have} bytes)"
        continue
      fi
      log "size mismatch for ${name}: have ${have}, want ${size} — refetching"
    fi
    url="${ENDPOINT}/${REPO_ID}/resolve/${REVISION}/${PREFIX}/${name}"
    log "fetching ${name} from ${url}"
    # Download to a temp name and move into place, so an interrupted transfer
    # can never be mistaken for a complete file on the next start.
    if ! curl -fL --retry 3 --retry-delay 5 -o "${dest}.part" "${url}"; then
      log "FATAL: download failed for ${name}"
      log "  If this host is behind the great firewall, set HF_ENDPOINT to a"
      log "  mirror (e.g. https://hf-mirror.com). Mirrors may lag the origin."
      rm -f "${dest}.part"
      exit 1
    fi
    mv "${dest}.part" "${dest}"
    if [ -n "${sha}" ] && command -v sha256sum >/dev/null 2>&1; then
      got=$(sha256sum "${dest}" | cut -d' ' -f1)
      if [ "${got}" != "${sha}" ]; then
        log "FATAL: sha256 mismatch for ${name}: got ${got}, want ${sha}"
        exit 1
      fi
      log "sha256 ok for ${name}"
    fi
  done < /tmp/_artifacts.tsv
  rm -f /tmp/_artifacts.tsv
else
  log "artifact auto-download disabled; expecting models already in ${MODEL_DIR}"
fi

# The four files are a MATCHED SET from one export — a mismatched .rknn/.weight
# pair does not load. Refuse to start on a partial set rather than emit a
# firmware-level ACK_FAIL that looks like a hardware problem.
missing=0
for f in $(python3 -c "
import json;print(' '.join(x['name'] for x in json.load(open('${MANIFEST}'))['files']))
"); do
  [ -f "${MODEL_DIR}/${f}" ] || { log "missing artifact: ${f}"; missing=1; }
done
[ "${missing}" = "0" ] || { log "FATAL: incomplete artifact set in ${MODEL_DIR}"; exit 1; }

log "serving ${RK1828_MODEL_ID:-Qwen3-4B} on ${RK1828_HOST:-0.0.0.0}:${RK1828_PORT:-1828}" \
    "(core_mask=${RK1828_CORE_MASK:-ff} max_context=${RK1828_MAX_CONTEXT:-8192})"

exec /opt/venv/bin/python /opt/rk1828-llm/rk1828_llm_server.py \
  --binary /opt/rk1828/rknn_qwen3_demo \
  --model-dir "${MODEL_DIR}" \
  --core-mask "${RK1828_CORE_MASK:-ff}" \
  --max-context "${RK1828_MAX_CONTEXT:-8192}" \
  --host "${RK1828_HOST:-0.0.0.0}" \
  --port "${RK1828_PORT:-1828}"
