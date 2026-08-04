#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy/install.sh [--target auto|jetson|orin-nx|rk3576|rk3588|rpi] [--pull] [--build] [--verify]

Examples:
  deploy/install.sh --pull --verify
  deploy/install.sh --target jetson --pull --verify
  deploy/install.sh --target orin-nx --pull --verify
  deploy/install.sh --target rk3588 --pull --verify
  deploy/install.sh --target rpi --pull --verify

Environment overrides:
  LANGUAGE_MODE, OVS_PROFILE, RK_ARTIFACT_SET,
  QWEN3_ARTIFACT_SET, QWEN3_HF_REPO_ID, OVS_PORT,
  HF_ENDPOINT, PARAFORMER_PREROLL_MS, EDGELLM_ENGINE_PROFILE,
  EDGELLM_4K_ENGINE_REVISION, EDGELLM_8K_ENGINE_REVISION
EOF
}

warn() {
  echo "WARN: $*" >&2
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

detect_target() {
  if [[ -e /etc/nv_tegra_release ]]; then
    model="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
    if [[ "${model}" == *"Orin NX"* ]]; then
      echo "orin-nx"
    else
      echo "jetson"
    fi
    return
  fi
  if [[ -e /dev/rknpu ]]; then
    if grep -qi "rk3588" /proc/cpuinfo 2>/dev/null; then
      echo "rk3588"
    else
      echo "rk3576"
    fi
    return
  fi
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' < /proc/device-tree/model | grep -qi "raspberry pi"; then
    echo "rpi"
    return
  fi
  case "$(uname -m)" in
    aarch64|arm64)
      warn "64-bit ARM host detected, but device family is ambiguous."
      ;;
  esac
  return 1
}

target="auto"
pull=0
build=0
verify=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="${2:-}"
      shift 2
      ;;
    --pull)
      pull=1
      shift
      ;;
    --build)
      build=1
      shift
      ;;
    --verify)
      verify=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$target" || "$target" == "auto" ]]; then
  if ! target="$(detect_target)"; then
    echo "Could not auto-detect target. Re-run with --target jetson|orin-nx|rk3576|rk3588|rpi." >&2
    usage >&2
    exit 2
  fi
  echo "Auto-detected target: ${target}"
fi

compose_files=()
canonical_target=""
case "$target" in
  orin-nx)
    canonical_target="orin-nx"
    case "${EDGELLM_ENGINE_PROFILE:-8k}" in
      8k)
        compose_files=(
          "deploy/docker-compose.edgellm-v091-voice.yml"
          "deploy/docker-compose.edgellm-v091-cutover.yml"
        )
        ;;
      4k)
        compose_files=(
          "deploy/docker-compose.edgellm-v091-voice.yml"
          "deploy/docker-compose.edgellm-v091-cutover-4k.yml"
        )
        ;;
      *)
        die "EDGELLM_ENGINE_PROFILE must be 4k or 8k for --target orin-nx"
        ;;
    esac
    ;;
  jetson|orin|orin-nano)
    canonical_target="jetson"
    compose_files=("deploy/docker-compose.yml")
    ;;
  rk3576|rk)
    canonical_target="rk3576"
    compose_files=("deploy/docker-compose.rk.yml")
    ;;
  rk3588|radxa)
    canonical_target="rk3588"
    compose_files=("deploy/docker-compose.radxa.yml")
    ;;
  rpi|rpi4|rpi5|cm4|cm5)
    canonical_target="rpi"
    compose_files=("deploy/docker-compose.rpi.yml")
    ;;
  *)
    echo "Unsupported target: $target" >&2
    usage >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  die "docker is required. Install Docker Engine first."
fi

if ! docker compose version >/dev/null 2>&1; then
  die "docker compose v2 is required."
fi

if ! docker info >/dev/null 2>&1; then
  die "docker daemon is not reachable. Start Docker, then rerun this command."
fi

compose_args=()
for compose_file in "${compose_files[@]}"; do
  [[ -f "$compose_file" ]] || die "missing compose file: $compose_file"
  compose_args+=( -f "$compose_file" )
done

case "$canonical_target" in
  jetson|orin-nx)
    if [[ ! -e /etc/nv_tegra_release ]]; then
      warn "this does not look like Jetson Linux; the container needs JetPack 6.x host libraries."
    fi
    if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
      warn "NVIDIA Container Runtime was not found in docker runtimes. Install nvidia-container-toolkit before starting the Jetson image."
    fi
    if [[ ! -x /usr/src/tensorrt/bin/trtexec ]]; then
      die "Jetson TensorRT trtexec is missing at /usr/src/tensorrt/bin/trtexec. Install the JetPack TensorRT samples/tools package or use a prebuilt engine artifact set."
    fi
    if [[ ! -e /usr/local/cuda/lib64/libcudla.so.1 ]]; then
      die "Jetson CUDA runtime library libcudla.so.1 is missing under /usr/local/cuda/lib64. Check the JetPack/CUDA installation before starting the slim image."
    fi
    profile="${OVS_PROFILE:-}"
    if [[ "$canonical_target" == "orin-nx" ]]; then
      echo "Jetson v0.9.1 model-level profile: ${profile:-jetson-edgellm-v091-matcha}"
      echo "Speech and Qwen3.5-4B GDN/MTP ${EDGELLM_ENGINE_PROFILE:-8k} will start as independent services."
      if [[ "${EDGELLM_ENGINE_PROFILE:-8k}" == "4k" ]]; then
        echo "4K payload SHA: 06273e358a579590bb8344b451aa35c89983cd99401339fb1858d61af4dbd107"
        echo "4K HF revision: ${EDGELLM_4K_ENGINE_REVISION:-9f2c2059341fd2135cc3a0ec09e05150277ea5b6}"
      else
        echo "8K payload SHA: 9208e46d61a4f1440ac68a312e35dde3d04b88edf0e4ee12b32210e7190d3325"
        echo "8K HF revision: ${EDGELLM_8K_ENGINE_REVISION:-adb1c78fb61513e2d7d8e7f889f6196dbefb1e5e}"
      fi
      echo "Rollback: deploy/install.sh --target jetson --pull --verify"
    elif [[ -n "${profile}" && "${profile}" == jetson-multilang-* ]]; then
      echo "Jetson Qwen3 profile: ${profile}"
      echo "Qwen3 artifacts: ${QWEN3_HF_REPO_ID:-harvestsu/qwen3-edgellm-jetson-artifacts}@${QWEN3_HF_REVISION:-main}"
    else
      echo "Jetson zh_en artifacts: harvestsu/seeed-local-voice-artifacts"
      echo "Paraformer preroll: ${PARAFORMER_PREROLL_MS:-100} ms"
    fi
    ;;
  rk3576|rk3588)
    if [[ ! -e /dev/rknpu ]]; then
      warn "/dev/rknpu is missing. Load the Rockchip NPU driver or use a kernel image with rknpu support."
    fi
    echo "RK artifact manifest: ${RK_ARTIFACT_MANIFEST:-deploy/artifacts/rk_manifest.json}"
    echo "RK artifact set: ${RK_ARTIFACT_SET:-compose default}"
    ;;
  rpi)
    arch="$(uname -m)"
    if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
      warn "Raspberry Pi images are built for 64-bit ARM; current architecture is ${arch}."
    fi
    ;;
esac

available_mb="$(df -Pm . | awk 'NR==2 {print $4}')"
if [[ -n "${available_mb}" && "${available_mb}" -lt 5120 ]]; then
  warn "less than 5 GB free under the project filesystem; first boot may fail while downloading models."
fi

echo "Target: $canonical_target"
echo "Compose: ${compose_files[*]}"
docker compose "${compose_args[@]}" config --quiet

if [[ "$pull" -eq 1 ]]; then
  docker compose "${compose_args[@]}" pull
fi

up_args=(up -d)
if [[ "$build" -eq 1 ]]; then
  up_args+=(--build)
fi

if ! docker compose "${compose_args[@]}" "${up_args[@]}"; then
  warn "container failed to start; recent logs follow"
  docker compose "${compose_args[@]}" ps >&2 || true
  docker compose "${compose_args[@]}" logs --tail=80 >&2 || true
  exit 1
fi

port="${OVS_PORT:-8621}"
url="http://127.0.0.1:${port}"

echo "Service URL: ${url}"

if [[ "$verify" -eq 1 ]]; then
  if ! deploy/verify.sh --url "${url}" --tts-smoke --roundtrip; then
    warn "verification failed; recent logs follow"
    docker compose "${compose_args[@]}" ps >&2 || true
    docker compose "${compose_args[@]}" logs --tail=120 >&2 || true
    exit 1
  fi
  if [[ "$canonical_target" == "orin-nx" ]]; then
    if ! deploy/verify-llm.sh "http://127.0.0.1:8000"; then
      warn "LLM verification failed; recent logs follow"
      docker compose "${compose_args[@]}" logs --tail=120 edge-llm >&2 || true
      exit 1
    fi
  fi
fi
