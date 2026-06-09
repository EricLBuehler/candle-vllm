#!/usr/bin/env bash
set -euo pipefail

# Build candle-vllm Docker image with automatic GPU/CUDA detection.
#
# Flags:
#   --help / -h    Show usage
#
# Positional args (all optional — auto-detected if omitted):
#   1: SM_ARG        Auto-detected via nvidia-smi, or provide sm_XX / XX
#   2: CUDA_VERSION  Auto-selected from SM (13.0.0 for SM80+, 12.9.0 for SM70/75)
#   3: IMAGE_TAG     (default: candle-vllm:latest)
#   4: CHINA_MIRROR  Auto-detected (China network → 1), or 0/1 to override
#
# Features auto-selected from SM:
#   SM80+      → cuda,nccl,flashinfer,cutlass
#   SM70/SM75  → cuda,nccl (no flashinfer/cutlass)
#
# Override via environment variables:
#   WITH_FEATURES="cuda,nccl" ./build_docker.sh

usage() {
  cat <<'EOF'
Usage:
  ./build_docker.sh [SM_ARG] [CUDA_VERSION] [IMAGE_TAG] [CHINA_MIRROR]

All arguments are optional. If SM_ARG is omitted, the GPU is auto-detected
via nvidia-smi. CUDA version and features are derived from SM automatically.

SM_ARG accepts: sm_XX or XX (e.g. sm_90, 80, sm_70)
CUDA_VERSION accepts: X.Y.Z or X.Y (e.g. 13.0.0, 12.9)

Auto-detection rules:
  SM70/SM75  → CUDA 12.9.0, features: cuda,nccl
  SM80+      → CUDA 13.0.0, features: cuda,nccl,flashinfer,cutlass

Examples:
  ./build_docker.sh                           # Auto-detect GPU, CUDA, features
  ./build_docker.sh sm_90                     # SM90, auto CUDA 13.0.0
  ./build_docker.sh sm_70                     # SM70, auto CUDA 12.9.0
  ./build_docker.sh sm_80 13.0.0             # SM80, explicit CUDA 13.0.0
  ./build_docker.sh 80 13.0.0 myimg:v1 1    # SM80, custom tag, China mirrors

Override features via environment variable:
  WITH_FEATURES="cuda,nccl" ./build_docker.sh sm_90
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do POSITIONAL+=("$1"); shift; done
      ;;
    -*) echo "ERROR: Unknown flag: $1" >&2; usage >&2; exit 2 ;;
    *)  POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"

UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"

normalize_sm() {
  local v="$1"
  if [[ "$v" =~ ^sm_([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$v" =~ ^[0-9]+$ ]]; then
    echo "$v"
  else
    echo "ERROR: Invalid SM arg '$v'. Use sm_XX or XX (e.g. sm_80, 90)." >&2
    exit 1
  fi
}

detect_sm() {
  if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found and no SM version supplied." >&2
    echo "  Install NVIDIA drivers or pass SM explicitly: ./build_docker.sh sm_80" >&2
    exit 1
  fi
  local cc
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
  if [[ -z "$cc" ]]; then
    echo "ERROR: Could not detect GPU compute capability." >&2
    exit 1
  fi
  local sm="${cc/./}"
  echo "$sm"
}

SM_INPUT="${1:-}"
if [[ -n "$SM_INPUT" ]]; then
  SM_NUM="$(normalize_sm "$SM_INPUT")"
  SM_SOURCE="user"
else
  SM_NUM="$(detect_sm)"
  SM_SOURCE="auto-detected"
fi

CUDA_VERSION_INPUT="${2:-}"
IMAGE_TAG="${3:-candle-vllm:latest}"
CHINA_MIRROR_INPUT="${4:-}"

detect_china() {
  if [[ "$(timedatectl show -p Timezone --value 2>/dev/null)" == "Asia/Shanghai" ]] ||
     [[ "$(cat /etc/timezone 2>/dev/null)" == "Asia/Shanghai" ]] ||
     [[ "${TZ:-}" == "Asia/Shanghai" ]] ||
     [[ "${LANG:-}" == zh_CN* ]]; then
    echo "1"
    return
  fi
  if curl -s --connect-timeout 2 --max-time 3 -o /dev/null https://www.baidu.com 2>/dev/null; then
    if ! curl -s --connect-timeout 2 --max-time 3 -o /dev/null https://www.google.com 2>/dev/null; then
      echo "1"
      return
    fi
  fi
  echo "0"
}

if [[ -n "$CHINA_MIRROR_INPUT" ]]; then
  CHINA_MIRROR="$CHINA_MIRROR_INPUT"
  CHINA_SOURCE="user"
else
  CHINA_MIRROR="$(detect_china)"
  if [[ "$CHINA_MIRROR" == "1" ]]; then
    CHINA_SOURCE="auto-detected (China network)"
  else
    CHINA_SOURCE="auto (international)"
  fi
fi

if [[ "$SM_NUM" -lt 80 ]]; then
  DEFAULT_CUDA="12.9.0"
  DEFAULT_FEATURES="cuda,nccl"
else
  DEFAULT_CUDA="13.0.0"
  DEFAULT_FEATURES="cuda,nccl,flashinfer,cutlass"
fi

if [[ -n "$CUDA_VERSION_INPUT" ]]; then
  CUDA_VERSION="$CUDA_VERSION_INPUT"
  CUDA_SOURCE="user"
else
  CUDA_VERSION="$DEFAULT_CUDA"
  CUDA_SOURCE="auto (from SM${SM_NUM})"
fi

if [[ "$CUDA_VERSION" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
  CUDA_VERSION="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.0"
fi

WITH_FEATURES="${WITH_FEATURES:-$DEFAULT_FEATURES}"

CUDA_MAJOR="${CUDA_VERSION%%.*}"
if [[ "$CUDA_MAJOR" -ge 13 ]]; then
  CUDA_FLAVOR="devel"
else
  CUDA_FLAVOR="cudnn-devel"
fi

echo "[build] IMAGE_TAG=${IMAGE_TAG}"
echo "[build] SM=${SM_NUM} (${SM_SOURCE})"
echo "[build] CUDA_VERSION=${CUDA_VERSION} (${CUDA_SOURCE})"
echo "[build] CUDA_FLAVOR=${CUDA_FLAVOR}"
echo "[build] WITH_FEATURES=${WITH_FEATURES}"
echo "[build] UBUNTU_VERSION=${UBUNTU_VERSION}"
echo "[build] CHINA_MIRROR=${CHINA_MIRROR} (${CHINA_SOURCE})"

docker build --network=host -t "${IMAGE_TAG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
  --build-arg CUDA_FLAVOR="${CUDA_FLAVOR}" \
  --build-arg WITH_FEATURES="${WITH_FEATURES}" \
  --build-arg CUDA_COMPUTE_CAP="${SM_NUM}" \
  --build-arg CHINA_MIRROR="${CHINA_MIRROR}" \
  .

cat <<EOF

============================================================
Build finished: ${IMAGE_TAG}

SM: ${SM_NUM}  (${SM_SOURCE})
CUDA: ${CUDA_VERSION}  (${CUDA_SOURCE}, flavor: ${CUDA_FLAVOR})
Features: ${WITH_FEATURES}
Ubuntu: ${UBUNTU_VERSION}
China mirror: ${CHINA_MIRROR} (${CHINA_SOURCE})

Commands:

1) Candle-vLLM Help:
   docker run --rm -it --gpus all --network host ${IMAGE_TAG} candle-vllm --help

2) Run API server:
   docker run --rm -it --gpus all --network host ${IMAGE_TAG} candle-vllm --m Qwen/Qwen3-0.6B

3) Run UI + API Server:
    a) Run interactively:
      docker run --rm -it --gpus all --network host -v /home:/home -v /data:/data ${IMAGE_TAG} bash
    b) Start the UI + API server:
      candle-vllm --w /home/path/Qwen3-Coder-30B-A3B-Instruct-FP8 --ui-server
============================================================

EOF
