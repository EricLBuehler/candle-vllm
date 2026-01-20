#!/usr/bin/env bash
set -euo pipefail

# Positional args:
#   1: WITH_FEATURES   (default: cuda,nccl,graph,flash-attn,flash-decoding)
#   2: SM_ARG          (default: sm_80)  accepts sm_XX, XX, or comma list sm_80,sm_86
#   3: CUDA_VERSION    (default: 12.9.0) accepts X.Y.Z, X.Y, or shorthand like 129/124
#   4: CHINA_MIRROR    (default: 0)      0=off, 1=on
#   5: IMAGE_TAG       (default: vllm-rs:latest)

WITH_FEATURES="${1:-cuda,nccl,graph,flash-attn,flash-decoding}"
SM_ARG="${2:-sm_80}"
CUDA_VERSION_ARG="${3:-12.9.0}"
CHINA_MIRROR="${4:-0}"
IMAGE_TAG="${5:-vllm-rs:latest}"

# Optional environment override (kept as env rather than positional to avoid breaking callers)
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"

# Accept: sm_80 -> 80
# Also accept: 80 -> 80
# Optionally accept list: sm_80,sm_86 -> 80,86
normalize_sm_list() {
  local in="$1"
  local out=""
  local part

  IFS=',' read -ra parts <<< "$in"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^sm_([0-9]+)$ ]]; then
      part="${BASH_REMATCH[1]}"
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      : # already numeric
    else
      echo "ERROR: Invalid compute cap '$part'. Use sm_XX (e.g., sm_80) or XX (e.g., 80)." >&2
      exit 1
    fi
    out+="${out:+,}${part}"
  done

  echo "$out"
}

# Accept CUDA version in forms:
# - X.Y.Z (pass through)
# - X.Y   -> X.Y.0
# - 129   -> 12.9.0
# - 124   -> 12.4.0
normalize_cuda_version() {
  local v="$1"

  if [[ "$v" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
    return 0
  fi

  if [[ "$v" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.0"
    return 0
  fi

  if [[ "$v" =~ ^([0-9]{2})([0-9]{1})$ ]]; then
    echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.0"
    return 0
  fi

  echo "ERROR: Invalid CUDA version '$v'. Use X.Y.Z (e.g., 12.9.0) or X.Y (e.g., 12.9) or shorthand like 129." >&2
  exit 1
}

cuda_major() {
  local v="$1"
  echo "${v%%.*}"
}

# IMPORTANT:
# Dockerfile cannot do shell evaluation inside FROM. We precompute a flavor string and pass it as a build arg.
# Your original intent:
#   - CUDA major >= 13 => "devel"
#   - else             => "cudnn-devel"
cuda_flavor_for_version() {
  local v="$1"
  local major
  major="$(cuda_major "$v")"
  if [[ "$major" -ge 13 ]]; then
    echo "devel"
  else
    echo "cudnn-devel"
  fi
}

CUDA_COMPUTE_CAP="$(normalize_sm_list "$SM_ARG")"
CUDA_VERSION="$(normalize_cuda_version "$CUDA_VERSION_ARG")"
CUDA_FLAVOR="$(cuda_flavor_for_version "$CUDA_VERSION")"

echo "[build] IMAGE_TAG=${IMAGE_TAG}"
echo "[build] WITH_FEATURES=${WITH_FEATURES}"
echo "[build] CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} (input: ${SM_ARG})"
echo "[build] CUDA_VERSION=${CUDA_VERSION} (input: ${CUDA_VERSION_ARG})"
echo "[build] CUDA_FLAVOR=${CUDA_FLAVOR}"
echo "[build] UBUNTU_VERSION=${UBUNTU_VERSION}"
echo "[build] CHINA_MIRROR=${CHINA_MIRROR}"

docker build --network=host -t "${IMAGE_TAG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
  --build-arg CUDA_FLAVOR="${CUDA_FLAVOR}" \
  --build-arg WITH_FEATURES="${WITH_FEATURES}" \
  --build-arg CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP}" \
  --build-arg CHINA_MIRROR="${CHINA_MIRROR}" \
  .


cat <<EOF

============================================================
Build finished: ${IMAGE_TAG}

WITH_FEATURES: ${WITH_FEATURES}
CUDA_COMPUTE_CAP: ${CUDA_COMPUTE_CAP}   (input: ${SM_ARG})
CUDA_VERSION: ${CUDA_VERSION}           (input: ${CUDA_VERSION_ARG})
CUDA_FLAVOR: ${CUDA_FLAVOR}
UBUNTU_VERSION: ${UBUNTU_VERSION}

China mirror mode: ${CHINA_MIRROR}
  - 0 = disabled
  - 1 = enabled (Rustup/Cargo mirrors)

Binary available in the image:
  - candle-vllm

Examples:

1) Show help:
   docker run --rm --gpus all ${IMAGE_TAG} candle-vllm --help

2) Serving model:
   # Start the docker
   docker run --rm -it --gpus all -v /home:/home -p 2000:2000 -p 1999:1999 ${IMAGE_TAG} bash
   # Run the server (choose local access, API server: http://host_ip:2000/v1)
   candle-vllm --m Qwen/Qwen3-0.6B --p 2000

3) Run interactively:
   docker run --rm -it --gpus all ${IMAGE_TAG} bash

============================================================

EOF
