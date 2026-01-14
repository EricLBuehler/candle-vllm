#!/usr/bin/env bash
set -euo pipefail

WITH_FEATURES="${1:-cuda,nccl,graph,flash-attn}"
CHINA_MIRROR="${2:-0}"          # 0=off, 1=on
IMAGE_TAG="${3:-candle-vllm:latest}"
CUDA_COMPUTE_CAP="${4:-80}"                  # optional 4th arg

docker build --network=host -t "${IMAGE_TAG}" \
  --build-arg CHINA_MIRROR="${CHINA_MIRROR}" \
  --build-arg WITH_FEATURES="${WITH_FEATURES}" \
  --build-arg CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP}" \
  .



cat <<EOF

============================================================
Build finished: ${IMAGE_TAG}

China mirror mode: ${CHINA_MIRROR}
WITH_FEATURES: ${WITH_FEATURES}
CUDA_COMPUTE_CAP: ${CUDA_COMPUTE_CAP}

Binary available in the image:
  - candle-vllm

Examples:

1) Show help:
   docker run --rm --gpus all ${IMAGE_TAG} candle-vllm --help
   # Expose host file system and default running ports
   docker run --rm --gpus all -v "$HOME":/workspace -p 2000:2000 -p 1999:1999 ${IMAGE_TAG} candle-vllm --m Qwen/Qwen3-0.6B --ui-server --p 2000

2) Run interactively:
   docker run --rm -it --gpus all ${IMAGE_TAG} bash

============================================================

EOF
