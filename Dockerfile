# syntax=docker/dockerfile:1

FROM docker.io/nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# 0=off, 1=on
ARG CHINA_MIRROR=0

# Build/runtime deps (single-stage, keep it simple)
RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends --allow-change-held-packages \
    ca-certificates \
    curl \
    libssl-dev \
    pkg-config \
    clang \
    libclang-dev; \
  rm -rf /var/lib/apt/lists/*

# Rust (stable) + optional China mirrors (SJTU for crates.io index)
RUN set -eux; \
  if [ "${CHINA_MIRROR}" = "1" ]; then \
    export RUSTUP_UPDATE_ROOT="https://mirrors.ustc.edu.cn/rust-static/rustup"; \
    export RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"; \
  fi; \
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
  if [ "${CHINA_MIRROR}" = "1" ]; then \
    mkdir -p /root/.cargo; \
    echo "RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static" >> /root/.cargo/env; \
    printf '%s\n' \
'[source.crates-io]' \
'replace-with = "ustc"' \
'' \
'[source.ustc]' \
'registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"' \
'' \
'[registries.ustc]' \
'index = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"' \
> /root/.cargo/config.toml; \
  fi

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /candle-vllm
COPY . .

# Build args (stable)
ARG CUDA_COMPUTE_CAP=80
ARG RAYON_NUM_THREADS=16
ARG WITH_FEATURES="cuda,nccl,graph"

# Make env visible to build scripts if they read it
ENV CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP}" \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS}"

# Build (no nightly flags)
RUN set -eux; \
  cargo build --release --features "${WITH_FEATURES}"; \
  install -Dm755 target/release/candle-vllm /usr/local/bin/candle-vllm

# Restore libnccl.so symlink if missing
RUN set -eux; \
  arch="$(uname -m)"; \
  libdir="/usr/lib/${arch}-linux-gnu"; \
  if [ ! -e "${libdir}/libnccl.so" ] && [ -e "${libdir}/libnccl.so.2" ]; then \
    ln -s libnccl.so.2 "${libdir}/libnccl.so"; \
  fi

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80

CMD ["bash"]
