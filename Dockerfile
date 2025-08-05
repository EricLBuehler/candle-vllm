# syntax=docker/dockerfile:1

FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    apt-get update
    apt-get install -y --no-install-recommends \
        curl \
        libssl-dev \
        pkg-config \
        clang \
        libclang-dev \
        libopenmpi-dev \
        openmpi-bin

    rm -rf /var/lib/apt/lists/*
HEREDOC

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup update nightly
RUN rustup default nightly

### Add support for the MKL feature ###
# 1. Add the upstream package repository for Intel oneAPI:
COPY <<HEREDOC /etc/apt/sources.list.d/upstream-intel-oneapi.sources
Types: deb
URIs: https://apt.repos.intel.com/oneapi
Suites: all
Components: main
Signed-By: /usr/share/keyrings/upstream-intel-oneapi.gpg
HEREDOC

# 2. Install required packages:
RUN <<HEREDOC
    # Add the associated package signing key to verify trust:
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor > /usr/share/keyrings/upstream-intel-oneapi.gpg

    # Refresh the package index and install packages + cleanup:
    apt-get -qq update
    apt-get -qq install --no-install-recommends \
        libomp-dev \
        intel-oneapi-hpc-toolkit

    rm -rf /var/lib/apt/lists/*
HEREDOC

WORKDIR /candle-vllm
COPY . .

# Rayon threads are limited to minimize memory requirements in CI, avoiding OOM
# Rust threads are increased with a nightly feature for faster compilation (single-threaded by default)
ARG CUDA_COMPUTE_CAP=70
ARG RAYON_NUM_THREADS=4
ARG RUST_NUM_THREADS=4
ARG RUSTFLAGS="-Z threads=${RUST_NUM_THREADS}"
ARG WITH_FEATURES="cuda,cudnn,nccl,mkl,mpi"
RUN cargo build --release --workspace --features "${WITH_FEATURES}"

FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base
ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80

ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    apt-get -qq update

    # NOTE: `openmpi-bin` is only provided for convenience when using the NCCL crate feature.
    apt-get -qq install --no-install-recommends \
        libomp-dev \
        ca-certificates \
        libssl-dev \
        curl \
        pkg-config \
        openmpi-bin

    rm -rf /var/lib/apt/lists/*
HEREDOC

# Add runtime support for the MKL feature:
COPY --from builder /etc/apt/sources.list.d/upstream-intel-oneapi.sources /etc/apt/sources.list.d/upstream-intel-oneapi.sources
COPY --from builder /usr/share/keyrings/upstream-intel-oneapi.gpg /usr/share/keyrings/upstream-intel-oneapi.gpg
RUN <<HEREDOC
    apt-get -qq update
    apt-get -qq install --no-install-recommends \
      intel-oneapi-hpc-toolkit

    rm -rf /var/lib/apt/lists/*
HEREDOC

FROM base

COPY --from=builder /candle-vllm/target/release/candle-vllm /usr/local/bin/candle-vllm
RUN chmod +x /usr/local/bin/candle-vllm

# Only the `devel` builder image provides symlinks, restore the `libnccl.so` symlink:
RUN ln -s libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so

