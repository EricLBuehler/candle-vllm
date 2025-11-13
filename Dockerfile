# syntax=docker/dockerfile:1

FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libssl-dev \
        pkg-config \
        clang \
        libclang-dev \
        libopenmpi-dev \
        openmpi-bin && \

    rm -rf /var/lib/apt/lists/*
HEREDOC

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup update nightly
RUN rustup default nightly

# MKL build dependencies
RUN echo "deb [trusted=yes] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y libomp-dev intel-oneapi-mkl-devel

WORKDIR /candle-vllm

COPY . .

# Rayon threads are limited to minimize memory requirements in CI, avoiding OOM
# Rust threads are increased with a nightly feature for faster compilation (single-threaded by default)
ARG CUDA_COMPUTE_CAP=80
ARG RAYON_NUM_THREADS=4
ARG RUST_NUM_THREADS=4
ARG RUSTFLAGS="-Z threads=${RUST_NUM_THREADS}"
ARG WITH_FEATURES="cuda,cudnn,nccl,mkl,mpi"
RUN cargo build --release --workspace --features "${WITH_FEATURES}"

FROM docker.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "deb [trusted=yes] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list

RUN <<HEREDOC
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libomp-dev \
        ca-certificates \
        libssl-dev \
        curl \
        pkg-config \
        openmpi-bin \
        intel-oneapi-hpc-toolkit && \

    rm -rf /var/lib/apt/lists/*
HEREDOC

FROM base

COPY --from=builder /candle-vllm/target/release/candle-vllm /usr/local/bin/candle-vllm
RUN chmod +x /usr/local/bin/candle-vllm

# Only the `devel` builder image provides symlinks, restore the `libnccl.so` symlink:
RUN ln -s libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so

