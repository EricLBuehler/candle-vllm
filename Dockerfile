FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Update default packages
RUN apt-get update

ENV DEBIAN_FRONTEND noninteractive

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    python-dev \
    git \
    curl \
    openssl \
    libssl-dev \
    pkg-config \
    wget

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
    source $HOME/.bashrc

RUN pip install setuptools && \
    python3 setup.py build
