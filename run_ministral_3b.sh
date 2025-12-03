#!/bin/bash

target/release/candle-vllm \
  --m mistralai/Ministral-3-3B-Reasoning-2512 \
  --mem 8192 \
  --kvcache-mem-cpu 2048 \
  --max-num-seqs 32 \
  --h 0.0.0.0 \
  --p 2000
