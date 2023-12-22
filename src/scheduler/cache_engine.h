
#pragma once
#include "rust/cxx.h"
#include "candle-vllm/src/scheduler/cache_engine.h"
#include "candle-vllm/src/scheduler/cache_engine.rs.h"

void _swap_blocks(Tensor src, Tensor dst, rust::Vec<SwapPair> _src_to_dst);
void _copy_blocks(rust::Vec<Tensor> key_caches, rust::Vec<Tensor> value_caches, rust::Vec<CopyPair> _src_to_dst);
