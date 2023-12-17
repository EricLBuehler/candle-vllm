
#pragma once
#include "rust/cxx.h"
#include "candle-vllm/src/scheduler/cache_engine.h"
#include "candle-vllm/src/scheduler/cache_engine.rs.h"

void _swap_blocks(rust::Vec<SwapPair> _src_to_dst);
void _copy_blocks(rust::Vec<CopyPair> _src_to_dst);
