

#include "candle-vllm/src/scheduler/cache_engine.h"
#include "candle-vllm/src/scheduler/cache_engine.rs.h"
#include "rust/cxx.h"
#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

void _swap_blocks(Tensor src, Tensor dst, rust::Vec<SwapPair> _src_to_dst)
{
    for (auto i : _src_to_dst)
    {
        // std::cout << i << std::endl;
    }
}

void _copy_blocks(rust::Vec<Tensor> key_caches, rust::Vec<Tensor> value_caches, rust::Vec<CopyPair> _src_to_dst)
{

    for (auto s : _src_to_dst)
    {
        // std::cout << s.k << "->" << s.v << std::endl;
    }
}
