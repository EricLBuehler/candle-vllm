

#include "candle-vllm/src/scheduler/cache_engine.h"
#include "candle-vllm/src/scheduler/cache_engine.rs.h"
#include "rust/cxx.h"
#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <iostream>
// #include "candle-vllm/csrc/cache.h" // [TODO] Need to figure out how to include these

void _swap_blocks(Tensor src, Tensor dst, rust::Vec<SwapPair> _src_to_dst)
{
    std::map<int64_t, int64_t> block_mappings;
    for (auto mapping : _src_to_dst)
    {
        std::pair<int64_t, int64_t> pair = {mapping.k, mapping.v};
        block_mappings.insert(pair);
    }
    // swap_blocks(&torch_src_tensor,&torch_src_tensor,&block_mappings);
}

void _copy_blocks(rust::Vec<Tensor> key_caches, rust::Vec<Tensor> value_caches, rust::Vec<CopyPair> _src_to_dst)
{

   
    std::map<int64_t, std::vector<int64_t>> block_mappings;
    for (auto mapping : _src_to_dst)
    {
        std::vector<int64_t> value;
        for (auto v : mapping.v) {
            value.push_back(v);
        }
        std::pair<int64_t, std::vector<int64_t>> pair = {mapping.k, value};
        block_mappings.insert(pair);
    }
    // copy_blocks(&torch_key_caches,&torch_value_caches,&block_mappings);
}
