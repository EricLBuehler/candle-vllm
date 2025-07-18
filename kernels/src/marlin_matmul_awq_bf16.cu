#include "marlin_cuda_kernel.cuh"
extern "C" void marlin_awq_4bit_bf16(const void* A, const void* B, void* scales, void* zeros, void* g_idx, void* C, int prob_m, int prob_k, 
                 int prob_n, void* workspace, int groupsize, int64_t stream
                 ) {
    marlin_matmul<nv_bfloat16, ScalarTypeID::kU4, false, true, 4>(A, B, scales, zeros, g_idx, C, prob_m, prob_k, prob_n, workspace, groupsize, stream);
}