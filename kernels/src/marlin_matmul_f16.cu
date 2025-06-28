#include "marlin_cuda_kernel.cuh"
extern "C" void marlin_4bit_f16(const void* A, const void* B, void* scales, void* zeros, void* g_idx, void* C, int prob_m, int prob_k, 
                 int prob_n, void* workspace, int groupsize, int64_t stream
                 ) {
    marlin_matmul<half, ScalarTypeID::kU4B8, false, false, 4>(A, B, scales, zeros, g_idx, C, prob_m, prob_k, prob_n, workspace, groupsize, stream);
}