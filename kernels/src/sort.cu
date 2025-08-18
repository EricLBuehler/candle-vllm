#include <stdint.h>
#include <limits>
#include <type_traits>
#include "cuda_fp16.h"
#include "cuda_bf16.h"

// Custom type trait for floating point types including __half and __nv_bfloat16 ---
// This helps in selecting the correct padding value for sorting.
template<typename T> struct is_custom_fp {
    static const bool value = std::is_floating_point_v<T>;
};
template<> struct is_custom_fp<__half> { static const bool value = true; };
template<> struct is_custom_fp<__nv_bfloat16> { static const bool value = true; };

template<typename T>
inline __device__ void swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

// Bitonic Sort
// This kernel performs one step of the bitonic sort algorithm.
// It sorts a bitonic sequence in either ascending or descending order.
template <typename T, bool ascending>
__global__ void bitonic_sort_kernel(T* arr, uint32_t* dst, int j, int k) {
    // Calculate the global thread ID.
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    // Determine the index of the element to compare with.
    unsigned int ij = i ^ j;

    // Ensure the comparison is only done once by the thread with the smaller index.
    if (ij > i) {
        // Determine the direction of the sort for this stage.
        // (i & k) == 0 checks if the current thread is in the first half of a bitonic sequence of size k.
        bool sort_direction = ((i & k) == 0);

        // The comparison logic is simplified.
        // We swap if the elements are in the wrong order according to the desired global sort direction (ascending)
        // and the local sort direction for this stage (sort_direction).
        if ((arr[i] > arr[ij]) == (sort_direction == ascending)) {
            swap(arr[i], arr[ij]);
            swap(dst[i], dst[ij]);
        }
    }
}


// Calculates the next power of 2 for a given integer.
// This is necessary because the bitonic sort algorithm requires a power-of-2-sized input.
int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n <<= 1;
    }
    return n;
}

// This macro generates the extern "C" function for a specific data type and sort order.
#define ASORT_OP(T, RUST_NAME, ASC) \
extern "C" void RUST_NAME( \
    void * x1, void * dst1, const int nrows, const int ncols, bool inplace, int64_t stream \
) { \
    T* x = reinterpret_cast<T*>(x1); \
    uint32_t* dst = reinterpret_cast<uint32_t*>(dst1); \
    const cudaStream_t custream = (cudaStream_t)stream; \
    /* Padding Calculation */ \
    int ncols_pad = next_power_of_2(ncols); \
    T* x_padded; \
    uint32_t* dst_padded; \
    cudaMallocAsync((void**)&x_padded, ncols_pad * sizeof(T), custream); \
    cudaMallocAsync((void**)&dst_padded, ncols_pad * sizeof(uint32_t), custream); \
    T* values_padded = nullptr; \
    if (ncols_pad > ncols) { \
        values_padded = new T[ncols_pad - ncols]; \
        T pad_value; \
        if constexpr (ASC) { \
            /* For ascending sort, pad with the maximum value. */ \
            pad_value = std::numeric_limits<T>::max(); \
        } else { \
            /* For descending sort, pad with the minimum value. */ \
            if constexpr (is_custom_fp<T>::value) { \
                /* For float types, lowest() is the most negative value. min() is the smallest positive. */ \
                pad_value = std::numeric_limits<T>::lowest(); \
            } else { \
                /* For integer types, min() is correct. */ \
                pad_value = std::numeric_limits<T>::min(); \
            } \
        } \
        for (int i = 0; i < ncols_pad - ncols; i++) { \
            values_padded[i] = pad_value; \
        } \
        /* Copy padding values to the device once. */ \
        cudaMemcpyAsync(x_padded + ncols, values_padded, (ncols_pad - ncols) * sizeof(T), cudaMemcpyHostToDevice, custream); \
    } \
    uint32_t* indices_padded = new uint32_t[ncols_pad]; \
    for (int i = 0; i < ncols_pad; i++) { \
        indices_padded[i] = i; \
    } \
    cudaMemcpyAsync(dst_padded, indices_padded, ncols_pad * sizeof(uint32_t), cudaMemcpyHostToDevice, custream); \
    int threads_per_block = 256; \
    int blocks_per_grid = (ncols_pad + threads_per_block - 1) / threads_per_block; \
    /* Sorting Loop (per row) */ \
    for (int row = 0; row < nrows; row++) { \
        T* x_row = x + row * ncols; \
        uint32_t* dst_row = dst + row * ncols; \
        \
        /* Copy the current row's data to the padded device buffer. */ \
        cudaMemcpyAsync(x_padded, x_row, ncols * sizeof(T), cudaMemcpyDeviceToDevice, custream); \
        \
        /* Bitonic Sort Execution */ \
        for (int k = 2; k <= ncols_pad; k <<= 1) { \
            for (int j = k >> 1; j > 0; j >>= 1) { \
                bitonic_sort_kernel<T, ASC><<<blocks_per_grid, threads_per_block, 0, custream>>>(x_padded, dst_padded, j, k); \
            } \
        } \
        /* If in-place, copy the sorted data back to the original array. */ \
        if (inplace) { \
            cudaMemcpyAsync(x_row, x_padded, ncols * sizeof(T), cudaMemcpyDeviceToDevice, custream); \
        } \
        /* Copy the sorted indices back. */ \
        cudaMemcpyAsync(dst_row, dst_padded, ncols * sizeof(uint32_t), cudaMemcpyDeviceToDevice, custream); \
    } \
    cudaFreeAsync(x_padded, custream); \
    cudaFreeAsync(dst_padded, custream); \
    cudaStreamSynchronize(custream); \
    delete[] indices_padded; \
    if (values_padded) { \
        delete[] values_padded; \
    } \
}

ASORT_OP(__nv_bfloat16, asort_asc_bf16, true)
ASORT_OP(__nv_bfloat16, asort_desc_bf16, false)

ASORT_OP(__half, asort_asc_f16, true)
ASORT_OP(__half, asort_desc_f16, false)

ASORT_OP(float, asort_asc_f32, true)
ASORT_OP(float, asort_desc_f32, false)

ASORT_OP(double, asort_asc_f64, true)
ASORT_OP(double, asort_desc_f64, false)

ASORT_OP(uint8_t, asort_asc_u8, true)
ASORT_OP(uint8_t, asort_desc_u8, false)

ASORT_OP(uint32_t, asort_asc_u32, true)
ASORT_OP(uint32_t, asort_desc_u32, false)

ASORT_OP(int64_t, asort_asc_i64, true)
ASORT_OP(int64_t, asort_desc_i64, false)
