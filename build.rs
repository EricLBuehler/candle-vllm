extern crate cc;
use std::env;

fn main() {
    /*if let Ok(cuda_path) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");*/

    /*
    extra_compile_args {'cxx': ['-g', '-O2', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
        'nvcc': ['-O2', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0', '-gencode', 'arch=compute_86,code=sm_86']}
     */

    cc::Build::new().cuda(true)
        .include("/home/ericbuehler/.local/lib/python3.10/site-packages/torch/include")
        .include("/home/ericbuehler/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include")
        .include("/usr/include/python3.10")
        .flag("-gencode").flag("arch=compute_86,code=sm_86")
        .flag("-D_GLIBCXX_USE_CXX11_ABI=0")
        .opt_level(2)
        .std("c++17")
        .file("csrc/attention/attention_kernels.cu")// csrc/quantized/awq/gemm_kernels.cu csrc/quantized/squeezellm/quant_cuda_kernel.cu csrc/activation_kernels.cu csrc/cache_kernels.cu csrc/cuda_utils_kernels.cu csrc/layernorm_kernels.cu csrc/pos_encoding_kernelsc.cu")
        .compile("libattentionkernels");
}
