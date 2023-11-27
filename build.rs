extern crate cc;

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

    cc::Build::new().cuda(true)
        .flag("-I /home/ericbuehler/.local/lib/python3.10/site-packages/torch/include")
        .flag("-I /home/ericbuehler/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include")
        .flag("-I /usr/include/python3.10")
        //.flag("-gencode").flag("arch=compute_52,code=sm_52") // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
        //.flag("-gencode").flag("arch=compute_53,code=sm_53") // Generate code for Maxwell (Jetson TX1).
        //.flag("-gencode").flag("arch=compute_61,code=sm_61") // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        //.flag("-gencode").flag("arch=compute_60,code=sm_60") // Generate code for Pascal (Tesla P100).
        //.flag("-gencode").flag("arch=compute_62,code=sm_62") // Generate code for Pascal (Jetson TX2).
        .flag("-gencode").flag("arch=compute_86,code=sm_86")
        .flag("-D_GLIBCXX_USE_CXX11_ABI=1")
        .flag("-O2")
        .flag("-std=c++17")
        .file("csrc/attention/attention_kernels.cu")// csrc/quantized/awq/gemm_kernels.cu csrc/quantized/squeezellm/quant_cuda_kernel.cu csrc/activation_kernels.cu csrc/cache_kernels.cu csrc/cuda_utils_kernels.cu csrc/layernorm_kernels.cu csrc/pos_encoding_kernelsc.cu")
        .compile("libattentionkernels.a");
}