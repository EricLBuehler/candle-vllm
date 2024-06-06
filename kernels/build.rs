fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/attention_kernel.cu");
    println!("cargo:rerun-if-changed=src/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/rotary_embedding_kernel.cu");
    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();
}
