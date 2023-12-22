fn main() {
    cxx_build::bridge("src/scheduler/cache_engine.rs")
        .file("src/scheduler/cache_engine.cc")
        .flag_if_supported("-std=c++14")
        .compile("candle-vllm");

    println!("cargo:rerun-if-changed=src/scheduler/cache_engine.rs");
    println!("cargo:rerun-if-changed=src/scheduler/cache_engine.cc");
    println!("cargo:rerun-if-changed=src/scheduler/cache_engine.h");

    //println!("cargo:rustc-link-search=native=");
    println!("cargo:rustc-link-lib=dylib=python3");
    println!("cargo:rustc-link-search=native=/home/ubuntu/candle-vllm/");
    println!("cargo:rustc-link-lib=rustbind");
}
