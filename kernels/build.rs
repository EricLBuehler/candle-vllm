use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;
fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/pagedattention.cu");
    println!("cargo:rerun-if-changed=src/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/marlin_cuda_kernel.cu");
    println!("cargo:rerun-if-changed=src/gptq_cuda_kernel.cu");
    println!("cargo:rerun-if-changed=src/nonzero_bitwise.cu");
    println!("cargo:rerun-if-changed=src/sort.cu");
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or("".to_string()));
    let mut builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math");

    let compute_cap = {
        if let Ok(var) = std::env::var("CUDA_COMPUTE_CAP") {
            var.parse::<usize>().unwrap() * 10
        } else {
            let mut cmd = Command::new("nvidia-smi");
            match cmd
                .args(["--query-gpu=compute_cap", "--format=csv"])
                .output()
            {
                Ok(out) => {
                    let output =
                        String::from_utf8(out.stdout).expect("Output of nvidia-smi was not utf8.");
                    (output
                        .split('\n')
                        .nth(1)
                        .unwrap()
                        .trim()
                        .parse::<f32>()
                        .unwrap()
                        * 100.) as usize
                }
                Err(_) => {
                    panic!(
                        "`CUDA_COMPUTE_CAP` env var not specified and `nvidia-smi` was not found."
                    );
                }
            }
        }
    };

    if compute_cap < 800 {
        builder = builder.arg("-DNO_MARLIN_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    println!("cargo:info={builder:?}");
    builder.build_lib(build_dir.join("libpagedattention.a"));

    let kernel_dir = PathBuf::from("../kernels/");
    let absolute_kernel_dir = std::fs::canonicalize(&kernel_dir)?;

    println!(
        "cargo:rustc-link-search=native={}",
        absolute_kernel_dir.display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=pagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}
