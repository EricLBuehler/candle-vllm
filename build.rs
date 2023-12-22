use std::{error::Error, fs, thread};

const OPT_LEVEL: usize = 2;
const CPP_STD: usize = 17;
const WORKERS: usize = 4;

fn main() {
    let compute_cap = compute_cap().unwrap();
    if compute_cap < 70 {
        panic!("GPUs with runtime capability below 7.0 (70) are not supported. Got {compute_cap}.");
    }

    let files = fs::read_dir("./kernels")
        .unwrap()
        .map(|file| file.expect("Did not get a valid .cu file"))
        .filter(|file| {
            file.path()
                .extension()
                .unwrap_or_else(|| panic!("No valid extension for {file:?}."))
                == "cu"
        })
        .map(|file| {
            file.file_name()
                .into_string()
                .expect("Could not be converted to a String.")
        })
        .collect::<Vec<_>>();

    if files.len() <= WORKERS {
        compile_files(files, compute_cap);
        return;
    }

    let chunked_files = files.chunks(files.len() / WORKERS);
    let mut handles = Vec::new();
    for files in chunked_files {
        let files = files.to_vec();
        let handle = thread::spawn(move || {
            compile_files(files, compute_cap);
        });
        handles.push(handle);
    }

    // Wait for all to finish
    while !handles.iter().all(|handle| handle.is_finished()) {}
}

fn compile_files(files: Vec<String>, compute_cap: usize) {
    for file in files {
        let mut command = std::process::Command::new("nvcc");
        command
            .arg(format!("--gpu-architecture=sm_{compute_cap}"))
            .arg("--ptx")
            .arg("--use_fast_math")
            .arg(&format!("-std=c++{CPP_STD}"))
            .args(["-O", &OPT_LEVEL.to_string()])
            .args(["--default-stream", "per-thread"])
            .args(["--output-directory", "kernels/"]);
        command.arg(&format!("kernels/{file}"));
        let mut res = command
            .spawn()
            .unwrap_or_else(|_| panic!("nvcc failed for {file}."));
        let res = res.wait().unwrap_or_else(|_| panic!("nvcc failed."));
        if !res.success() {
            panic!("{command:?} failed with exit code {res}");
        }
    }
}

#[allow(unused)]
fn compute_cap() -> Result<usize, Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Try to parse compute caps from env
    let mut compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap_str}");
        compute_cap_str
            .parse::<usize>()
            .expect("Could not parse code")
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=compute_cap")
                .arg("--format=csv")
                .output()
                .expect("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).expect("stdout is not a utf8 string");
        let mut lines = out.lines();
        assert_eq!(lines.next().expect("missing line in stdout"), "compute_cap");
        let cap = lines
            .next()
            .expect("missing line in stdout")
            .replace('.', "");
        let cap = cap
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("cannot parse as int {cap}"));
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    // Grab available GPU codes from nvcc and select the highest one
    let (supported_nvcc_codes, max_nvcc_code) = {
        let out = std::process::Command::new("nvcc")
                .arg("--list-gpu-code")
                .output()
                .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).unwrap();

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        let max_nvcc_code = *codes.last().expect("no gpu codes parsed from nvcc");
        (codes, max_nvcc_code)
    };

    // Check that nvcc supports the asked compute caps
    if !supported_nvcc_codes.contains(&compute_cap) {
        panic!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        panic!(
            "CUDA compute cap {compute_cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
        );
    }

    Ok(compute_cap)
}
