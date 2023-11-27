fn main() {
    println!("cargo:rustc-link-lib=dylib=torch_cuda");
    println!("cargo:rustc-link-search=native=/usr/lib/python3/dist-packages/torch/lib");
}