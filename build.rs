fn main() {
    println!("cargo:rustc-link-search=native=");
    println!("cargo:rustc-link-lib=python3");
    println!("cargo:rustc-link-search=native=/home/ubuntu/candle-vllm/");
    println!("cargo:rustc-link-lib=rustbind");
}
