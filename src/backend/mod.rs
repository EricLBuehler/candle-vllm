mod cache;
mod layers;
mod paged_attention;

struct KernelInformation {
    ptx: &'static str,
    function: &'static str,
}

const COPY_BLOCKS_KERNEL: KernelInformation = KernelInformation {
    ptx: "kernels/copy_blocks_kernel.ptx",
    function: "copy_blocks_kernel_i32",
};

pub use cache::*;
pub use layers::*;
pub use paged_attention::*;
