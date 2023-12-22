mod cache;
mod layers;
mod paged_attention;

const COPY_BLOCKS_PTX: &'static str = "kernels/copy_blocks_kernel.ptx";

const COPY_BLOCKS_KERNEL_U8: &'static str = "copy_blocks_kernel_u8";
const COPY_BLOCKS_KERNEL_U32: &'static str = "copy_blocks_kernel_u32";
const COPY_BLOCKS_KERNEL_I64: &'static str = "copy_blocks_kernel_i64";
const COPY_BLOCKS_KERNEL_BF16: &'static str = "copy_blocks_kernel_bf16";
const COPY_BLOCKS_KERNEL_F16: &'static str = "copy_blocks_kernel_f16";
const COPY_BLOCKS_KERNEL_F32: &'static str = "copy_blocks_kernel_f32";
const COPY_BLOCKS_KERNEL_F64: &'static str = "copy_blocks_kernel_f64";

pub use cache::*;
pub use layers::*;
pub use paged_attention::*;
