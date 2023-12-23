mod cache;
mod layers;
mod paged_attention;

const COPY_BLOCKS_PTX: &str = "kernels/copy_blocks_kernel.ptx";

const COPY_BLOCKS_KERNEL: &str = "copy_blocks_kernel";

pub fn get_or_load_func(
    ptx_file: &'static str,
    kernel_base: &str,
    dtype: DType,
    device: &CudaDevice,
) -> Result<CudaFunction, APIError> {
    let suffix = match dtype {
        DType::U8 => "_u8",
        DType::U32 => "_u32",
        DType::I64 => "_i64",
        DType::BF16 => "_bf16",
        DType::F16 => "_f16",
        DType::F32 => "_f32",
        DType::F64 => "_f64",
    };
    let kernel = kernel_base.to_owned() + suffix;
    device
        .get_or_load_func(&kernel, ptx_file)
        .map_err(APIError::from)
}

struct Conjoined<'a, T, R> {
    raw: NonNull<T>,
    _ref: &'a mut R,
}

impl<'a, T, R> Deref for Conjoined<'a, T, R> {
    type Target = NonNull<T>;
    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

pub use cache::*;
use candle_core::{cuda_backend::cudarc::driver::CudaFunction, CudaDevice, DType};
pub use layers::*;
pub use paged_attention::*;
pub use std::ops::Deref;
use std::ptr::NonNull;

use crate::openai::responses::APIError;
