mod cache;
pub mod gguf;
pub mod gptq;
mod paged_attention;
#[cfg(feature = "cuda")]
pub fn get_or_load_func(
    ptx_file: &'static str,
    kernel_base: &str,
    dtype: DType,
    suffix: Option<&str>,
    device: &CudaDevice,
) -> Result<CudaFunction, APIError> {
    let spec = match dtype {
        DType::U8 => "_u8",
        DType::U32 => "_u32",
        DType::I64 => "_i64",
        DType::BF16 => "_bf16",
        DType::F16 => "_f16",
        DType::F32 => "_f32",
        DType::F64 => "_f64",
    };
    let spec = if let Some(suffix) = suffix {
        spec.to_owned() + suffix
    } else {
        spec.to_owned()
    };
    let kernel = kernel_base.to_owned() + &spec;
    device
        .get_or_load_func(&kernel, ptx_file)
        .map_err(APIError::from)
}

use crate::openai::responses::APIError;
pub use cache::*;
use candle_core::DType;
#[cfg(feature = "cuda")]
use candle_core::{cuda_backend::cudarc::driver::CudaFunction, CudaDevice};
pub use gptq::*;
pub use paged_attention::*;
pub use std::ops::Deref;
pub mod custom_ops;
#[cfg(feature = "nccl")]
pub mod heartbeat;
pub mod progress;
