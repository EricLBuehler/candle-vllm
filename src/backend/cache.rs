use std::{collections::HashMap, mem::ManuallyDrop};

use candle_core::{CudaDevice, Device, Tensor};

use crate::openai::responses::APIError;

pub fn reshape_and_cache(
    key: Tensor,
    value: Tensor,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: Tensor,
) {
    todo!()
}

pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: HashMap<usize, Vec<usize>>,
) {
    todo!()
}

enum SwapDirection<'a> {
    HtoD(&'a CudaDevice),
    DtoH(&'a CudaDevice),
    DtoD {
        src: &'a CudaDevice,
        dst: &'a CudaDevice,
    },
}

pub fn swap_blocks(
    src: Tensor,
    dst: &mut Tensor,
    block_mapping: HashMap<usize, usize>,
) -> Result<(), APIError> {
    let swap_direction = match (src.device(), dst.device()) {
        (Device::Cuda(src_dev), Device::Cuda(dst_dev)) => {
            if src_dev.ordinal() != dst_dev.ordinal() {
                return Err(APIError::new(format!("Tensors must be on the same device to copy, got ordinals {} (src) and {} (dst).", src_dev.ordinal(), dst_dev.ordinal())))
            }
            SwapDirection::DtoD{src: src_dev, dst: dst_dev}
        }
        (Device::Cpu, Device::Cuda(dst_dev)) => {
            SwapDirection::HtoD(dst_dev)
        }
        (Device::Cuda(src_dev), Device::Cpu) => {
            SwapDirection::HtoD(src_dev)
        }
        (src, dst) => {
            return Err(APIError::new(format!("Tensors must be on either the GPU or CPU to swap,, got {src:?} (src) and {dst:?} (dst).")))
        }
    };
    todo!()
}
