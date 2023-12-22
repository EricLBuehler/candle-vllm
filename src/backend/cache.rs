use std::{collections::HashMap, mem::ManuallyDrop};

use candle_core::{cuda_backend::cudarc::driver::CudaSlice, Device, Storage, Tensor};

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

pub fn swap_blocks(
    src: Tensor,
    dst: &mut Tensor,
    block_mapping: HashMap<usize, usize>,
) -> Result<(), APIError> {
    let block_size_in_bytes = src.dtype().size_in_bytes() * src.dims()[0];
    match (src.device(), dst.device()) {
        (Device::Cuda(src_dev), Device::Cuda(dst_dev)) => {
            if src_dev.ordinal() != dst_dev.ordinal() {
                return Err(APIError::new(format!("Tensors must be on the same device to copy, got ordinals {} (src) and {} (dst).", src_dev.ordinal(), dst_dev.ordinal())))
            }
            let (src_storage, _) = src.storage_and_layout();
            let (dst_storage, _) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Cuda(_)));
            assert!(matches!(&*dst_storage, Storage::Cuda(_)));
            let Storage::Cuda(src_storage) = &*src_storage;
            let Storage::Cuda(dst_storage) = &*dst_storage;
            // u8s because we copy by bytes
            let src_slice: &CudaSlice<u8> = src_storage.as_cuda_slice().map_err(APIError::from)?;
            let dst_slice: &CudaSlice<u8> = dst_storage.as_cuda_slice().map_err(APIError::from)?;
            
            let stream = ManuallyDrop::new(src_dev.cu_stream());
            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset = src_block_number * block_size_in_bytes;
                let dst_offset = dst_block_number * block_size_in_bytes;
                let src_slice = src_slice.slice(src_offset..src_offset+block_size_in_bytes);
                let mut dst_slice = dst_slice.slice_mut(dst_offset..dst_offset+block_size_in_bytes);
                
                src_dev.dtod_copy(&src_slice, &mut dst_slice);
            }
        }
        (Device::Cpu, Device::Cuda(dst_dev)) => {
            
        }
        (Device::Cuda(src_dev), Device::Cpu) => {
            
        }
        (src, dst) => {
            return Err(APIError::new(format!("Tensors must be on either the GPU or CPU to swap,, got {src:?} (src) and {dst:?} (dst).")))
        }
    }

    Ok(())
}
