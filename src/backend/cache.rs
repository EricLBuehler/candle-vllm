use std::{collections::HashMap, mem::ManuallyDrop};

use candle_core::{
    cuda_backend::cudarc::driver::{
        result::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async},
        DevicePtr,
    },
    CudaDevice, Device, Storage, Tensor,
};

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
            assert!(matches!(src_storage, Storage::Cuda(_)));
            assert!(matches!(dst_storage, Storage::Cuda(_)));
            let Storage::Cuda(src_storage) = src_storage;
            let Storage::Cuda(dst_storage) = dst_storage;
            let src_ptr = src_storage.as_cuda_slice().map_err(APIError::from)?.device_ptr();
            let dst_ptr = dst_storage.as_cuda_slice().map_err(APIError::from)?.device_ptr();
            
            // Same device, this is OK
            let stream = ManuallyDrop::new(src_dev.cu_stream());
            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset = src_block_number * block_size_in_bytes;
                let dst_offset = dst_block_number * block_size_in_bytes;
                unsafe { memcpy_dtod_async(dst_ptr + dst_offset, src_ptr + src_offset, block_size_in_bytes, stream) }.map_err(APIError::from)?
            }
        }
        (Device::Cpu, Device::Cuda(dst_dev)) => {
            let (src_storage, _) = src.storage_and_layout();
            let (dst_storage, _) = dst.storage_and_layout();
            assert!(matches!(src_storage, Storage::Cpu));
            assert!(matches!(dst_storage, Storage::Cuda(_)));
            let Storage::Cpu(src_storage) = src_storage;
            let Storage::Cuda(dst_storage) = dst_storage;
            let src_slice = src_storage.as_slice().map_err(APIError::from)?;
            let dst_ptr = dst_storage.as_cuda_slice().map_err(APIError::from)?.device_ptr();
            
            let stream = ManuallyDrop::new(src_dev.cu_stream());
            for (src_block_number, dst_block_number) in block_mapping {
                let dst_offset = dst_block_number * block_size_in_bytes;
                unsafe { memcpy_htod_async(dst_ptr + dst_offset, src_slice[src_block_number], stream) }.map_err(APIError::from)?
            }
        }
        (Device::Cuda(src_dev), Device::Cpu) => {
            let (src_storage, _) = src.storage_and_layout();
            let (dst_storage, _) = dst.storage_and_layout();
            assert!(matches!(src_storage, Storage::Cuda(_)));
            assert!(matches!(dst_storage, Storage::Cpu));
            let Storage::Cuda(src_storage) = src_storage;
            let Storage::Cpu(dst_storage) = dst_storage;
            let src_ptr = src_storage.as_cuda_slice().map_err(APIError::from)?.device_ptr();
            let dst_slice = dst_storage.as_slice().map_err(APIError::from)?;
            
            let stream = ManuallyDrop::new(src_dev.cu_stream());
            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset = src_block_number * block_size_in_bytes;
                unsafe { memcpy_dtoh_async(src_ptr + src_offset, dst_slice[dst_block_number], stream) }.map_err(APIError::from)?
            }
        }
        (src, dst) => {
            return Err(APIError::new(format!("Tensors must be on either the GPU or CPU to swap,, got {src:?} (src) and {dst:?} (dst).")))
        }
    }

    Ok(())
}
