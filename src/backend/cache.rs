#[cfg(feature = "cuda")]
use attention_rs::kernels::ffi::{copy_blocks_bf16, copy_blocks_f16, copy_blocks_f32};
#[cfg(feature = "metal")]
use candle_core::{backend::BackendStorage, Device, IndexOp, Result, Storage, Tensor};
#[cfg(feature = "cuda")]
use candle_core::{
    cuda_backend::cudarc::driver::DevicePtr, cuda_backend::CudaStorageSlice, Device, IndexOp,
    Result, Storage, Tensor,
};
use std::{collections::HashMap, iter::zip};

/// # Safety
/// Unsafe due to passing pointers
#[cfg(feature = "cuda")]
pub unsafe fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: HashMap<usize, Vec<usize>>,
) -> Result<()> {
    use candle_core::DType;

    let cache_dev = key_caches.first().unwrap().device();
    let Device::Cuda(dev) = cache_dev else {
        panic!("Expected the key caches to be on a CUDA device.")
    };
    if !cache_dev.same_device(value_caches.first().unwrap().device()) {
        candle_core::bail!(
            "`key` and `value` caches have different devices, got {:?} and {:?} respectively.",
            cache_dev,
            value_caches.first().unwrap().device()
        )
    }
    if key_caches.first().unwrap().dtype() != value_caches.first().unwrap().dtype() {
        candle_core::bail!(
            "Key and value caches have different types, got {:?} and {:?}.",
            key_caches.first().unwrap().dtype(),
            value_caches.first().unwrap().dtype()
        )
    }
    let num_layers: u32 = key_caches.len().try_into().unwrap();
    if num_layers == 0 {
        return Ok(());
    }

    let mut key_cache_ptrs = Vec::new();
    key_cache_ptrs.reserve_exact(num_layers as usize);
    let mut value_cache_ptrs = Vec::new();
    value_cache_ptrs.reserve_exact(num_layers as usize);
    let mut dtype = DType::F32;

    for (key_cache, value_cache) in zip(&key_caches, &value_caches) {
        key_cache.to_device(cache_dev)?;
        value_cache.to_device(cache_dev)?;

        let key_offset: u64 = key_cache
            .storage_and_layout()
            .1
            .start_offset()
            .try_into()
            .unwrap();
        let Storage::Cuda(key_storage) = &*key_cache.storage_and_layout().0 else {
            unreachable!()
        };

        // let key_ptr = *try_api!(key_storage.as_cuda_slice::<u8>()).device_ptr();

        let value_offset: u64 = value_cache
            .storage_and_layout()
            .1
            .start_offset()
            .try_into()
            .unwrap();
        let Storage::Cuda(value_storage) = &*value_cache.storage_and_layout().0 else {
            unreachable!()
        };
        // let value_ptr = *try_api!(value_storage.as_cuda_slice::<u8>()).device_ptr();
        let (key_ptr, value_ptr) = match (&key_storage.slice, &value_storage.slice) {
            (CudaStorageSlice::BF16(slice_key), CudaStorageSlice::BF16(slice_value)) => {
                let ptr_key = *slice_key.slice(0..).device_ptr();
                let ptr_value = *slice_value.slice(0..).device_ptr();
                dtype = DType::BF16;
                (ptr_key, ptr_value)
            }
            (CudaStorageSlice::F16(slice_key), CudaStorageSlice::F16(slice_value)) => {
                let ptr_key = *slice_key.slice(0..).device_ptr();
                let ptr_value = *slice_value.slice(0..).device_ptr();
                dtype = DType::F16;
                (ptr_key, ptr_value)
            }
            (CudaStorageSlice::F32(slice_key), CudaStorageSlice::F32(slice_value)) => {
                let ptr_key = *slice_key.slice(0..).device_ptr();
                let ptr_value = *slice_value.slice(0..).device_ptr();
                (ptr_key, ptr_value)
            }
            _ => {
                candle_core::bail!("only f32, f16 and bf16 input data type supported!")
            }
        };
        key_cache_ptrs.push(key_ptr + key_offset);
        value_cache_ptrs.push(value_ptr + value_offset);
    }

    let mut block_mapping_vec: Vec<i64> = Vec::new();
    for (src_block_number, dst_blocks) in block_mapping {
        for dst_block_number in dst_blocks {
            block_mapping_vec.push(src_block_number.try_into().unwrap());
            block_mapping_vec.push(dst_block_number.try_into().unwrap());
        }
    }
    let num_pairs: u32 = (block_mapping_vec.len() / 2).try_into().unwrap();

    let key_cache_ptr = key_cache_ptrs.as_mut_ptr() as *mut core::ffi::c_void;
    let value_cache_ptr = value_cache_ptrs.as_mut_ptr() as *mut core::ffi::c_void;
    let block_mapping_ptr = block_mapping_vec.as_mut_ptr() as *const core::ffi::c_void;

    let numel_per_block: u32 = key_caches
        .first()
        .unwrap()
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();

    match dtype {
        DType::BF16 => unsafe {
            copy_blocks_bf16(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers as i32,
                num_pairs as i32,
                numel_per_block as i32,
                *dev.cu_stream() as i64,
            );
        },
        DType::F16 => unsafe {
            copy_blocks_f16(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers as i32,
                num_pairs as i32,
                numel_per_block as i32,
                *dev.cu_stream() as i64,
            );
        },
        DType::F32 => unsafe {
            copy_blocks_f32(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers as i32,
                num_pairs as i32,
                numel_per_block as i32,
                *dev.cu_stream() as i64,
            );
        },
        _ => {}
    }

    Ok(())
}

#[cfg(feature = "metal")]
pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: HashMap<usize, Vec<usize>>,
) -> Result<()> {
    let cache_dev = key_caches.first().unwrap().device();
    let Device::Metal(dev) = cache_dev else {
        panic!("Expected the key caches to be on a Metal device.")
    };
    if !cache_dev.same_device(value_caches.first().unwrap().device()) {
        candle_core::bail!(
            "`key` and `value` caches have different devices, got {:?} and {:?} respectively.",
            cache_dev,
            value_caches.first().unwrap().device()
        );
    }
    if key_caches.first().unwrap().dtype() != value_caches.first().unwrap().dtype() {
        candle_core::bail!(
            "Key and value caches have different types, got {:?} and {:?}.",
            key_caches.first().unwrap().dtype(),
            value_caches.first().unwrap().dtype()
        );
    }

    let mut block_mapping_vec: Vec<i64> = Vec::new();
    for (src_block_number, dst_blocks) in block_mapping {
        for dst_block_number in dst_blocks {
            block_mapping_vec.push(src_block_number.try_into().unwrap());
            block_mapping_vec.push(dst_block_number.try_into().unwrap());
        }
    }
    let block_mapping = dev.new_buffer_with_data(&block_mapping_vec)?;

    let num_pairs: u64 = (block_mapping_vec.len() / 2).try_into().unwrap();

    let numel_per_block: u64 = key_caches
        .first()
        .unwrap()
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();

    for (key_cache, value_cache) in zip(&key_caches, &value_caches) {
        key_cache.to_device(cache_dev)?;
        value_cache.to_device(cache_dev)?;

        let key_offset = key_cache.storage_and_layout().1.start_offset();
        let Storage::Metal(key_storage) = &*key_cache.storage_and_layout().0 else {
            unreachable!()
        };

        let value_offset = value_cache.storage_and_layout().1.start_offset();
        let Storage::Metal(value_storage) = &*value_cache.storage_and_layout().0 else {
            unreachable!()
        };

        let command_buffer = dev.command_buffer()?;
        command_buffer.set_label("copy-blocks");

        attention_rs::metal_kernels::call_copy_blocks(
            dev.device(),
            &command_buffer,
            attention_rs::metal_kernels::Kernels::default(),
            key_cache.dtype(),
            key_storage.buffer(),
            key_offset * key_storage.dtype().size_in_bytes(),
            value_storage.buffer(),
            value_offset * value_storage.dtype().size_in_bytes(),
            &block_mapping,
            0,
            num_pairs,
            numel_per_block,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    Ok(())
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub unsafe fn copy_blocks(
    _: Vec<&mut candle_core::Tensor>,
    _: Vec<&mut candle_core::Tensor>,
    _: HashMap<usize, Vec<usize>>,
) -> candle_core::Result<()> {
    candle_core::bail!("copy_blocks not implemented for CPU")
}
