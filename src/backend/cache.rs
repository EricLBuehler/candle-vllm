#[cfg(feature = "cuda")]
use crate::{openai::responses::APIError, try_api};
#[cfg(feature = "metal")]
use candle_core::{
    backend::BackendStorage, CpuStorage, Device, IndexOp, Layout, MetalDevice, MetalStorage,
    Result, Storage, Tensor, WithDType,
};
#[cfg(feature = "cuda")]
use candle_core::{
    cuda_backend::cudarc::driver::{CudaSlice, DevicePtr},
    cuda_backend::CudaStorageSlice,
    Device, IndexOp, Storage, Tensor,
};
#[cfg(feature = "cuda")]
use kernels::ffi::{copy_blocks_bf16, copy_blocks_f16, copy_blocks_f32};
use std::{collections::HashMap, iter::zip};

/// # Safety
/// Unsafe due to passing pointers
#[cfg(feature = "cuda")]
pub unsafe fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: HashMap<usize, Vec<usize>>,
) -> Result<(), APIError> {
    use candle_core::DType;

    let cache_dev = key_caches.first().unwrap().device();
    let Device::Cuda(dev) = cache_dev else {
        panic!("Expected the key caches to be on a CUDA device.")
    };
    if !cache_dev.same_device(value_caches.first().unwrap().device()) {
        return Err(APIError::new(format!(
            "`key` and `value` caches have different devices, got {:?} and {:?} respectively.",
            cache_dev,
            value_caches.first().unwrap().device()
        )));
    }
    if key_caches.first().unwrap().dtype() != value_caches.first().unwrap().dtype() {
        return Err(APIError::new(format!(
            "Key and value caches have different types, got {:?} and {:?}.",
            key_caches.first().unwrap().dtype(),
            value_caches.first().unwrap().dtype()
        )));
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
        try_api!(key_cache.to_device(cache_dev));
        try_api!(value_cache.to_device(cache_dev));

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
                return Err(APIError::from(
                    "only f32, f16 and bf16 input data type supported!",
                ));
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

    let numel_per_block: u32 = try_api!(key_caches.first().unwrap().i(0))
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

#[cfg(feature = "cuda")]
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
            let (src_storage, src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Cuda(_)));
            assert!(matches!(&*dst_storage, Storage::Cuda(_)));
            let Storage::Cuda(src_storage) = &*src_storage else { unreachable!() };
            let Storage::Cuda(dst_storage) = &*dst_storage else { unreachable!() };
            let (src_ptr, dst_ptr) = match (&src_storage.slice, &dst_storage.slice) {
                (CudaStorageSlice::BF16(slice_src), CudaStorageSlice::BF16(slice_dst)) => {
                    let ptr_src = *slice_src.slice(src_layout.start_offset()..).device_ptr();
                    let ptr_dst = *slice_dst.slice(dst_layout.start_offset()..).device_ptr();
                    (ptr_src, ptr_dst)
                }
                (CudaStorageSlice::F16(slice_src), CudaStorageSlice::F16(slice_dst)) => {
                    let ptr_src = *slice_src.slice(src_layout.start_offset()..).device_ptr();
                    let ptr_dst = *slice_dst.slice(dst_layout.start_offset()..).device_ptr();
                    (ptr_src, ptr_dst)
                }
                (CudaStorageSlice::F32(slice_src), CudaStorageSlice::F32(slice_dst)) => {
                    let ptr_src = *slice_src.slice(src_layout.start_offset()..).device_ptr();
                    let ptr_dst = *slice_dst.slice(dst_layout.start_offset()..).device_ptr();
                    (ptr_src, ptr_dst)
                }
                _ => {
                    return Err(APIError::from("only f32, f16 and bf16 input data type supported!"));
                }
            };
            // let src_ptr = src_storage.as_cuda_slice::<u8>().map_err(APIError::from)?.device_ptr() + TryInto::<u64>::try_into(src_layout.start_offset()).unwrap();
            // let dst_ptr = dst_storage.as_cuda_slice::<u8>().map_err(APIError::from)?.device_ptr() + TryInto::<u64>::try_into(dst_layout.start_offset()).unwrap();

            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset: u64 = (src_block_number * block_size_in_bytes).try_into().unwrap();
                let dst_offset: u64 = (dst_block_number * block_size_in_bytes).try_into().unwrap();
                // u8s because we copy by bytes
                let src_slice: CudaSlice<u8> = unsafe { src_dev.upgrade_device_ptr(src_ptr+src_offset, block_size_in_bytes) };
                let mut dst_slice = unsafe { dst_dev.upgrade_device_ptr(dst_ptr+dst_offset, block_size_in_bytes) };

                try_api!(src_dev.dtod_copy(&src_slice, &mut dst_slice));
            }
        }
        (Device::Cpu, Device::Cuda(dst_dev)) => {
            let (src_storage, _src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Cpu(_)));
            assert!(matches!(&*dst_storage, Storage::Cuda(_)));
            let Storage::Cpu(src_storage) = &*src_storage else { unreachable!() };
            let Storage::Cuda(dst_storage) = &*dst_storage else { unreachable!() };
            let dst_ptr = dst_storage.as_cuda_slice::<u8>().map_err(APIError::from)?.device_ptr() + TryInto::<u64>::try_into(dst_layout.start_offset()).unwrap();
            let src_slice = try_api!(src_storage.as_slice());

            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset = src_block_number * block_size_in_bytes;
                let dst_offset: u64 = (dst_block_number * block_size_in_bytes).try_into().unwrap();
                // u8s because we copy by bytes
                let mut dst_slice: CudaSlice<u8> = unsafe { dst_dev.upgrade_device_ptr(dst_ptr+dst_offset, block_size_in_bytes) };

                try_api!(dst_dev.htod_sync_copy_into(&src_slice[src_offset..src_offset+block_size_in_bytes], &mut dst_slice));
            }
        }
        (src, dst) => {
            return Err(APIError::new(format!("Tensors must be on either the GPU or CPU to swap,, got {src:?} (src) and {dst:?} (dst).")))
        }
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

        metal_kernels::call_copy_blocks(
            dev.device(),
            &command_buffer,
            metal_kernels::Kernels::default(),
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

// `dst` REALLY should be &mut. That's the only reason this is unsafe.
/// # Safety
/// `dst` is the only shared reference and upholds the `&mut` aliasing guarantee.
#[cfg(feature = "metal")]
pub fn swap_blocks(src: Tensor, dst: &Tensor, block_mapping: HashMap<usize, usize>) -> Result<()> {
    let block_size_in_bytes = src.dtype().size_in_bytes() * src.dims()[0];
    if src.device().location() != dst.device().location() {
        candle_core::bail!(
            "Tensors must be on the same device to copy, got locations {:?} (src) and {:?} (dst).",
            src.device().location(),
            dst.device().location()
        );
    }
    match (src.device(), dst.device()) {
        (Device::Metal(src_dev), Device::Metal(_)) => {
            let (src_storage, src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Metal(_)));
            assert!(matches!(&*dst_storage, Storage::Metal(_)));
            let Storage::Metal(src_storage) = &*src_storage else {
                unreachable!()
            };
            let Storage::Metal(dst_storage) = &*dst_storage else {
                unreachable!()
            };

            for (src_block_number, dst_block_number) in block_mapping {
                // We copy by bytes
                let src_offset = src_block_number * block_size_in_bytes
                    + src_layout.start_offset() * src_storage.dtype().size_in_bytes();
                let dst_offset = dst_block_number * block_size_in_bytes
                    + dst_layout.start_offset() * dst_storage.dtype().size_in_bytes();

                let command_buffer = src_dev.command_buffer()?;
                command_buffer.set_label("swap-blocks-gpu-gpu");
                let blit = command_buffer.new_blit_command_encoder();
                blit.set_label("swap-blocks-gpu-gpu");
                let length = (src_layout.shape().elem_count() * src_storage.dtype().size_in_bytes())
                    as metal::NSUInteger;
                blit.copy_from_buffer(
                    src_storage.buffer(),
                    src_offset as u64,
                    dst_storage.buffer(),
                    dst_offset as u64,
                    length,
                );
                blit.end_encoding();
            }
        }
        (Device::Cpu, Device::Metal(dev)) => {
            let (src_storage, src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Cpu(_)));
            assert!(matches!(&*dst_storage, Storage::Metal(_)));
            let Storage::Cpu(src_storage) = &*src_storage else {
                unreachable!()
            };
            let Storage::Metal(dst_storage) = &*dst_storage else {
                unreachable!()
            };

            fn swap_thunk<SRCT: WithDType>(
                src_slice: &[SRCT],
                src_layout: &Layout,
                dst_storage: &MetalStorage,
                dst_layout: &Layout,
                dev: &MetalDevice,
                block_size_in_bytes: usize,
                block_mapping: HashMap<usize, usize>,
            ) -> Result<()> {
                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset = src_block_number * block_size_in_bytes
                        + src_layout.start_offset() * SRCT::DTYPE.size_in_bytes();
                    let dst_offset = dst_block_number * block_size_in_bytes
                        + dst_layout.start_offset() * dst_storage.dtype().size_in_bytes();
                    // We copy by bytes
                    let src_buffer = dev.new_buffer_with_data(
                        &src_slice[src_offset..src_offset + block_size_in_bytes],
                    )?;

                    let command_buffer = dev.command_buffer()?;
                    command_buffer.set_label("swap-blocks-cpu-gpu");
                    let blit = command_buffer.new_blit_command_encoder();
                    blit.set_label("swap-blocks-cpu-gpu");
                    let length = (src_layout.shape().elem_count() * SRCT::DTYPE.size_in_bytes())
                        as metal::NSUInteger;
                    blit.copy_from_buffer(
                        &src_buffer,
                        src_offset as u64,
                        dst_storage.buffer(),
                        dst_offset as u64,
                        length,
                    );
                    blit.end_encoding();
                }
                Ok(())
            }

            match src_storage {
                CpuStorage::BF16(s) => swap_thunk(
                    s,
                    src_layout,
                    dst_storage,
                    dst_layout,
                    dev,
                    block_size_in_bytes,
                    block_mapping,
                )?,
                CpuStorage::F16(s) => swap_thunk(
                    s,
                    src_layout,
                    dst_storage,
                    dst_layout,
                    dev,
                    block_size_in_bytes,
                    block_mapping,
                )?,
                CpuStorage::F32(s) => swap_thunk(
                    s,
                    src_layout,
                    dst_storage,
                    dst_layout,
                    dev,
                    block_size_in_bytes,
                    block_mapping,
                )?,
                _ => candle_core::bail!("expected bf16, f16, or f32 for cpu<>gpu swap-blocks"),
            }
        }
        (src, dst) => {
            candle_core::bail!("Tensors must be on either the GPU or CPU to swap, got {src:?} (src) and {dst:?} (dst).");
        }
    }

    Ok(())
}
