use std::{collections::HashMap, iter::zip, ptr::NonNull};

use candle_core::{
    cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, LaunchAsync},
    Device, IndexOp, Storage, Tensor,
};

use crate::{backend::{Conjoined, get_or_load_func, COPY_BLOCKS_PTX, COPY_BLOCKS_KERNEL}, openai::responses::APIError, try_api};

pub fn reshape_and_cache(
    _key: Tensor,
    _value: Tensor,
    _key_cache: &mut Tensor,
    _value_cache: &mut Tensor,
    _slot_mapping: Tensor,
) {
    todo!()
}

pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: HashMap<usize, Vec<usize>>,
) -> Result<(), APIError> {
    let candle_dev = key_caches.first().unwrap().device();
    let Device::Cuda(dev) = dev else {
        panic!("Expected the key caches to be on a CUDA device.")
    };
    if key_caches.first().unwrap().dtype() != value_caches.first().unwrap().dtype() {
        return Err(APIError::new(format!("Key and value caches have different types, got {} and {}.", key_caches.first().unwrap().dtype(), value_caches.first().unwrap().dtype())));
    }
    let num_layers: u32 = key_caches.len().try_into().unwrap();

    let mut key_cache_ptrs = Vec::new();
    key_cache_ptrs.reserve_exact(num_layers as usize);
    let mut value_cache_ptrs = Vec::new();
    value_cache_ptrs.reserve_exact(num_layers as usize);
    for (key_cache, value_cache) in zip(&key_caches, &value_caches) {
        let key_offset: u64 = key_cache
            .storage_and_layout()
            .1
            .start_offset()
            .try_into()
            .unwrap();
        let Storage::Cuda(key_storage) = &*key_cache.storage_and_layout().0 else {
            unreachable!()
        };
        let key_ptr = *try_api!(key_storage.as_cuda_slice::<u8>()).device_ptr();
        key_cache_ptrs.push(key_ptr + key_offset);

        let value_offset: u64 = value_cache
            .storage_and_layout()
            .1
            .start_offset()
            .try_into()
            .unwrap();
        let Storage::Cuda(value_storage) = &*value_cache.storage_and_layout().0 else {
            unreachable!()
        };
        let value_ptr = *try_api!(value_storage.as_cuda_slice::<u8>()).device_ptr();
        value_cache_ptrs.push(value_ptr + value_offset);
    }

    let mut block_mapping_vec: Vec<i64> = Vec::new();
    for (src_block_number, dst_blocks) in block_mapping {
        for dst_block_number in dst_blocks {
            block_mapping_vec.push(src_block_number.try_into().unwrap());
            block_mapping_vec.push(dst_block_number.try_into().unwrap());
        }
    }
    // Safety: Drop block_mapping_array before block_mapping_vec
    let block_mapping_ptr = Conjoined::new(
        NonNull::new(block_mapping_vec.as_mut_ptr()).unwrap(),
        &mut block_mapping_vec,
    );
    let num_pairs: u32 = (block_mapping_vec.len() / 2).try_into().unwrap();

    let key_cache_ptr = Conjoined::new(
        NonNull::new(key_cache_ptrs.as_mut_ptr()).unwrap(),
        &mut key_cache_ptrs,
    );
    let value_cache_ptr = Conjoined::new(
        NonNull::new(value_cache_ptrs.as_mut_ptr()).unwrap(),
        &mut value_cache_ptrs,
    );

    let numel_per_block: u32 = try_api!(key_caches.first().unwrap().i(0))
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();
    let launch_conf = LaunchConfig {
        grid_dim: (num_layers, num_pairs, 1u32),
        block_dim: (numel_per_block.min(1024), 1u32, 1u32),
        shared_mem_bytes: 0,
    };
    let stream = try_api!(dev.fork_default_stream());

    let kernel = try_api!(get_or_load_func(COPY_BLOCKS_PTX, COPY_BLOCKS_KERNEL, key_caches.first().unwrap().dtype(), candle_dev));

    unsafe { kernel.launch_on_stream(&stream, launch_conf, (key_cache_ptr, value_cache_ptr, block_mapping_ptr)) };

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
            let (src_storage, src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Cuda(_)));
            assert!(matches!(&*dst_storage, Storage::Cuda(_)));
            let Storage::Cuda(src_storage) = &*src_storage else { unreachable!() };
            let Storage::Cuda(dst_storage) = &*dst_storage else { unreachable!() };
            let src_ptr = src_storage.as_cuda_slice::<u8>().map_err(APIError::from)?.device_ptr() + TryInto::<u64>::try_into(src_layout.start_offset()).unwrap();
            let dst_ptr = dst_storage.as_cuda_slice::<u8>().map_err(APIError::from)?.device_ptr() + TryInto::<u64>::try_into(dst_layout.start_offset()).unwrap();

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
        (Device::Cuda(src_dev), Device::Cpu) => {
            todo!();
            // Pending on huggingface/candle#1467
            /*let (src_storage, src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_mut_and_layout();
            assert!(matches!(&*src_storage, Storage::Cuda(_)));
            assert!(matches!(&*dst_storage, Storage::Cpu(_)));
            let Storage::Cuda(src_storage) = &*src_storage else { unreachable!() };
            let Storage::Cpu(dst_storage) = &*dst_storage else { unreachable!() };
            let src_ptr = src_storage.as_cuda_slice::<u8>().map_err(APIError::from)?.device_ptr() + TryInto::<u64>::try_into(src_layout.start_offset()).unwrap();
            let dst_slice: &[u8] = try_api!(dst_storage.as_slice());
            let ptr = dst_slice.as_ptr() as *mut u8;
            // Safety:
            let dst_slice = unsafe { slice::from_raw_parts_mut(ptr, dst_slice.len()) };

            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset: u64 = (src_block_number * block_size_in_bytes).try_into().unwrap();
                let dst_offset: u64 = (dst_block_number * block_size_in_bytes).try_into().unwrap();
                // u8s because we copy by bytes
                let src_slice: CudaSlice<u8> = unsafe { src_dev.upgrade_device_ptr(src_ptr+src_offset, block_size_in_bytes) };
                
                try_api!(src_dev.dtoh_sync_copy_into(&src_slice, dst_slice));
            }*/
        }
        (src, dst) => {
            return Err(APIError::new(format!("Tensors must be on either the GPU or CPU to swap,, got {src:?} (src) and {dst:?} (dst).")))
        }
    }

    Ok(())
}
