use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Tensor};

use crate::openai::{models::ConfigLike, responses::APIError};

use self::ffi::_copy_blocks;

#[cxx::bridge]
mod ffi {

    struct SwapPair {
        k: usize,
        v: usize,
    }
    struct CopyPair {
        k: usize,
        v: Vec<usize>,
    }
    extern "Rust" {}

    unsafe extern "C++" {
        include!("candle-vllm/src/scheduler/cache_engine.h");

        fn _swap_blocks(_src_to_dst: Vec<SwapPair>);
        fn _copy_blocks(_src_to_dst: Vec<CopyPair>);
    }
}

#[derive(Clone)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: Option<usize>, // Set after profiling init
    pub num_cpu_blocks: Option<usize>, // Set after profiling init
    pub fully_init: bool,
}

impl CacheConfig {
    pub fn set_num_gpu_blocks(&mut self, num_gpu_blocks: usize) {
        if self.num_cpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_gpu_blocks = Some(num_gpu_blocks);
    }
    pub fn set_num_cpu_blocks(&mut self, num_gpu_blocks: usize) {
        if self.num_gpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_gpu_blocks = Some(num_gpu_blocks);
    }
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Arc<Vec<KVCache>>,
    cpu_cache: Vec<KVCache>,
    num_layers: usize,
}

impl CacheEngine {
    pub fn new(
        model_config: Box<dyn ConfigLike>,
        cache_config: CacheConfig,
        dtype: DType,
    ) -> Result<Self, APIError> {
        Ok(Self {
            gpu_cache: Arc::new(Self::allocate_gpu_cache(
                &*model_config,
                &cache_config,
                dtype,
            )?),
            cpu_cache: Self::allocate_cpu_cache(&*model_config, &cache_config, dtype)?,
            num_layers: model_config.get_num_hidden_layers(),
        })
    }

    pub fn get_kv_cache(&self) -> Arc<Vec<KVCache>> {
        self.gpu_cache.clone()
    }

    fn allocate_gpu_cache(
        model_config: &dyn ConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
    ) -> Result<Vec<KVCache>, APIError> {
        assert!(cache_config.fully_init);

        let key_block_shape =
            Self::calculate_key_block_shape(model_config, dtype, cache_config.block_size);
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size);
        let mut gpu_cache = Vec::new();
        for _ in 0..model_config.get_num_hidden_layers() {
            let cuda_device = Device::new_cuda(0).map_err(APIError::from)?;
            let key_blocks = Tensor::zeros(
                (
                    cache_config.num_gpu_blocks.unwrap(),
                    key_block_shape.0,
                    key_block_shape.1,
                    key_block_shape.2,
                    key_block_shape.3,
                ),
                dtype,
                &cuda_device,
            )
            .map_err(APIError::from)?;
            let value_blocks = Tensor::zeros(
                (
                    cache_config.num_gpu_blocks.unwrap(),
                    value_block_shape.0,
                    value_block_shape.1,
                    value_block_shape.2,
                ),
                dtype,
                &cuda_device,
            )
            .map_err(APIError::from)?;
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn allocate_cpu_cache(
        model_config: &dyn ConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
    ) -> Result<Vec<KVCache>, APIError> {
        assert!(cache_config.fully_init);

        let key_block_shape =
            Self::calculate_key_block_shape(model_config, dtype, cache_config.block_size);
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size);
        let mut cpu_cache = Vec::new();
        for _ in 0..model_config.get_num_hidden_layers() {
            let cuda_device = Device::new_cuda(0).map_err(APIError::from)?;
            let key_blocks = Tensor::zeros(
                (
                    cache_config.num_cpu_blocks.unwrap(),
                    key_block_shape.0,
                    key_block_shape.1,
                    key_block_shape.2,
                    key_block_shape.3,
                ),
                dtype,
                &cuda_device,
            )
            .map_err(APIError::from)?;
            let value_blocks = Tensor::zeros(
                (
                    cache_config.num_cpu_blocks.unwrap(),
                    value_block_shape.0,
                    value_block_shape.1,
                    value_block_shape.2,
                ),
                dtype,
                &cuda_device,
            )
            .map_err(APIError::from)?;
            cpu_cache.push((key_blocks, value_blocks));
        }
        Ok(cpu_cache)
    }
}

impl CacheEngine {
    fn calculate_key_block_shape(
        model_config: &dyn ConfigLike,
        dtype: DType,
        block_size: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.get_num_kv_heads(),
            model_config.get_head_size() / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ConfigLike,
        block_size: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.get_num_kv_heads(),
            model_config.get_head_size(),
            block_size,
        )
    }
}

impl CacheEngine {
    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) {
        self._swap(&self.cpu_cache, &self.gpu_cache, src_to_dst);
    }
    pub fn swap_out(&self, src_to_dst: HashMap<usize, usize>) {
        self._swap(&self.gpu_cache, &self.cpu_cache, src_to_dst);
    }

    unsafe fn _swap_blocks(
        _src_cache: Tensor,
        _dst_cache: Tensor,
        _src_to_dst: HashMap<usize, usize>,
    ) {
        let mut _src_to_dst_pairs: Vec<ffi::SwapPair> = Vec::new();

        for (key, value) in _src_to_dst.iter() {
            _src_to_dst_pairs.push(ffi::SwapPair {
                k: key.clone(),
                v: value.clone(),
            });
        }
        ffi::_swap_blocks(_src_to_dst_pairs)
    }

    fn _swap(
        &self,
        src: &[(Tensor, Tensor)],
        dst: &[(Tensor, Tensor)],
        src_to_dst: HashMap<usize, usize>,
    ) {
        for i in 0..self.num_layers {
            let (src_key_cache, src_value_cache) = src.get(i).unwrap();
            let (dst_key_cache, dst_value_cache) = dst.get(i).unwrap();
            // Swap (copy) key blocks
            unsafe {
                Self::_swap_blocks(
                    src_key_cache.clone(),
                    dst_key_cache.clone(),
                    src_to_dst.clone(),
                )
            };
            // Swap (copy) key blocks
            unsafe {
                Self::_swap_blocks(
                    src_value_cache.clone(),
                    dst_value_cache.clone(),
                    src_to_dst.clone(),
                )
            };
        }
    }

    unsafe fn _copy_blocks(
        _key_caches: Vec<Tensor>,
        _value_caches: Vec<Tensor>,
        _src_to_dst: HashMap<usize, Vec<usize>>,
    ) {
        let mut _src_to_dst_pairs: Vec<ffi::CopyPair> = Vec::new();

        for (key, value) in _src_to_dst.iter() {
            _src_to_dst_pairs.push(ffi::CopyPair {
                k: key.clone(),
                v: value.clone(),
            });
        }
        ffi::_copy_blocks(_src_to_dst_pairs)
    }

    pub fn copy(&self, src_to_dst: HashMap<usize, Vec<usize>>) {
        let key_caches = self
            .gpu_cache
            .iter()
            .map(|(key_cache, _)| key_cache.clone())
            .collect::<Vec<_>>();
        let value_caches = self
            .gpu_cache
            .iter()
            .map(|(_, value_cache)| value_cache.clone())
            .collect::<Vec<_>>();
        // NOTE(EricLBuehler): from a NOTE(woosuk): This implicitly synchronizes the CPU and GPU
        unsafe { Self::_copy_blocks(key_caches, value_caches, src_to_dst) };
    }
}
