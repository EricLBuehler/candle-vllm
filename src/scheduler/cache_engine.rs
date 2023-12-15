use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use crate::openai::{models::ConfigLike, responses::APIError};

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
}

impl CacheEngine {
    pub fn new(
        model_config: Box<dyn ConfigLike>,
        cache_config: CacheConfig,
        dtype: DType,
    ) -> Result<Self, APIError> {
        Ok(Self {
            gpu_cache: Arc::new(Self::allocate_gpu_cache(
                &model_config,
                &cache_config,
                dtype,
            )?),
            cpu_cache: Self::allocate_cpu_cache(&model_config, &cache_config, dtype)?,
        })
    }

    pub fn get_kv_cache(&self) -> Arc<Vec<KVCache>> {
        self.gpu_cache.clone()
    }

    fn allocate_gpu_cache(
        model_config: &Box<dyn ConfigLike>,
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
        model_config: &Box<dyn ConfigLike>,
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
        model_config: &Box<dyn ConfigLike>,
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
        model_config: &Box<dyn ConfigLike>,
        block_size: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.get_num_kv_heads(),
            model_config.get_head_size(),
            block_size,
        )
    }
}
