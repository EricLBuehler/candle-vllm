use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Tensor};

use range_checked::F64Bounded;

use crate::openai::{models::ConfigLike, responses::APIError};

const _GB: usize = 1 << 30;

pub struct CacheConfig {
    pub(crate) block_size: usize,
    pub(crate) gpu_mem_utilization: f64,
    pub(crate) swap_space_bytes: usize,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) num_gpu_blocks: Option<usize>,
    pub(crate) num_cpu_blocks: Option<usize>,
}

impl CacheConfig {
    pub fn new(
        block_size: usize,
        gpu_mem_utilization: F64Bounded<0, 1, false>,
        swap_space_bytes: usize,
        sliding_window: Option<usize>,
    ) -> Self {
        Self {
            block_size,
            gpu_mem_utilization: *gpu_mem_utilization,
            swap_space_bytes: swap_space_bytes * _GB,
            sliding_window,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
        }
    }
}

pub struct ParallelConfig {
    pipeline_parallel_size: usize,
    tensor_parallel_size: usize,
    worker_use_ray: bool,
    max_parallel_loading_workers: Option<usize>,
    world_size: usize,
}

impl ParallelConfig {
    pub fn new(
        pipeline_parallel_size: usize,
        tensor_parallel_size: usize,
        mut worker_use_ray: bool,
        max_parallel_loading_workers: Option<usize>,
    ) -> Result<Self, APIError> {
        if pipeline_parallel_size > 1 {
            return Err(APIError::new_str(
                "Pipeline parallelization not supported yet.",
            ));
        }

        let world_size = pipeline_parallel_size * tensor_parallel_size;
        if world_size > 1 {
            worker_use_ray = true;
        }

        Ok(Self {
            pipeline_parallel_size,
            tensor_parallel_size,
            worker_use_ray,
            max_parallel_loading_workers,
            world_size,
        })
    }
}

pub struct ModelConfig {
    config: Box<dyn ConfigLike>,
    dtype: DType,
}

impl ModelConfig {
    pub fn new(config: Box<dyn ConfigLike>, dtype: DType) -> Self {
        Self { config, dtype }
    }

    fn get_head_size(&self) -> usize {
        self.config.get_hidden_size() / self.config.get_num_attention_heads()
    }

    fn get_num_layers(&self, parallel_config: &ParallelConfig) -> usize {
        self.config.get_num_hidden_layers() / parallel_config.pipeline_parallel_size
    }

    fn get_num_kv_heads(&self, parallel_config: &ParallelConfig) -> usize {
        (self.config.get_num_kv_heads() / parallel_config.tensor_parallel_size).max(1)
    }
}

pub struct CacheEngine {
    block_size: usize,
    num_gpu_blocks: Option<usize>,
    num_cpu_blocks: Option<usize>,
    gpu_cache: Arc<Vec<(Tensor, Tensor)>>,
    cpu_cache: Arc<Vec<(Tensor, Tensor)>>,
    value_block_shape: ValueBlockShape,
    dtype: DType,
    num_layers: usize,
}

#[derive(Clone)]
struct ValueBlockShape {
    num_heads: usize,
    head_size: usize,
    block_size: usize,
}

struct KeyBlockShape {
    num_heads: usize,
    head_size: usize,
    block_size: usize,
    x: usize,
}

impl CacheEngine {
    pub fn new(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> Result<Self, APIError> {
        let value_block_shape = ValueBlockShape {
            num_heads: model_config.get_num_kv_heads(&parallel_config),
            head_size: model_config.get_head_size(),
            block_size: cache_config.block_size,
        };
        Ok(Self {
            block_size: cache_config.block_size,
            num_gpu_blocks: cache_config.num_gpu_blocks,
            num_cpu_blocks: cache_config.num_cpu_blocks,
            gpu_cache: Arc::new(Self::allocate_gpu_cache(
                &value_block_shape,
                cache_config.num_gpu_blocks,
                model_config.dtype,
            )?),
            cpu_cache: Arc::new(Self::allocate_cpu_cache(
                &value_block_shape,
                cache_config.num_cpu_blocks,
                model_config.dtype,
            )?),
            value_block_shape,
            dtype: model_config.dtype,
            num_layers: model_config.get_num_layers(&parallel_config),
        })
    }

    pub fn get_cache_block_size(
        block_size: usize,
        model_config: &ModelConfig,
        parallel_config: &ParallelConfig,
    ) -> usize {
        let head_size = model_config.get_head_size();
        let num_heads = model_config.get_num_kv_heads(parallel_config);
        let num_layers = model_config.get_num_layers(parallel_config);

        let key_cache_block = block_size * num_heads * head_size;
        let value_cache_block = key_cache_block;
        let total = num_layers * (key_cache_block + value_cache_block);
        let dtype_size = model_config.dtype.size_in_bytes();

        dtype_size * total
    }

    fn get_value_block_shape(value_block_shape: &ValueBlockShape) -> ValueBlockShape {
        value_block_shape.clone()
    }

    fn get_key_block_shape(
        value_block_shape: &ValueBlockShape,
        dtype: DType,
    ) -> Result<KeyBlockShape, APIError> {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        Ok(KeyBlockShape {
            num_heads: value_block_shape.num_heads,
            head_size: value_block_shape.head_size / x,
            block_size: value_block_shape.block_size,
            x,
        })
    }

    fn allocate_gpu_cache(
        value_block_shape: &ValueBlockShape,
        num_gpu_blocks: Option<usize>,
        dtype: DType,
    ) -> Result<Vec<(Tensor, Tensor)>, APIError> {
        let key_block_shape = Self::get_key_block_shape(value_block_shape, dtype)?;
        let value_block_shape = Self::get_value_block_shape(value_block_shape);
        let mut cache = Vec::new();
        for _ in 0..value_block_shape.num_heads {
            //use Tensor::empty, huggingface/candle#1374
            let key_blocks = Tensor::zeros(
                (
                    num_gpu_blocks.unwrap(),
                    key_block_shape.num_heads,
                    key_block_shape.head_size,
                    key_block_shape.block_size,
                    key_block_shape.x,
                ),
                dtype,
                &Device::cuda_if_available(0).map_err(APIError::from)?,
            )
            .map_err(APIError::from)?;

            //use Tensor::empty, huggingface/candle#1374
            let value_blocks = Tensor::zeros(
                (
                    num_gpu_blocks.unwrap(),
                    value_block_shape.num_heads,
                    value_block_shape.head_size,
                    value_block_shape.block_size,
                ),
                dtype,
                &Device::cuda_if_available(0).map_err(APIError::from)?,
            )
            .map_err(APIError::from)?;
            cache.push((key_blocks, value_blocks));
        }
        Ok(cache)
    }

    fn allocate_cpu_cache(
        value_block_shape: &ValueBlockShape,
        num_cpu_blocks: Option<usize>,
        dtype: DType,
    ) -> Result<Vec<(Tensor, Tensor)>, APIError> {
        let key_block_shape = Self::get_key_block_shape(value_block_shape, dtype)?;
        let value_block_shape = Self::get_value_block_shape(value_block_shape);
        let mut cache = Vec::new();
        for _ in 0..value_block_shape.num_heads {
            //use Tensor::empty, huggingface/candle#1374
            //TODO(EricLBuehler): should use pin_memory.
            let key_blocks = Tensor::zeros(
                (
                    num_cpu_blocks.unwrap(),
                    key_block_shape.num_heads,
                    key_block_shape.head_size,
                    key_block_shape.block_size,
                    key_block_shape.x,
                ),
                dtype,
                &Device::Cpu,
            )
            .map_err(APIError::from)?;

            //use Tensor::empty, huggingface/candle#1374
            //TODO(EricLBuehler): should use pin_memory.
            let value_blocks = Tensor::zeros(
                (
                    num_cpu_blocks.unwrap(),
                    value_block_shape.num_heads,
                    value_block_shape.head_size,
                    value_block_shape.block_size,
                ),
                dtype,
                &Device::Cpu,
            )
            .map_err(APIError::from)?;
            cache.push((key_blocks, value_blocks));
        }
        Ok(cache)
    }

    //TODO_URGENT(EricLBuehler)
    fn swap_blocks(src: &Tensor, dst: &Tensor, map: HashMap<usize, usize>) {
        todo!()
    }
    fn copy_blocks(
        key_caches: Vec<&Tensor>,
        value_caches: Vec<&Tensor>,
        block_mapping: HashMap<usize, Vec<usize>>,
    ) {
        todo!()
    }

    fn _swap(
        &self,
        src: Arc<Vec<(Tensor, Tensor)>>,
        dst: Arc<Vec<(Tensor, Tensor)>>,
        src_to_dst: HashMap<usize, usize>,
    ) {
        for i in 0..self.num_layers {
            let (src_key_cache, src_value_cache) = src.get(i).unwrap();
            let (dst_key_cache, dst_value_cache) = dst.get(i).unwrap();
            Self::swap_blocks(src_key_cache, dst_key_cache, src_to_dst);
            Self::swap_blocks(src_value_cache, dst_value_cache, src_to_dst);
        }
    }

    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) {
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst);
    }

    pub fn swap_out(&self, src_to_dst: HashMap<usize, usize>) {
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst);
    }

    pub fn copy(&self, src_to_dsts: HashMap<usize, Vec<usize>>) {
        let key_caches = self
            .gpu_cache
            .iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
        let value_caches = self
            .gpu_cache
            .iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        Self::copy_blocks(key_caches, value_caches, src_to_dsts);
    }
}
