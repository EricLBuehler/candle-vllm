use crate::openai::models::Config;
use candle_core::{DType, Device, Result, Tensor};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use crate::backend::{copy_blocks, swap_blocks};

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: Option<usize>, // Set after profiling init
    pub num_cpu_blocks: Option<usize>, // Set after profiling init
    pub fully_init: bool,
    pub dtype: DType,
    pub kvcache_mem_gpu: usize, // in MB
}

impl CacheConfig {
    pub fn set_num_gpu_blocks(&mut self, num_gpu_blocks: usize) {
        if self.num_cpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_gpu_blocks = Some(num_gpu_blocks);
    }
    pub fn set_num_cpu_blocks(&mut self, num_cpu_blocks: usize) {
        if self.num_gpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_cpu_blocks = Some(num_cpu_blocks);
    }
}

pub type KVCache = (Tensor, Tensor);

#[derive(Debug)]
pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
    cpu_cache: Vec<KVCache>,
    num_layers: usize,
}

impl CacheEngine {
    pub fn new(
        model_config: &Config,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        num_shards: usize,
    ) -> Result<Self> {
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_kv_cache(
                model_config,
                cache_config,
                dtype,
                device,
                num_shards,
            )?)),
            cpu_cache: Self::allocate_kv_cache(
                model_config,
                cache_config,
                dtype,
                &Device::Cpu,
                num_shards,
            )?,
            num_layers: model_config.kv_cache_num_layers(),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        loop {
            if let Ok(v) = self.gpu_cache.try_lock() {
                return v;
            }
        }
    }

    fn allocate_kv_cache(
        model_config: &Config,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        num_shards: usize,
    ) -> Result<Vec<KVCache>> {
        #[cfg(feature = "cuda")]
        let num_blocks = cache_config.num_gpu_blocks.unwrap_or(32);
        // dummy cpu kvcache on Metal
        #[cfg(not(feature = "cuda"))]
        let num_blocks = if device.is_cpu() {
            1
        } else {
            cache_config.num_gpu_blocks.unwrap_or(32)
        };

        if cfg!(feature = "flash-decoding") {
            let kv_shape = Self::calculate_flash_key_value_block_shape(
                model_config,
                cache_config.block_size,
                num_shards,
            );

            let mut cache = Vec::new();
            for _ in 0..model_config.kv_cache_num_layers() {
                let key_blocks = Tensor::zeros(
                    (num_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    device,
                )?;
                let value_blocks = Tensor::zeros(
                    (num_blocks, kv_shape.0, kv_shape.1, kv_shape.2),
                    dtype,
                    device,
                )?;
                cache.push((key_blocks, value_blocks));
            }
            Ok(cache)
        } else {
            let fp8_kvcache = matches!(dtype, DType::U8);
            if !device.is_cpu() {
                println!(
                    "Using FP8 KV Cache? {}, cache dtype {:?}",
                    fp8_kvcache, dtype
                );
            }

            let kshape = Self::calculate_key_block_shape(
                model_config,
                dtype,
                cache_config.block_size,
                num_shards,
            );
            let vshape = Self::calculate_value_block_shape(
                model_config,
                cache_config.block_size,
                num_shards,
            );

            let mut cache = Vec::new();
            for _ in 0..model_config.kv_cache_num_layers() {
                let key_blocks = Tensor::zeros(
                    (num_blocks, kshape.0, kshape.1, kshape.2, kshape.3),
                    dtype,
                    device,
                )?;
                let value_blocks =
                    Tensor::zeros((num_blocks, vshape.0, vshape.1, vshape.2), dtype, device)?;
                cache.push((key_blocks, value_blocks));
            }
            Ok(cache)
        }
    }
}

impl CacheEngine {
    fn calculate_key_block_shape(
        cfg: &Config,
        dtype: DType,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards,
            cfg.k_head_dim() / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        cfg: &Config,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize) {
        (
            cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards,
            cfg.v_head_dim(),
            block_size,
        )
    }

    //[num_blocks, block_size, num_kv_heads, head_size]
    fn calculate_flash_key_value_block_shape(
        cfg: &Config,
        block_size: usize,
        num_shards: usize,
    ) -> (usize, usize, usize) {
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        (
            block_size,
            cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards,
            head_dim,
        )
    }
}

impl CacheEngine {
    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        for i in 0..self.num_layers {
            let (src_key_cache, src_value_cache) = self.cpu_cache.get(i).unwrap();
            let mut gpu_cache = self.get_kv_cache();
            let (dst_key_cache, dst_value_cache) = gpu_cache.get_mut(i).unwrap();
            // Swap (copy) key blocks
            swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone())?;
            // Swap (copy) key blocks
            swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone())?;
        }
        Ok(())
    }

    pub fn swap_out(&mut self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        for i in 0..self.num_layers {
            let gpu_cache = self.get_kv_cache();
            let (src_key_cache, src_value_cache) = gpu_cache.get(i).unwrap().clone();
            drop(gpu_cache);

            let (dst_key_cache, dst_value_cache) = self.cpu_cache.get_mut(i).unwrap();
            // Swap (copy) key blocks
            swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone())?;
            // Swap (copy) key blocks
            swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone())?;
        }
        Ok(())
    }
    #[allow(unused_unsafe)]
    pub fn copy(&mut self, src_to_dst: HashMap<usize, Vec<usize>>) -> Result<()> {
        let mut gpu_cache = self.get_kv_cache();
        #[allow(clippy::map_identity)]
        let caches: (Vec<&mut Tensor>, Vec<&mut Tensor>) =
            gpu_cache.iter_mut().map(|(a, b)| (a, b)).unzip();
        let (key_caches, value_caches) = caches;

        // NOTE(EricLBuehler): This may synchronize the CPU and GPU
        unsafe {
            copy_blocks(key_caches, value_caches, src_to_dst)?;
        }
        Ok(())
    }
}
