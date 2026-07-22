use crate::openai::models::{Config, KvCacheDtype};
use candle_core::{DType, Device, Result, Tensor};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

use crate::backend::copy_blocks;

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: Option<usize>, // Set after profiling init
    pub num_cpu_blocks: Option<usize>, // Set after profiling init
    pub fully_init: bool,
    pub dtype: DType,
    pub kvcache_dtype: crate::openai::models::KvCacheDtype,
    pub kvcache_mem_gpu: usize, // in MB
    pub mamba_cache_budget_bytes: usize,
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

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
    cpu_cache: Vec<KVCache>,
    cpu_turboquant_cache: Option<Vec<attention_rs::TurboquantLayerCache>>,
    cpu_swap_enabled: bool,
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
        // CPU KV offload is a CUDA-only path. Metal uses unified memory and
        // must not allocate or schedule a separate CPU swap tier.
        let cpu_swap_enabled = cfg!(feature = "cuda") && !device.is_cpu();
        if !cpu_swap_enabled {
            tracing::info!("CPU KV cache swapping disabled for non-CUDA device");
        }

        let cpu_turboquant_cache = if cpu_swap_enabled && cache_config.kvcache_dtype.is_turboquant()
        {
            Some(Self::allocate_turboquant_layers(
                model_config,
                cache_config,
                cache_config.num_cpu_blocks.unwrap_or(0),
                &Device::Cpu,
                num_shards,
            )?)
        } else {
            None
        };

        let engine = Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_kv_cache(
                model_config,
                cache_config,
                dtype,
                device,
                num_shards,
            )?)),
            cpu_cache: if cpu_swap_enabled {
                Self::allocate_kv_cache(
                    model_config,
                    cache_config,
                    dtype,
                    &Device::Cpu,
                    num_shards,
                )?
            } else {
                Vec::new()
            },
            cpu_turboquant_cache,
            cpu_swap_enabled,
            num_layers: model_config.kv_cache_num_layers(),
        };

        if cache_config.kvcache_dtype.is_turboquant() && !device.is_cpu() {
            let num_gpu_blocks = cache_config.num_gpu_blocks.unwrap_or(32);
            Self::init_turboquant_cache(
                model_config,
                cache_config,
                num_gpu_blocks,
                device,
                num_shards,
            )?;
        }

        Ok(engine)
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
        let tq_full = matches!(
            cache_config.kvcache_dtype,
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3
        );

        #[cfg(feature = "cuda")]
        let num_blocks = if tq_full {
            1
        } else if device.is_cpu() {
            cache_config.num_cpu_blocks.unwrap_or(1)
        } else {
            cache_config.num_gpu_blocks.unwrap_or(32)
        };
        #[cfg(not(feature = "cuda"))]
        let num_blocks = if device.is_cpu() {
            if tq_full {
                1
            } else {
                cache_config.num_cpu_blocks.unwrap_or(1)
            }
        } else {
            if tq_full {
                1
            } else {
                cache_config.num_gpu_blocks.unwrap_or(32)
            }
        };

        #[cfg(all(feature = "flashattn", not(feature = "flashinfer"), feature = "cuda"))]
        if matches!(dtype, DType::U8) && !device.is_cpu() {
            let sm = device
                .as_cuda_device()
                .ok()
                .and_then(|d| attention_rs::cuda_utils::sm_version(d))
                .unwrap_or(0);
            if sm != 90 {
                candle_core::bail!(
                    "FP8 KV cache with FlashAttention requires SM90 (Hopper), \
                     but detected SM{sm}. Use FlashInfer backend for FP8 KV cache on current GPU."
                );
            }
        }

        if model_config.is_mla() {
            let block_size = cache_config.block_size;
            let kv_lora_rank = model_config.mla_kv_lora_rank();
            let qk_rope_head_dim = model_config.mla_qk_rope_head_dim();
            let mut cache = Vec::new();
            for _ in 0..model_config.kv_cache_num_layers() {
                let ckv_blocks =
                    Tensor::zeros((num_blocks, block_size, 1, kv_lora_rank), dtype, device)?;
                let kpe_blocks =
                    Tensor::zeros((num_blocks, block_size, 1, qk_rope_head_dim), dtype, device)?;
                cache.push((ckv_blocks, kpe_blocks));
            }
            return Ok(cache);
        }

        let per_layer_config = model_config.gemma4_per_layer_cache_config();
        let use_flash_layout = cfg!(any(
            feature = "flash",
            feature = "flashattn",
            feature = "flashinfer",
            feature = "metal"
        ));
        let block_size = cache_config.block_size;
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;

        if let Some(ref configs) = per_layer_config {
            if !device.is_cpu() {
                println!(
                    "Using per-layer KV cache config for Gemma4: {} layers, max head_dim={}",
                    configs.len(),
                    configs.iter().map(|(_, hd)| *hd).max().unwrap_or(0)
                );
            }
            let mut cache = Vec::new();
            for (layer_kv_heads, layer_head_dim) in configs.iter().copied() {
                let kv_heads = (layer_kv_heads / num_shards.max(1)).max(1);
                if use_flash_layout {
                    let key_blocks = Tensor::zeros(
                        (num_blocks, block_size, kv_heads, layer_head_dim),
                        dtype,
                        device,
                    )?;
                    let value_blocks = Tensor::zeros(
                        (num_blocks, block_size, kv_heads, layer_head_dim),
                        dtype,
                        device,
                    )?;
                    cache.push((key_blocks, value_blocks));
                } else {
                    let key_blocks = Tensor::zeros(
                        (num_blocks, kv_heads, layer_head_dim / x, block_size, x),
                        dtype,
                        device,
                    )?;
                    let value_blocks = Tensor::zeros(
                        (num_blocks, kv_heads, layer_head_dim, block_size),
                        dtype,
                        device,
                    )?;
                    cache.push((key_blocks, value_blocks));
                }
            }
            return Ok(cache);
        }

        if use_flash_layout && !model_config.needs_paged_kvcache_layout() {
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
            let kvcache_dtype = model_config.kvcache_dtype;
            if !device.is_cpu() {
                println!(
                    "KV cache dtype: {}, storage dtype {:?}",
                    kvcache_dtype, dtype
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
            (cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards.max(1)).max(1),
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
            (cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards.max(1)).max(1),
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
            (cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) / num_shards.max(1)).max(1),
            head_dim,
        )
    }
}

impl CacheEngine {
    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        if !self.cpu_swap_enabled {
            candle_core::bail!("CPU KV cache swap-in is disabled for this device");
        }
        let started = Instant::now();
        let mut bytes = 0usize;
        if !Self::turboquant_full_mode() {
            for i in 0..self.num_layers {
                let (src_key_cache, src_value_cache) = self.cpu_cache.get(i).unwrap();
                let mut gpu_cache = self.get_kv_cache();
                let (dst_key_cache, dst_value_cache) = gpu_cache.get_mut(i).unwrap();
                bytes += Self::swap_tensor(&src_key_cache, dst_key_cache, &src_to_dst)?;
                bytes += Self::swap_tensor(&src_value_cache, dst_value_cache, &src_to_dst)?;
            }
        }
        bytes += self.swap_turboquant(&src_to_dst, true)?;
        Self::log_swap("in", src_to_dst.len(), bytes, started);
        Ok(())
    }

    pub fn swap_out(&mut self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        if !self.cpu_swap_enabled {
            candle_core::bail!("CPU KV cache swap-out is disabled for this device");
        }
        let started = Instant::now();
        let mut bytes = 0usize;
        if !Self::turboquant_full_mode() {
            for i in 0..self.num_layers {
                let gpu_cache = self.get_kv_cache();
                let (src_key_cache, src_value_cache) = gpu_cache.get(i).unwrap().clone();
                drop(gpu_cache);

                let (dst_key_cache, dst_value_cache) = self.cpu_cache.get_mut(i).unwrap();
                bytes += Self::swap_tensor(&src_key_cache, dst_key_cache, &src_to_dst)?;
                bytes += Self::swap_tensor(&src_value_cache, dst_value_cache, &src_to_dst)?;
            }
        }
        bytes += self.swap_turboquant(&src_to_dst, false)?;
        Self::log_swap("out", src_to_dst.len(), bytes, started);
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

    fn allocate_turboquant_layers(
        model_config: &Config,
        cache_config: &CacheConfig,
        num_blocks: usize,
        device: &Device,
        num_shards: usize,
    ) -> Result<Vec<attention_rs::TurboquantLayerCache>> {
        let tq_mode = match cache_config.kvcache_dtype {
            KvCacheDtype::Turbo8 => attention_rs::TurboquantMode::Turbo8,
            KvCacheDtype::Turbo4 => attention_rs::TurboquantMode::Turbo4,
            KvCacheDtype::Turbo3 => attention_rs::TurboquantMode::Turbo3,
            _ => return Ok(Vec::new()),
        };

        let block_size = cache_config.block_size;
        let num_kv_layers = model_config.kv_cache_num_layers();
        let per_layer_config = model_config.gemma4_per_layer_cache_config();

        let mut tq_layers = Vec::new();
        for layer_idx in 0..num_kv_layers {
            let (kv_heads, hd) = if let Some(ref configs) = per_layer_config {
                let (h, d) = configs[layer_idx];
                ((h / num_shards.max(1)).max(1), d)
            } else {
                (
                    (model_config
                        .num_key_value_heads
                        .unwrap_or(model_config.num_attention_heads)
                        / num_shards.max(1))
                    .max(1),
                    model_config
                        .head_dim
                        .unwrap_or(model_config.hidden_size / model_config.num_attention_heads),
                )
            };

            let v_absmax = Tensor::zeros(
                (num_blocks, block_size, kv_heads),
                candle_core::DType::F32,
                device,
            )?;
            let v_quant = Tensor::zeros(
                (num_blocks, block_size, kv_heads, hd / 2),
                candle_core::DType::U8,
                device,
            )?;

            let (k_absmax, k_quant) = match tq_mode {
                attention_rs::TurboquantMode::Turbo4 => {
                    let ka = Tensor::zeros(
                        (num_blocks, block_size, kv_heads),
                        candle_core::DType::F32,
                        device,
                    )?;
                    let kq = Tensor::zeros(
                        (num_blocks, block_size, kv_heads, hd / 2),
                        candle_core::DType::U8,
                        device,
                    )?;
                    (Some(ka), Some(kq))
                }
                attention_rs::TurboquantMode::Turbo3 => {
                    let ka = Tensor::zeros(
                        (num_blocks, block_size, kv_heads),
                        candle_core::DType::F32,
                        device,
                    )?;
                    let k_bytes_per_head = (hd * 3 + 7) / 8;
                    let kq = Tensor::zeros(
                        (num_blocks, block_size, kv_heads, k_bytes_per_head),
                        candle_core::DType::U8,
                        device,
                    )?;
                    (Some(ka), Some(kq))
                }
                _ => (None, None),
            };

            tq_layers.push(attention_rs::TurboquantLayerCache {
                k_absmax,
                k_quant,
                v_absmax,
                v_quant,
            });
        }

        Ok(tq_layers)
    }

    fn init_turboquant_cache(
        model_config: &Config,
        cache_config: &CacheConfig,
        num_gpu_blocks: usize,
        device: &Device,
        num_shards: usize,
    ) -> Result<()> {
        let layers = Self::allocate_turboquant_layers(
            model_config,
            cache_config,
            num_gpu_blocks,
            device,
            num_shards,
        )?;
        let mode = match cache_config.kvcache_dtype {
            KvCacheDtype::Turbo8 => attention_rs::TurboquantMode::Turbo8,
            KvCacheDtype::Turbo4 => attention_rs::TurboquantMode::Turbo4,
            KvCacheDtype::Turbo3 => attention_rs::TurboquantMode::Turbo3,
            _ => return Ok(()),
        };
        tracing::warn!(
            "Initialized TurboQuant {} cache: {} layers, {} blocks",
            cache_config.kvcache_dtype,
            layers.len(),
            num_gpu_blocks,
        );
        attention_rs::init_turboquant_cache(mode, layers, cache_config.block_size);
        Ok(())
    }

    fn turboquant_full_mode() -> bool {
        matches!(
            attention_rs::get_turboquant_mode(),
            Some(attention_rs::TurboquantMode::Turbo4) | Some(attention_rs::TurboquantMode::Turbo3)
        )
    }

    fn swap_tensor(src: &Tensor, dst: &Tensor, mapping: &HashMap<usize, usize>) -> Result<usize> {
        let bytes_per_block = src
            .elem_count()
            .checked_div(src.dim(0)?)
            .unwrap_or(0)
            .saturating_mul(src.dtype().size_in_bytes());
        attention_rs::cache::swap_blocks(src, dst, mapping)?;
        Ok(bytes_per_block.saturating_mul(mapping.len()))
    }

    fn swap_turboquant(&self, mapping: &HashMap<usize, usize>, swap_in: bool) -> Result<usize> {
        let Some(cpu_layers) = &self.cpu_turboquant_cache else {
            return Ok(0);
        };
        let mut bytes = 0usize;
        for (layer_idx, cpu_layer) in cpu_layers.iter().enumerate() {
            let layer_bytes = attention_rs::with_turboquant_layer(layer_idx, |gpu_layer, _| {
                let mut bytes = 0usize;
                if swap_in {
                    bytes += Self::swap_tensor(&cpu_layer.v_absmax, &gpu_layer.v_absmax, mapping)?;
                    bytes += Self::swap_tensor(&cpu_layer.v_quant, &gpu_layer.v_quant, mapping)?;
                    if let (Some(cpu), Some(gpu)) = (&cpu_layer.k_absmax, &gpu_layer.k_absmax) {
                        bytes += Self::swap_tensor(cpu, gpu, mapping)?;
                    }
                    if let (Some(cpu), Some(gpu)) = (&cpu_layer.k_quant, &gpu_layer.k_quant) {
                        bytes += Self::swap_tensor(cpu, gpu, mapping)?;
                    }
                } else {
                    bytes += Self::swap_tensor(&gpu_layer.v_absmax, &cpu_layer.v_absmax, mapping)?;
                    bytes += Self::swap_tensor(&gpu_layer.v_quant, &cpu_layer.v_quant, mapping)?;
                    if let (Some(gpu), Some(cpu)) = (&gpu_layer.k_absmax, &cpu_layer.k_absmax) {
                        bytes += Self::swap_tensor(gpu, cpu, mapping)?;
                    }
                    if let (Some(gpu), Some(cpu)) = (&gpu_layer.k_quant, &cpu_layer.k_quant) {
                        bytes += Self::swap_tensor(gpu, cpu, mapping)?;
                    }
                }
                Ok::<usize, candle_core::Error>(bytes)
            })
            .transpose()?;
            if let Some(layer_bytes) = layer_bytes {
                bytes = bytes.saturating_add(layer_bytes);
            }
        }
        Ok(bytes)
    }

    fn log_swap(direction: &str, blocks: usize, bytes: usize, started: Instant) {
        let elapsed = started.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let elapsed_ms = elapsed_seconds * 1000.0;
        let bandwidth_gbps = if elapsed_seconds > 0.0 {
            bytes as f64 / elapsed_seconds / 1e9
        } else {
            0.0
        };
        tracing::info!(
            "KV cache swap {}: {} block(s), {:.2} MB, {:.2} ms, {:.2} GB/s",
            direction,
            blocks,
            bytes as f64 / 1024.0 / 1024.0,
            elapsed_ms,
            bandwidth_gbps,
        );
    }
}
