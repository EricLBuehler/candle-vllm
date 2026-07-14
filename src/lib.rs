#![warn(clippy::cast_lossless)]
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result};
use candle_core as candle;
use std::path::Path;
use tracing::warn;
pub mod backend;
pub mod openai;
pub mod scheduler;
pub use attention_rs::{InputMetadata, PagedAttention};

pub mod api;

#[cfg(feature = "flashinfer")]
#[derive(Clone, Copy, Debug)]
pub struct FlashInferKvParams {
    pub kv_dtype: candle::DType,
    pub out_dtype: candle::DType,
    pub page_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_qo_heads: usize,
}

pub fn get_dtype(dtype: Option<String>) -> candle::DType {
    let dtype = match dtype.as_deref() {
        Some("f16") => candle::DType::F16,
        Some("bf16") => candle::DType::BF16,
        Some("f32") => candle::DType::F32,
        Some(dtype) => panic!("Unsupported dtype {dtype}"),
        None => candle::DType::BF16,
    };

    #[cfg(feature = "cuda")]
    let dtype = {
        use candle_core::cuda_backend::cudarc::driver::result::{device, init};
        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
        match (init(), device::get(0)) {
            (Ok(_), Ok(d)) => {
                let (compute_major, _compute_minor) = unsafe {
                    (
                        device::get_attribute(
                            d,
                            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        )
                        .unwrap_or(8),
                        device::get_attribute(
                            d,
                            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        )
                        .unwrap_or(8),
                    )
                };
                if dtype != candle::DType::F32 && compute_major < 8 {
                    tracing::warn!(
                        "CUDA compute capability: {} (<8), switched to F16 cause no BF16 support.",
                        compute_major
                    );
                    candle::DType::F16
                } else {
                    dtype
                }
            }
            _ => dtype,
        }
    };
    dtype
}
pub fn hub_load_local_safetensors(
    path: &String,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    tracing::info!("{:}", Path::new(path).join(json_file).display());
    let jsfile = std::fs::File::open(Path::new(path).join(json_file))?;
    let reader = std::io::BufReader::new(jsfile);

    #[derive(serde::Deserialize)]
    struct IndexFile {
        weight_map: std::collections::HashMap<String, String>,
    }

    let index: IndexFile = serde_json::from_reader(reader).map_err(candle::Error::wrap)?;
    let safetensors_files: Vec<_> = index
        .weight_map
        .into_values()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .map(|v| Path::new(path).join(v))
        .collect();
    Ok(safetensors_files)
}

pub fn new_device(ordinal: usize) -> Result<Device> {
    if cuda_is_available() {
        use candle_core::CudaDevice;
        let device = Device::Cuda(CudaDevice::new_with_stream(ordinal)?);
        Ok(device)
    } else if metal_is_available() {
        Ok(Device::new_metal(ordinal)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            warn!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            warn!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn get_cache_config(
    kvcache_mem_gpu: usize,
    kvcache_mem_cpu: usize,
    block_size: usize,
    config: &crate::openai::models::Config,
    kv_dtype: candle::DType,
    num_shards: usize,
    kvcache_dtype: crate::openai::models::KvCacheDtype,
) -> crate::scheduler::cache_engine::CacheConfig {
    use crate::openai::models::KvCacheDtype;

    let kvcache_dtype =
        if kvcache_dtype.is_turboquant() && !cfg!(feature = "cuda") && !cfg!(feature = "metal") {
            tracing::warn!(
                "TurboQuant ({}) requires CUDA or Metal. Falling back to auto.",
                kvcache_dtype
            );
            KvCacheDtype::Auto
        } else if config.is_mla() && kvcache_dtype.is_turboquant() {
            tracing::warn!(
                "TurboQuant ({}) is not supported for MLA models. Falling back to auto.",
                kvcache_dtype
            );
            KvCacheDtype::Auto
        } else {
            kvcache_dtype
        };

    let kv_layers = config.kv_cache_num_layers().max(1);
    let dsize = kv_dtype.size_in_bytes();
    let size_in_mb = 1024 * 1024;
    let kv_heads_per_shard = |global_heads: usize| {
        let shards = num_shards.max(1);
        if global_heads < shards {
            1
        } else {
            global_heads / shards
        }
    };

    let tq_full = matches!(kvcache_dtype, KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3);

    let base_per_block = if tq_full {
        0
    } else if config.is_mla() {
        block_size * (config.mla_kv_lora_rank() + config.mla_qk_rope_head_dim()) * dsize * kv_layers
    } else if let Some(ref per_layer_cfg) = config.gemma4_per_layer_cache_config() {
        let mut total = 0usize;
        for &(kv_heads, head_dim) in per_layer_cfg {
            let kv_heads_sharded = kv_heads_per_shard(kv_heads);
            total += block_size * kv_heads_sharded * head_dim * dsize * 2;
        }
        total
    } else {
        dsize
            * block_size
            * kv_heads_per_shard(config.num_key_value_heads.unwrap())
            * config.k_head_dim()
            * kv_layers
            * 2
    };

    let tq_per_block = match kvcache_dtype {
        KvCacheDtype::Turbo8 => {
            if let Some(ref per_layer_cfg) = config.gemma4_per_layer_cache_config() {
                per_layer_cfg
                    .iter()
                    .map(|&(kv_heads, hd)| {
                        let heads = kv_heads_per_shard(kv_heads);
                        block_size * heads * 4 + block_size * heads * (hd / 2)
                    })
                    .sum()
            } else {
                let heads = kv_heads_per_shard(config.num_key_value_heads.unwrap());
                let hd = config.k_head_dim();
                (block_size * heads * 4 + block_size * heads * (hd / 2)) * kv_layers
            }
        }
        KvCacheDtype::Turbo4 => {
            if let Some(ref per_layer_cfg) = config.gemma4_per_layer_cache_config() {
                per_layer_cfg
                    .iter()
                    .map(|&(kv_heads, hd)| {
                        let heads = kv_heads_per_shard(kv_heads);
                        block_size * heads * 4 * 2 + block_size * heads * (hd / 2) * 2
                    })
                    .sum()
            } else {
                let heads = kv_heads_per_shard(config.num_key_value_heads.unwrap());
                let hd = config.k_head_dim();
                (block_size * heads * 4 * 2 + block_size * heads * (hd / 2) * 2) * kv_layers
            }
        }
        KvCacheDtype::Turbo3 => {
            if let Some(ref per_layer_cfg) = config.gemma4_per_layer_cache_config() {
                per_layer_cfg
                    .iter()
                    .map(|&(kv_heads, hd)| {
                        let heads = kv_heads_per_shard(kv_heads);
                        block_size * heads * 4 * 2
                            + block_size * heads * ((hd * 3 + 7) / 8)
                            + block_size * heads * (hd / 2)
                    })
                    .sum()
            } else {
                let heads = kv_heads_per_shard(config.num_key_value_heads.unwrap());
                let hd = config.k_head_dim();
                (block_size * heads * 4 * 2
                    + block_size * heads * ((hd * 3 + 7) / 8)
                    + block_size * heads * (hd / 2))
                    * kv_layers
            }
        }
        _ => 0,
    };

    let per_block = (base_per_block + tq_per_block).max(1);
    let num_gpu_blocks = kvcache_mem_gpu * size_in_mb / per_block;
    // Match xInfer's default CPU swap policy: reserve half as many CPU KV
    // blocks as GPU KV blocks. A non-zero `kvcache_mem_cpu` remains an
    // explicit megabyte override for callers that need a fixed budget.
    let num_cpu_blocks = if cfg!(feature = "cuda") {
        if kvcache_mem_cpu == 0 {
            num_gpu_blocks / 2
        } else {
            kvcache_mem_cpu * size_in_mb / per_block
        }
    } else {
        0
    };
    tracing::info!(
        "KV cache block allocation: GPU {} block(s), CPU {} block(s) ({})",
        num_gpu_blocks,
        num_cpu_blocks,
        if !cfg!(feature = "cuda") {
            "CPU swap disabled for non-CUDA device"
        } else if kvcache_mem_cpu == 0 {
            "CPU default 0.5x GPU blocks"
        } else {
            "explicit CPU memory budget"
        }
    );

    crate::scheduler::cache_engine::CacheConfig {
        block_size,
        num_gpu_blocks: Some(num_gpu_blocks),
        num_cpu_blocks: Some(num_cpu_blocks),
        fully_init: true,
        dtype: kv_dtype,
        kvcache_dtype,
        kvcache_mem_gpu,
        mamba_cache_budget_bytes: 0,
    }
}

const SIZE_IN_MB: usize = 1024 * 1024;
const MIN_ACTIVATION_RESERVE_BYTES: usize = 256 * 1024 * 1024; // 256 MB floor

#[derive(Debug, Clone)]
pub struct GpuMemoryBudget {
    pub flashinfer_bytes: usize,
    pub cutlass_bytes: usize,
    pub moe_pool_bytes: usize,
    pub flash_splitk_bytes: usize,
    pub transient_bytes: usize,
    pub total_bytes: usize,
}

pub struct WorkspaceBudgetParams {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_layers: usize,
    pub head_dim: usize,
    pub prefill_chunk_size: usize,
    pub model_dtype_size: usize,
    pub num_shards: usize,
    pub is_moe: bool,
    pub moe_num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
}

impl WorkspaceBudgetParams {
    pub fn from_config(
        config: &crate::openai::models::Config,
        model_dtype: candle::DType,
        num_shards: usize,
        prefill_chunk_size: usize,
    ) -> Self {
        use crate::openai::models::MoEConfig;
        let (is_moe, moe_num_experts_per_tok, moe_intermediate_size) = match &config.moe_config {
            Some(MoEConfig::QwenMoE(cfg)) => {
                (true, cfg.num_experts_per_tok, cfg.moe_intermediate_size)
            }
            Some(MoEConfig::DeepSeekMoE(cfg)) => (
                true,
                cfg.num_experts_per_tok.unwrap_or(0),
                cfg.moe_intermediate_size,
            ),
            None => (false, 0, 0),
        };
        Self {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_kv_layers: config.kv_cache_num_layers().max(1),
            head_dim: config.get_head_size(),
            prefill_chunk_size,
            model_dtype_size: model_dtype.size_in_bytes(),
            num_shards: num_shards.max(1),
            is_moe,
            moe_num_experts_per_tok,
            moe_intermediate_size,
        }
    }
}

pub fn compute_workspace_budget(params: &WorkspaceBudgetParams) -> GpuMemoryBudget {
    let flashinfer_bytes: usize = if cfg!(feature = "flashinfer") {
        (512 + 128) * 1024 * 1024
    } else {
        0
    };

    let cutlass_bytes: usize = if cfg!(feature = "cutlass") {
        512 * 1024 * 1024
    } else {
        0
    };

    let moe_pool_bytes: usize =
        if cfg!(feature = "cutlass") && params.is_moe && params.moe_num_experts_per_tok > 0 {
            let topk = params.moe_num_experts_per_tok;
            let size_m = params.prefill_chunk_size * topk;
            let hidden = params.hidden_size / params.num_shards;
            let inter = params.moe_intermediate_size / params.num_shards;
            let largest_dim = hidden.max(2 * inter);
            let gathered = size_m * hidden * params.model_dtype_size;
            let rep_out = size_m * inter * params.model_dtype_size;
            let act_packed = size_m * hidden / 2;
            let act_scales = size_m * (hidden / 16 + 128);
            let pool_total = gathered + rep_out + act_packed + act_scales;
            let transient_output = size_m * largest_dim * params.model_dtype_size;
            pool_total + transient_output
        } else {
            0
        };

    let flash_splitk_bytes: usize = if cfg!(feature = "flash") || cfg!(feature = "flashattn") {
        let q_heads_per_shard = params.num_attention_heads / params.num_shards;
        let splits = 8usize;
        let per_layer = 64 * q_heads_per_shard * splits * (params.head_dim + 2) * 4;
        per_layer * params.num_kv_layers
    } else {
        0
    };

    let transient_bytes =
        2 * params.prefill_chunk_size * params.hidden_size * params.model_dtype_size;

    let total_bytes =
        (flashinfer_bytes + cutlass_bytes + moe_pool_bytes + flash_splitk_bytes + transient_bytes)
            .max(MIN_ACTIVATION_RESERVE_BYTES);

    GpuMemoryBudget {
        flashinfer_bytes,
        cutlass_bytes,
        moe_pool_bytes,
        flash_splitk_bytes,
        transient_bytes,
        total_bytes,
    }
}

pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "CANDLE_VLLM_MAMBA_SNAPSHOT_STRIDE_BLOCKS";

pub const STREAM_AS_REASONING_CONTENT_ENV: &str = "CANDLE_VLLM_STREAM_AS_REASONING_CONTENT";

static STREAM_AS_REASONING_CONTENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn stream_as_reasoning_content() -> bool {
    *STREAM_AS_REASONING_CONTENT.get_or_init(|| {
        std::env::var(STREAM_AS_REASONING_CONTENT_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(true)
    })
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceMemoryReport {
    pub total_bytes: usize,
    pub free_bytes: usize,
    pub used_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct HybridMambaCacheEstimate {
    pub slot_bytes: usize,
    pub num_gdn_layers: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct HybridMambaCachePlan {
    pub active_slot_capacity: usize,
    pub prefix_slot_capacity: usize,
    pub budget_bytes: usize,
}

const DEFAULT_HYBRID_MAMBA_FRACTION: f32 = 0.15;
const MAX_HYBRID_MAMBA_FRACTION: f32 = 0.3;
const HYBRID_MAMBA_PREFIX_SLOT_MULTIPLIER: usize = 2;
const HYBRID_MAMBA_MIN_ACTIVE_SLOTS: usize = 8;

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
fn compute_kvcache_budget_bytes(free_bytes: usize, fraction: f32) -> Result<usize> {
    if !(0.0 < fraction && fraction <= 1.0) {
        return Err(candle::Error::msg(format!(
            "kv_fraction must be in (0, 1], got {fraction}"
        )));
    }

    Ok(((free_bytes as f64) * (fraction as f64)).round() as usize)
}

pub fn query_device_memory(device: &Device) -> Result<DeviceMemoryReport> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda) => {
            use candle_core::cuda_backend::cudarc::driver::result::mem_get_info;

            cuda.bind_to_thread().map_err(candle::Error::wrap)?;
            let (free_bytes, total_bytes) = mem_get_info().map_err(candle::Error::wrap)?;
            let used_bytes = total_bytes.saturating_sub(free_bytes);
            Ok(DeviceMemoryReport {
                total_bytes,
                free_bytes,
                used_bytes,
            })
        }
        #[cfg(feature = "metal")]
        Device::Metal(metal) => {
            let total_bytes = metal.recommended_max_working_set_size() as usize;
            let used_bytes = metal.current_allocated_size() as usize;
            let free_bytes = total_bytes.saturating_sub(used_bytes) as usize;
            Ok(DeviceMemoryReport {
                total_bytes,
                free_bytes,
                used_bytes,
            })
        }
        Device::Cpu => Err(candle::Error::msg(
            "kv_fraction requires a CUDA or Metal device",
        )),
        #[allow(unreachable_patterns)]
        _ => Err(candle::Error::msg(
            "kv_fraction is not supported on this backend",
        )),
    }
}

pub fn query_device_memory_for_devices(devices: &[&Device]) -> Result<Vec<DeviceMemoryReport>> {
    devices
        .iter()
        .map(|device| query_device_memory(device))
        .collect()
}

pub fn detect_kvcache_mem_gpu_mb(device: &Device, kv_fraction: f32) -> Result<usize> {
    let report = query_device_memory(device)?;
    let budget_bytes = compute_kvcache_budget_bytes(report.free_bytes, kv_fraction)?;

    let budget_mb = budget_bytes / SIZE_IN_MB;
    if budget_mb == 0 {
        return Err(candle::Error::msg(format!(
            "kv_fraction {} leaves no room for KV cache after model load",
            kv_fraction
        )));
    }

    Ok(budget_mb)
}

pub fn detect_kvcache_mem_gpu_mb_for_devices(
    devices: &[&Device],
    kv_fraction: f32,
) -> Result<usize> {
    detect_kvcache_mem_gpu_mb_for_devices_with_workspace(devices, kv_fraction, None)
}

pub fn detect_kvcache_mem_gpu_mb_for_devices_with_workspace(
    devices: &[&Device],
    kv_fraction: f32,
    workspace: Option<&GpuMemoryBudget>,
) -> Result<usize> {
    let mut min_budget_mb: Option<usize> = None;
    let reports = query_device_memory_for_devices(devices)?;
    let workspace_bytes = workspace.map_or(0, |w| w.total_bytes);

    for (rank, report) in reports.iter().enumerate() {
        let usable_bytes = compute_kvcache_budget_bytes(report.free_bytes, kv_fraction)?;
        let cache_bytes = usable_bytes.saturating_sub(workspace_bytes);
        let budget_mb = cache_bytes / SIZE_IN_MB;

        tracing::info!(
            "Rank {} GPU memory: total {:.2} GB, free {:.2} GB, used {:.2} GB, \
             kv_fraction {:.0}% -> usable {:.2} GB, workspace reserve {:.2} GB, \
             cache budget {:.2} GB",
            rank,
            report.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            report.free_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            report.used_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            kv_fraction as f64 * 100.0,
            usable_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            workspace_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            cache_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        );

        min_budget_mb = Some(min_budget_mb.map_or(budget_mb, |current| current.min(budget_mb)));
    }

    min_budget_mb.ok_or_else(|| candle::Error::msg("No devices available for KV cache sizing"))
}

pub fn estimate_hybrid_mamba_cache(
    config: &crate::openai::models::Config,
    model_dtype: candle::DType,
    num_shards: usize,
) -> Option<HybridMambaCacheEstimate> {
    let hybrid = crate::openai::models::resolve_qwen3_hybrid_config(config);
    let num_gdn_layers = hybrid
        .layer_types
        .iter()
        .filter(|layer_type| layer_type.as_str() == "linear_attention")
        .count();
    if num_gdn_layers == 0 {
        return None;
    }

    let shard_count = num_shards.max(1);
    let num_v_heads = std::cmp::max(1, hybrid.num_v_heads / shard_count);
    let num_k_heads = std::cmp::max(1, hybrid.num_k_heads / shard_count);
    let conv_window = hybrid.conv_kernel_size.saturating_sub(1);
    let d_conv = num_k_heads
        .saturating_mul(hybrid.key_head_dim)
        .saturating_mul(2)
        .saturating_add(num_v_heads.saturating_mul(hybrid.value_head_dim));
    let conv_bytes = d_conv
        .saturating_mul(conv_window)
        .saturating_mul(model_dtype.size_in_bytes());
    let recurrent_bytes = num_v_heads
        .saturating_mul(hybrid.key_head_dim)
        .saturating_mul(hybrid.value_head_dim)
        .saturating_mul(candle::DType::F32.size_in_bytes());
    let slot_bytes = num_gdn_layers.saturating_mul(conv_bytes.saturating_add(recurrent_bytes));
    if slot_bytes == 0 {
        return None;
    }

    Some(HybridMambaCacheEstimate {
        slot_bytes,
        num_gdn_layers,
    })
}

pub fn plan_hybrid_mamba_cache(
    total_cache_budget_bytes: usize,
    estimate: HybridMambaCacheEstimate,
    min_active_slots: usize,
    prefix_cache_enabled: bool,
) -> Option<HybridMambaCachePlan> {
    plan_hybrid_mamba_cache_with_fraction(
        total_cache_budget_bytes,
        estimate,
        min_active_slots,
        prefix_cache_enabled,
        None,
    )
}

pub fn plan_hybrid_mamba_cache_with_fraction(
    total_cache_budget_bytes: usize,
    estimate: HybridMambaCacheEstimate,
    min_active_slots: usize,
    prefix_cache_enabled: bool,
    mamba_fraction: Option<f32>,
) -> Option<HybridMambaCachePlan> {
    if total_cache_budget_bytes == 0 || estimate.slot_bytes == 0 {
        return None;
    }
    let mamba_fraction = mamba_fraction
        .unwrap_or(DEFAULT_HYBRID_MAMBA_FRACTION)
        .clamp(0.0, MAX_HYBRID_MAMBA_FRACTION);
    if mamba_fraction <= 0.0 {
        return None;
    }

    let active_slot_target = if prefix_cache_enabled {
        min_active_slots.max(HYBRID_MAMBA_MIN_ACTIVE_SLOTS)
    } else {
        min_active_slots.max(1)
    };
    let min_prefix_slot_target = if prefix_cache_enabled {
        active_slot_target.saturating_mul(HYBRID_MAMBA_PREFIX_SLOT_MULTIPLIER)
    } else {
        0
    };
    let baseline_budget_bytes = active_slot_target
        .saturating_add(min_prefix_slot_target)
        .saturating_mul(estimate.slot_bytes);
    let target_budget_bytes =
        ((total_cache_budget_bytes as f64) * (mamba_fraction as f64)).round() as usize;
    let budget_bytes = target_budget_bytes.max(baseline_budget_bytes);
    let total_slot_capacity = budget_bytes / estimate.slot_bytes;
    let active_slot_capacity = if prefix_cache_enabled {
        active_slot_target
            .min(total_slot_capacity.saturating_sub(min_prefix_slot_target))
            .max(1)
    } else {
        total_slot_capacity.max(1)
    };
    let prefix_slot_capacity = if prefix_cache_enabled {
        total_slot_capacity.saturating_sub(active_slot_capacity)
    } else {
        0
    };

    Some(HybridMambaCachePlan {
        active_slot_capacity,
        prefix_slot_capacity,
        budget_bytes,
    })
}

pub fn mamba_snapshot_block_stride_blocks(default: usize) -> usize {
    let default = default.max(1);
    let Ok(raw) = std::env::var(MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV) else {
        return default;
    };
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            tracing::warn!(
                "{} must be >= 1, got 0. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                default
            );
            default
        }
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                "Invalid {}='{}'. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                raw,
                default
            );
            default
        }
    }
}
pub mod mcp;
pub mod tools;

#[cfg(test)]
mod tests {
    use super::{
        compute_kvcache_budget_bytes, plan_hybrid_mamba_cache_with_fraction,
        HybridMambaCacheEstimate,
    };

    #[test]
    fn test_compute_kvcache_budget_bytes() {
        let free = 700usize;
        let budget = compute_kvcache_budget_bytes(free, 0.9).unwrap();
        assert_eq!(budget, 630);
    }

    #[test]
    fn test_compute_kvcache_budget_bytes_clamps_to_zero() {
        let free = 0usize;
        let budget = compute_kvcache_budget_bytes(free, 0.9).unwrap();
        assert_eq!(budget, 0);
    }

    #[test]
    fn test_hybrid_mamba_plan_keeps_two_prefix_slots_per_active_slot() {
        let estimate = HybridMambaCacheEstimate {
            slot_bytes: 10,
            num_gdn_layers: 1,
        };
        let plan =
            plan_hybrid_mamba_cache_with_fraction(1_000, estimate, 16, true, Some(0.15)).unwrap();

        assert_eq!(plan.active_slot_capacity, 16);
        assert_eq!(plan.prefix_slot_capacity, 32);
        assert_eq!(plan.budget_bytes, 480);
    }

    #[test]
    fn test_hybrid_mamba_plan_adds_fraction_leftover_to_prefix_slots() {
        let estimate = HybridMambaCacheEstimate {
            slot_bytes: 10,
            num_gdn_layers: 1,
        };
        let plan =
            plan_hybrid_mamba_cache_with_fraction(2_000, estimate, 16, true, Some(0.3)).unwrap();

        assert_eq!(plan.active_slot_capacity, 16);
        assert_eq!(plan.prefix_slot_capacity, 44);
        assert_eq!(plan.budget_bytes, 600);
    }
}
