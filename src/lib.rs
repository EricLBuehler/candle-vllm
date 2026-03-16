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
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files
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
) -> crate::scheduler::cache_engine::CacheConfig {
    let kv_layers = config.kv_cache_num_layers().max(1);
    let dsize = kv_dtype.size_in_bytes();
    let size_in_mb = 1024 * 1024;
    let num_gpu_blocks = kvcache_mem_gpu * size_in_mb
        / dsize
        / block_size
        / (config.num_key_value_heads.unwrap() / num_shards)
        / config.k_head_dim()
        / kv_layers
        / 2;
    let num_cpu_blocks = kvcache_mem_cpu * size_in_mb
        / dsize
        / block_size
        / (config.num_key_value_heads.unwrap() / num_shards)
        / config.k_head_dim()
        / kv_layers
        / 2;
    crate::scheduler::cache_engine::CacheConfig {
        block_size,
        num_gpu_blocks: Some(num_gpu_blocks),
        num_cpu_blocks: Some(num_cpu_blocks),
        fully_init: true,
        dtype: kv_dtype,
        kvcache_mem_gpu,
        mamba_cache_budget_bytes: 0,
    }
}

const SIZE_IN_MB: usize = 1024 * 1024;
pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "VLLM_RS_MAMBA_SNAPSHOT_STRIDE_BLOCKS";
pub const DEFAULT_MAMBA_SNAPSHOT_BLOCK_STRIDE: usize = 8;

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

const DEFAULT_HYBRID_MAMBA_FRACTION: f32 = 0.1;

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
fn compute_kvcache_budget_bytes(free_bytes: usize, fraction: f32) -> Result<usize> {
    if !(0.0 < fraction && fraction <= 1.0) {
        return Err(candle::Error::msg(format!(
            "gpu_memory_fraction must be in (0, 1], got {fraction}"
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
            "gpu_memory_fraction requires a CUDA or Metal device",
        )),
        #[allow(unreachable_patterns)]
        _ => Err(candle::Error::msg(
            "gpu_memory_fraction is not supported on this backend",
        )),
    }
}

pub fn query_device_memory_for_devices(devices: &[&Device]) -> Result<Vec<DeviceMemoryReport>> {
    devices
        .iter()
        .map(|device| query_device_memory(device))
        .collect()
}

pub fn detect_kvcache_mem_gpu_mb(device: &Device, gpu_memory_fraction: f32) -> Result<usize> {
    let report = query_device_memory(device)?;
    let budget_bytes = compute_kvcache_budget_bytes(report.free_bytes, gpu_memory_fraction)?;

    let budget_mb = budget_bytes / SIZE_IN_MB;
    if budget_mb == 0 {
        return Err(candle::Error::msg(format!(
            "gpu_memory_fraction {} leaves no room for KV cache after model load",
            gpu_memory_fraction
        )));
    }

    Ok(budget_mb)
}

pub fn detect_kvcache_mem_gpu_mb_for_devices(
    devices: &[&Device],
    gpu_memory_fraction: f32,
) -> Result<usize> {
    let mut min_budget_mb: Option<usize> = None;
    let reports = query_device_memory_for_devices(devices)?;
    for (rank, report) in reports.iter().enumerate() {
        let budget_bytes = compute_kvcache_budget_bytes(report.free_bytes, gpu_memory_fraction)?;
        let budget_mb = budget_bytes / SIZE_IN_MB;
        tracing::info!(
            "Rank {} GPU memory after model load: total {:.2} GB, free {:.2} GB, used {:.2} GB, gpu_memory_fraction {:.0}% of remaining memory, usable combined cache budget {:.2} GB",
            rank,
            report.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            report.free_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            report.used_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            gpu_memory_fraction as f64 * 100.0,
            budget_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
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
    if total_cache_budget_bytes == 0 || estimate.slot_bytes == 0 {
        return None;
    }

    let active_slot_capacity = min_active_slots.max(1);
    let baseline_prefix_slots = if prefix_cache_enabled {
        active_slot_capacity
    } else {
        0
    };
    let baseline_budget_bytes = active_slot_capacity
        .saturating_add(baseline_prefix_slots)
        .saturating_mul(estimate.slot_bytes);
    let target_budget_bytes = ((total_cache_budget_bytes as f64)
        * (DEFAULT_HYBRID_MAMBA_FRACTION as f64))
        .round() as usize;
    let budget_bytes = target_budget_bytes.max(baseline_budget_bytes);
    let total_slot_capacity = budget_bytes / estimate.slot_bytes;
    let prefix_slot_capacity = if prefix_cache_enabled && total_slot_capacity > active_slot_capacity
    {
        total_slot_capacity - active_slot_capacity
    } else {
        0
    };

    Some(HybridMambaCachePlan {
        active_slot_capacity,
        prefix_slot_capacity,
        budget_bytes,
    })
}

pub fn mamba_snapshot_block_stride_blocks() -> usize {
    let default = DEFAULT_MAMBA_SNAPSHOT_BLOCK_STRIDE;
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
    use super::compute_kvcache_budget_bytes;

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
}
