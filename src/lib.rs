#![warn(clippy::cast_lossless)]
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result};
use candle_core as candle;
use std::path::Path;
use tracing::warn;
pub mod backend;
pub mod openai;
// pub mod paged_attention;
pub mod scheduler;
pub use attention_rs::{InputMetadata, PagedAttention};

pub mod api;

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
    let dsize = kv_dtype.size_in_bytes();
    let size_in_mb = 1024 * 1024;
    let num_gpu_blocks = kvcache_mem_gpu * size_in_mb
        / dsize
        / block_size
        / (config.num_key_value_heads.unwrap() / num_shards)
        / config.k_head_dim()
        / config.num_hidden_layers
        / 2;
    let num_cpu_blocks = kvcache_mem_cpu * size_in_mb
        / dsize
        / block_size
        / (config.num_key_value_heads.unwrap() / num_shards)
        / config.k_head_dim()
        / config.num_hidden_layers
        / 2;
    crate::scheduler::cache_engine::CacheConfig {
        block_size,
        num_gpu_blocks: Some(num_gpu_blocks),
        num_cpu_blocks: Some(num_cpu_blocks),
        fully_init: true,
        dtype: kv_dtype,
        kvcache_mem_gpu,
    }
}
pub mod mcp;
pub mod tools;
