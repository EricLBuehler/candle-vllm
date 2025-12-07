//! Mock tensor creation utilities for testing.

use candle_core::{DType, Device, Shape, Tensor};

/// Create a mock tensor for testing.
pub fn create_mock_tensor(shape: &[usize], device: &Device) -> candle_core::Result<Tensor> {
    Tensor::zeros(shape, DType::F32, device)
}

/// Create a mock KV cache tensor.
pub fn create_mock_kv_cache(
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    device: &Device,
) -> candle_core::Result<Vec<(Tensor, Tensor)>> {
    let mut cache = Vec::new();
    for _ in 0..num_layers {
        let k = create_mock_tensor(&[1, num_heads, max_seq_len, head_dim], device)?;
        let v = create_mock_tensor(&[1, num_heads, max_seq_len, head_dim], device)?;
        cache.push((k, v));
    }
    Ok(cache)
}

/// Mock device for testing.
pub struct MockDevice;

impl MockDevice {
    /// Get a CPU device for testing.
    pub fn cpu() -> Device {
        Device::Cpu
    }
    
    /// Get a GPU device if available, otherwise CPU.
    #[allow(dead_code)]
    pub fn gpu_or_cpu() -> Device {
        #[cfg(feature = "cuda")]
        {
            use candle_core::CudaDevice;
            if let Ok(device) = CudaDevice::new(0) {
                return Device::Cuda(device);
            }
        }
        
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return device;
            }
        }
        
        Device::Cpu
    }
}
