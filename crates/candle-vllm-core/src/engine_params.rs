use serde::{Deserialize, Serialize};

/// Unified engine parameters for model initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineParams {
    /// Data type for model weights and activations
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Quantization method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,

    /// Block size for attention computation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,

    /// Maximum number of concurrent sequences
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_num_seqs: Option<usize>,

    /// GPU memory allocation for KV cache (in MB)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem: Option<usize>,

    /// CPU memory allocation for KV cache (in MB)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kvcache_mem_cpu: Option<usize>,

    /// Prefill chunk size for long contexts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_chunk_size: Option<usize>,

    /// Enable multithreading
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multithread: Option<bool>,

    /// Device IDs for multi-GPU setup
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_ids: Option<Vec<usize>>,

    /// Sampling temperature
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p sampling parameter
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling parameter
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<isize>,

    /// Frequency penalty
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// In-situ quantization method
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub isq: Option<String>,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            dtype: Some("bf16".to_string()),
            quantization: None,
            block_size: Some(16),
            max_num_seqs: Some(16),
            mem: Some(1024), // 1GB default
            kvcache_mem_cpu: None,
            prefill_chunk_size: Some(8192),
            multithread: Some(true),
            device_ids: None,
            temperature: Some(1.0),
            top_p: Some(1.0),
            top_k: Some(-1),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            isq: None,
        }
    }
}

impl EngineParams {
    /// Create engine parameters with sensible defaults for text models
    pub fn text_model_defaults() -> Self {
        Self::default()
    }

    /// Create engine parameters with defaults optimized for vision models
    pub fn vision_model_defaults() -> Self {
        Self {
            dtype: Some("bf16".to_string()),
            quantization: None,
            block_size: Some(16),
            max_num_seqs: Some(4), // Lower for vision models due to memory
            mem: Some(8192),       // 8GB for vision models
            kvcache_mem_cpu: None,
            prefill_chunk_size: Some(4096), // Smaller chunks for images
            multithread: Some(true),
            device_ids: None,
            temperature: Some(0.1), // Lower temperature for vision descriptions
            top_p: Some(0.9),
            top_k: Some(-1),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            isq: None,
        }
    }

    /// Merge with another EngineParams, preferring values from `other` when present
    pub fn merge_with(mut self, other: &EngineParams) -> Self {
        if other.dtype.is_some() {
            self.dtype = other.dtype.clone();
        }
        if other.quantization.is_some() {
            self.quantization = other.quantization.clone();
        }
        if other.block_size.is_some() {
            self.block_size = other.block_size;
        }
        if other.max_num_seqs.is_some() {
            self.max_num_seqs = other.max_num_seqs;
        }
        if other.mem.is_some() {
            self.mem = other.mem;
        }
        if other.kvcache_mem_cpu.is_some() {
            self.kvcache_mem_cpu = other.kvcache_mem_cpu;
        }
        if other.prefill_chunk_size.is_some() {
            self.prefill_chunk_size = other.prefill_chunk_size;
        }
        if other.multithread.is_some() {
            self.multithread = other.multithread;
        }
        if other.device_ids.is_some() {
            self.device_ids = other.device_ids.clone();
        }
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        if other.top_p.is_some() {
            self.top_p = other.top_p;
        }
        if other.top_k.is_some() {
            self.top_k = other.top_k;
        }
        if other.frequency_penalty.is_some() {
            self.frequency_penalty = other.frequency_penalty;
        }
        if other.presence_penalty.is_some() {
            self.presence_penalty = other.presence_penalty;
        }
        if other.isq.is_some() {
            self.isq = other.isq.clone();
        }
        self
    }

    /// Validate the parameters for consistency
    pub fn validate(&self) -> Result<(), String> {
        // Validate memory settings
        if let Some(mem) = self.mem {
            if mem < 256 {
                return Err("Memory allocation too low, minimum 256MB required".to_string());
            }
        }

        // Validate max_num_seqs
        if let Some(seqs) = self.max_num_seqs {
            if seqs == 0 {
                return Err("max_num_seqs must be greater than 0".to_string());
            }
        }

        // Validate temperature
        if let Some(temp) = self.temperature {
            if temp < 0.0 || temp > 2.0 {
                return Err("Temperature must be between 0.0 and 2.0".to_string());
            }
        }

        // Validate top_p
        if let Some(top_p) = self.top_p {
            if top_p < 0.0 || top_p > 1.0 {
                return Err("top_p must be between 0.0 and 1.0".to_string());
            }
        }

        // Validate penalties
        if let Some(freq_penalty) = self.frequency_penalty {
            if freq_penalty < -2.0 || freq_penalty > 2.0 {
                return Err("frequency_penalty must be between -2.0 and 2.0".to_string());
            }
        }

        if let Some(pres_penalty) = self.presence_penalty {
            if pres_penalty < -2.0 || pres_penalty > 2.0 {
                return Err("presence_penalty must be between -2.0 and 2.0".to_string());
            }
        }

        Ok(())
    }

    /// Get memory allocation in MB, with fallback to default
    pub fn get_mem_mb(&self) -> usize {
        self.mem.unwrap_or(1024)
    }

    /// Get max number of sequences, with fallback to default
    pub fn get_max_num_seqs(&self) -> usize {
        self.max_num_seqs.unwrap_or(16)
    }

    /// Get data type, with fallback to default
    pub fn get_dtype(&self) -> &str {
        self.dtype.as_deref().unwrap_or("bf16")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = EngineParams::default();
        assert_eq!(params.get_dtype(), "bf16");
        assert_eq!(params.get_mem_mb(), 1024);
        assert_eq!(params.get_max_num_seqs(), 16);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_vision_model_defaults() {
        let params = EngineParams::vision_model_defaults();
        assert_eq!(params.get_mem_mb(), 8192);
        assert_eq!(params.get_max_num_seqs(), 4);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_merge_params() {
        let base = EngineParams::default();
        let override_params = EngineParams {
            mem: Some(2048),
            temperature: Some(0.5),
            ..Default::default()
        };

        let merged = base.merge_with(&override_params);
        assert_eq!(merged.get_mem_mb(), 2048);
        assert_eq!(merged.temperature, Some(0.5));
        assert_eq!(merged.get_dtype(), "bf16"); // Should retain original
    }

    #[test]
    fn test_validation() {
        // Valid params
        let valid = EngineParams::default();
        assert!(valid.validate().is_ok());

        // Invalid memory
        let invalid_mem = EngineParams {
            mem: Some(100),
            ..Default::default()
        };
        assert!(invalid_mem.validate().is_err());

        // Invalid temperature
        let invalid_temp = EngineParams {
            temperature: Some(3.0),
            ..Default::default()
        };
        assert!(invalid_temp.validate().is_err());

        // Invalid max_num_seqs
        let invalid_seqs = EngineParams {
            max_num_seqs: Some(0),
            ..Default::default()
        };
        assert!(invalid_seqs.validate().is_err());
    }

    #[test]
    fn test_serde_roundtrip() {
        let params = EngineParams::vision_model_defaults();
        let serialized = serde_yaml::to_string(&params).unwrap();
        let deserialized: EngineParams = serde_yaml::from_str(&serialized).unwrap();

        assert_eq!(params.get_mem_mb(), deserialized.get_mem_mb());
        assert_eq!(params.temperature, deserialized.temperature);
    }
}
