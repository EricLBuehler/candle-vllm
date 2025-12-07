//! Mock model configurations and loaders for testing.

use candle_vllm_core::openai::models::Config;
use serde_json::json;

/// Create a minimal Llama config for CPU testing.
pub fn create_tiny_llama_config() -> Config {
    let config_json = json!({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 1000,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "model_type": "llama"
    });
    
    serde_json::from_value(config_json).expect("Failed to create tiny Llama config")
}

/// Create a minimal Mistral config for CPU testing.
pub fn create_tiny_mistral_config() -> Config {
    let config_json = json!({
        "architectures": ["MistralForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "model_type": "mistral"
    });
    
    serde_json::from_value(config_json).expect("Failed to create tiny Mistral config")
}

/// Mock model loader for testing without actual weights.
pub struct MockModelLoader;

impl MockModelLoader {
    pub fn new() -> Self {
        Self
    }
    
    /// Create a mock model path that doesn't require actual files.
    pub fn mock_model_path() -> String {
        "/tmp/mock_model".to_string()
    }
}

impl Default for MockModelLoader {
    fn default() -> Self {
        Self::new()
    }
}
