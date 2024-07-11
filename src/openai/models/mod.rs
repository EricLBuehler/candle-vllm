pub mod gemma;
pub mod llama;
pub mod mistral;
pub mod phi2;
pub mod phi3;
pub mod qwen2;
use candle_core::DType;
use either::Either;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug, Clone)]
pub struct RopeScaling(#[serde(with = "either::serde_untagged")] pub Either<Vec<f64>, String>);

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub max_seq_len: usize,
    pub sliding_window: Option<usize>,
    pub hidden_act: Option<candle_nn::Activation>,
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<HashMap<String, RopeScaling>>,
    pub original_max_position_embeddings: Option<usize>,
    pub attention_bias: bool,
    pub partial_rotary_factor: Option<f32>,
    pub qk_layer_rms_norm: Option<bool>,
    pub kv_cache_dtype: DType,
}

impl Config {
    pub fn get_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
