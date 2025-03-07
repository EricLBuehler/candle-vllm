pub mod deepseek;
pub mod gemma;
pub mod linear;
pub mod llama;
pub mod mistral;
pub mod phi2;
pub mod phi3;
pub mod quantized_llama;
pub mod quantized_phi3;
pub mod quantized_qwen2;
pub mod qwen2;
pub mod stable_lm;
pub mod yi;
use crate::SpecificConfig;
use candle_core::DType;
use either::Either;
use serde::Deserialize;
use std::collections::HashMap;
#[derive(Deserialize, Debug, Clone)]
pub struct RopeScaling(#[serde(with = "either::serde_untagged")] pub Either<Vec<f64>, String>);

#[derive(Deserialize, Debug, Clone)]
pub struct TokenID(
    #[serde(with = "either::serde_untagged")] pub Either<Option<u32>, Option<Vec<u32>>>,
);

#[derive(Deserialize, PartialEq, Debug, Clone)]
pub struct QuantConfig {
    pub quant_method: String,
    pub bits: usize,
    pub group_size: i32,
    pub sym: bool,
    pub desc_act: Option<bool>,
    pub checkpoint_format: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub enum TopkMethod {
    #[serde(rename = "noaux_tc")]
    NoAuxTc,
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Debug)]
pub enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
    #[serde(rename = "sigmoid")]
    Sigmoid,
}

#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub num_experts_per_tok: Option<usize>,
    pub n_routed_experts: usize,
    pub moe_intermediate_size: usize,
    pub scoring_func: ScoringFunc,
    pub topk_method: TopkMethod,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub n_shared_experts: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub q_lora_rank: Option<usize>,
    pub rope_scaling: Option<DeepSeekRopeScaling>,
    pub n_group: usize,
    pub topk_group: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaledRopeType {
    #[serde(alias = "su")]
    #[serde(alias = "longrope")]
    Su,
    #[serde(alias = "yarn")]
    Yarn,
    #[serde(alias = "dynamic")]
    Dynamic,
    #[serde(alias = "linear")]
    Linear,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DeepSeekRopeScaling {
    Yarn {
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all_dim: f32,
        factor: f32,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    LinearOrDynamic {
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        factor: f64,
    },
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub head_dim: Option<usize>,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: TokenID,
    pub eos_token_id: TokenID,
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
    pub use_qkv_bias: Option<bool>,
    pub custom_stop_tokens: Option<Vec<String>>,
    pub specific_config: SpecificConfig,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub quantization_config: Option<QuantConfig>,
    pub moe_config: Option<MoEConfig>,
}

impl Config {
    pub fn get_head_size(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn q_head_dim(&self) -> usize {
        match &self.moe_config {
            Some(cfg) => cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            _ => self.get_head_size(),
        }
    }

    pub fn k_head_dim(&self) -> usize {
        match &self.moe_config {
            Some(cfg) => {
                //q_head_dim
                cfg.qk_rope_head_dim + cfg.qk_nope_head_dim
            }
            _ => self.get_head_size(),
        }
    }

    pub fn v_head_dim(&self) -> usize {
        match &self.moe_config {
            Some(cfg) => {
                //q_head_dim
                cfg.qk_rope_head_dim + cfg.qk_nope_head_dim
            }
            _ => self.get_head_size(),
        }
    }
}
