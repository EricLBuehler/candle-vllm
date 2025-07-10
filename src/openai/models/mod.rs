pub mod deepseek;
pub mod gemma;
pub mod gemma3;
pub mod glm4;
pub mod linear;
pub mod llama;
pub mod mistral;
pub mod phi2;
pub mod phi3;
pub mod quantized_glm4;
pub mod quantized_llama;
pub mod quantized_phi3;
pub mod quantized_qwen;
pub mod qwen;
pub mod stable_lm;
pub mod yi;
use crate::openai::distributed::Comm;
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use crate::SpecificConfig;
use candle_core::{DType, Device, Result, Tensor};
use either::Either;
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
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
    pub sym: Option<bool>,
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
    pub num_experts_offload_per_rank: Option<usize>,
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
    pub rope_local_base_freq: Option<f64>,
    pub bos_token_id: TokenID,
    pub eos_token_id: TokenID,
    pub max_seq_len: usize,
    pub sliding_window: Option<usize>,
    pub sliding_window_pattern: Option<usize>,
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

#[cfg(feature = "flash-attn")]
pub fn get_attention_casual_mask(
    _: &Device,
    _: DType,
    _: usize,
    _: usize,
    _: &[Vec<usize>],
    _: Option<usize>,
) -> Option<Tensor> {
    None
}

#[cfg(not(feature = "flash-attn"))]
pub fn get_attention_casual_mask(
    device: &Device,
    dtype: DType,
    b_size: usize,
    tgt_len: usize,
    positions: &[Vec<usize>],
    sliding_window: Option<usize>,
) -> Option<Tensor> {
    let seqlen_offset = positions[0][0]; //TODO(guoqingbao): position for each request
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device).ok();
    let mask = if seqlen_offset > 0 && mask.is_some() {
        match Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device) {
            Ok(mask0) => Tensor::cat(&[&mask0, &mask.unwrap()], candle_core::D::Minus1).ok(),
            Err(_) => {
                return None;
            }
        }
    } else {
        mask
    };
    match mask {
        Some(m) => m
            .expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))
            .unwrap()
            .to_dtype(dtype)
            .ok(),
        _ => None,
    }
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

pub struct NaiveAttention {
    kv_cache: RefCell<KvCache>,
    num_kv_groups: usize,
    scale: f64,
}

impl NaiveAttention {
    pub fn new(cfg: &Config, sliding_window: Option<usize>) -> Self {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let scale = 1f64 / f64::sqrt(cfg.head_dim.unwrap() as f64);

        let kv_cache = if let Some(sliding_window) = sliding_window {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(2, sliding_window))
        } else {
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(2, cfg.max_seq_len))
        };
        Self {
            kv_cache: RefCell::new(kv_cache),
            num_kv_groups,
            scale,
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        let (b_sz, _, seq_len, _) = q.dims4()?;
        {
            if seq_len > 1 {
                self.clear_kv_cache();
            }
        }
        let mut cache = self.kv_cache.borrow_mut();
        let (k, v) = match &mut *cache {
            KvCache::Normal(c) => c.append(k, v)?,
            KvCache::Rotating(c) => c.append(k, v)?,
        };

        let k = candle_transformers::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        let attn_weights = match softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_out = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_out
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, ()))
    }

    pub fn clear_kv_cache(&self) {
        let mut cache = self.kv_cache.borrow_mut();
        match &mut *cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

pub enum AttentionSelect {
    Naive(NaiveAttention),
    Paged(PagedAttention),
}

impl AttentionSelect {
    pub fn new(
        cfg: &Config,
        sliding_window: Option<usize>,
        comm: Rc<Comm>,
        device: &Device,
        paged: bool,
    ) -> Self {
        if !paged {
            AttentionSelect::Naive(NaiveAttention::new(cfg, sliding_window))
        } else {
            let head_dim = cfg.head_dim.unwrap();
            let attention_heads = cfg.num_attention_heads / comm.world_size();
            let kv_heads = cfg.num_key_value_heads / comm.world_size();
            AttentionSelect::Paged(
                PagedAttention::new(
                    attention_heads,
                    head_dim,
                    1. / ((head_dim as f32).sqrt()),
                    Some(kv_heads),
                    sliding_window,
                    device.clone(),
                    None,
                )
                .unwrap(),
            )
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
        softcapping: Option<f64>,
    ) -> Result<Tensor> {
        match self {
            AttentionSelect::Naive(att) => att.forward(q, k, v, attention_mask, softcapping),
            AttentionSelect::Paged(pag) => {
                let (b_sz, _, seq_len, _) = q.dims4()?;
                pag.forward(
                    q,
                    k,
                    v,
                    attention_mask,
                    cache.map(|(k_, _)| k_.clone()),
                    cache.map(|(_, v_)| v_.clone()),
                    input_metadata,
                    softcapping,
                )?
                .reshape((b_sz, seq_len, ()))
            }
        }
    }
}
