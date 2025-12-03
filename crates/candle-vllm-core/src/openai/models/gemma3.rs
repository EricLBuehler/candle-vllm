use super::{
    attention::Attention, mlp::Mlp, rotary_emb::ScalingRotaryEmbedding, Config, QuantConfig,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{embedding, Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::rotary_emb::DefaultRotaryEmbedding;
use crate::openai::models::ScalingValue;
use crate::openai::models::TokenID;
use crate::InputMetadata;
use candle::{DType, Device, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::{Activation, RmsNorm};
use either::Either;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}
#[derive(serde::Deserialize, Debug, Clone)]
pub(crate) struct Gemma3RopeScaling(
    #[serde(with = "either::serde_untagged")] pub Either<f64, String>,
);

#[derive(serde::Deserialize, Debug, Clone)]
pub struct GemmaTextConfig {
    #[serde(default = "vocab_size")]
    pub(crate) vocab_size: usize,
    #[serde(default = "hidden_size")]
    pub(crate) hidden_size: usize,
    #[serde(default = "intermediate_size")]
    pub(crate) intermediate_size: usize,
    #[serde(default = "num_hidden_layers")]
    pub(crate) num_hidden_layers: usize,
    #[serde(default = "num_attention_heads")]
    pub(crate) num_attention_heads: usize,
    #[serde(default = "num_key_value_heads")]
    pub(crate) num_key_value_heads: usize,
    #[serde(default = "head_dim")]
    pub(crate) head_dim: usize,
    #[serde(default = "hidden_activation")]
    pub(crate) hidden_activation: Activation,
    #[serde(default = "max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    pub(crate) original_max_position_embeddings: Option<usize>,
    #[serde(default = "rms_norm_eps")]
    pub(crate) rms_norm_eps: f64,
    pub(crate) eos_token_id: Option<TokenID>,
    pub(crate) bos_token_id: Option<TokenID>,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    #[serde(default = "rope_theta")]
    pub(crate) rope_theta: f64,
    #[serde(default = "attention_bias")]
    pub(crate) attention_bias: bool,
    // #[serde(default = "query_pre_attn_scalar")]
    // pub(crate) query_pre_attn_scalar: usize,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) final_logit_softcapping: Option<f64>,
    pub(crate) attn_logit_softcapping: Option<f64>,
    #[serde(default = "rope_local_base_freq")]
    pub(crate) rope_local_base_freq: f64,
    #[serde(default = "sliding_window_pattern")]
    pub(crate) sliding_window_pattern: usize,
    pub(crate) quantization_config: Option<QuantConfig>,
    pub(crate) rope_scaling: Option<HashMap<String, Gemma3RopeScaling>>,
}

serde_default!(usize, vocab_size, 262208);
serde_default!(usize, hidden_size, 2304);
serde_default!(usize, intermediate_size, 9216);
serde_default!(usize, num_hidden_layers, 26);
serde_default!(usize, num_attention_heads, 8);
serde_default!(usize, num_key_value_heads, 4);
serde_default!(usize, head_dim, 256);
serde_default!(usize, max_position_embeddings, 131072);
serde_default!(f64, rms_norm_eps, 1e-6);
serde_default!(bool, tie_word_embeddings, true);
serde_default!(f64, rope_theta, 1_000_000.0);
serde_default!(bool, attention_bias, false);
// serde_default!(usize, query_pre_attn_scalar, 256);
serde_default!(f64, rope_local_base_freq, 10_000.0);
serde_default!(usize, sliding_window_pattern, 6);
serde_default!(Activation, hidden_activation, Activation::Silu);

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Gemma3Config {
    pub architectures: Option<Vec<String>>,
    pub bos_token_id: Option<TokenID>,
    pub eos_token_id: Option<TokenID>,
    pub text_config: GemmaTextConfig,
}

impl Gemma3 {
    pub fn load_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let config = match std::fs::read(filename.clone()) {
            Ok(f) => {
                let config: Gemma3Config =
                    serde_json::from_slice(&f).map_err(candle_core::Error::wrap)?;
                config
            }
            Err(e) => panic!("Unable to load config file {:?}", e),
        };

        let bos_token_id = config
            .text_config
            .bos_token_id
            .or(config.bos_token_id)
            .unwrap_or(super::TokenID(Either::Left(Some(2))));

        let eos_token_id = config
            .text_config
            .eos_token_id
            .or(config.eos_token_id)
            .unwrap_or(super::TokenID(Either::Right(Some(vec![1, 106]))));

        let ropescaling = if config.text_config.rope_scaling.is_some() {
            //convert gemma3 rope scaling into standard scaling
            let mut ropescaling = HashMap::<String, ScalingValue>::new();
            for (key, value) in config.text_config.rope_scaling.as_ref().unwrap() {
                match value {
                    Gemma3RopeScaling(Either::Left(l)) => {
                        ropescaling.insert(key.to_string(), ScalingValue::Single(*l));
                    }
                    Gemma3RopeScaling(Either::Right(r)) => {
                        ropescaling.insert(key.to_string(), ScalingValue::String(r.to_string()));
                    }
                }
            }
            Some(ropescaling)
        } else {
            None
        };

        let quant = if config.text_config.quantization_config.is_some() {
            Some(
                config
                    .text_config
                    .quantization_config
                    .as_ref()
                    .unwrap()
                    .quant_method
                    .clone(),
            )
        } else if isq.is_some() {
            Some(isq.unwrap().to_string())
        } else {
            None
        };

        let config = Config {
            architectures: config.architectures,
            hidden_size: config.text_config.hidden_size,
            head_dim: Some(config.text_config.head_dim),
            intermediate_size: config.text_config.intermediate_size,
            vocab_size: config.text_config.vocab_size,
            num_hidden_layers: config.text_config.num_hidden_layers,
            num_attention_heads: config.text_config.num_attention_heads,
            num_key_value_heads: Some(config.text_config.num_key_value_heads),
            rms_norm_eps: config.text_config.rms_norm_eps,
            rope_theta: config.text_config.rope_theta,
            rope_local_base_freq: Some(config.text_config.rope_local_base_freq),
            bos_token_id: Some(bos_token_id),
            eos_token_id,
            max_seq_len: config.text_config.max_position_embeddings,
            sliding_window: config.text_config.sliding_window,
            sliding_window_pattern: Some(config.text_config.sliding_window_pattern),
            hidden_act: Some(config.text_config.hidden_activation),
            hidden_activation: None,
            tie_word_embeddings: config.text_config.tie_word_embeddings,
            rope_scaling: ropescaling,
            max_position_embeddings: Some(config.text_config.max_position_embeddings),
            original_max_position_embeddings: config.text_config.original_max_position_embeddings,
            attention_bias: Some(config.text_config.attention_bias),
            partial_rotary_factor: None,
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: config.text_config.attn_logit_softcapping,
            final_logit_softcapping: config.text_config.final_logit_softcapping,
            quantization_config: config.text_config.quantization_config.clone(),
            moe_config: None,
            quant,
            fp8_kvcache: None,
        };
        Ok(config)
    }
}

fn rms_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(dim, "weight")?;
    Ok(RmsNorm::new((weight + 1.0f64)?, eps))
}

impl ScalingRotaryEmbedding {
    pub fn new_sliding(
        dtype: DType,
        sliding_window: Option<usize>,
        cfg: &Config,
        dev: &Device,
    ) -> Result<ScalingRotaryEmbedding> {
        let rope_freq = sliding_window
            .and(cfg.rope_local_base_freq)
            .unwrap_or(cfg.rope_theta);

        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let max_seq_len = cfg.max_seq_len;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let factor = 1.0f64;
        let t_len = (max_seq_len as f64 * factor) as u32;
        let t = Tensor::arange(0u32, t_len, dev)?
            .to_dtype(DType::F32)?
            .reshape((t_len as usize, 1))?;
        let t = (t / factor)?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        Ok(ScalingRotaryEmbedding(DefaultRotaryEmbedding {
            cos,
            sin,
            is_gpt_neox: true,
            rotary_dim: None,
        }))
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    sliding_window: Option<usize>,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb.clone(),
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            sliding_window,
        )?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), comm.clone())?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;

        let pre_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;

        let post_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            sliding_window,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

pub struct Gemma3 {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    cfg: Config,
}

impl Gemma3 {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("language_model.model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new_sliding(
            DType::F32,
            None,
            cfg,
            device,
        )?);
        let sliding_emb = Arc::new(ScalingRotaryEmbedding::new_sliding(
            DType::F32,
            cfg.sliding_window,
            cfg,
            device,
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let is_sliding_window =
                if cfg.sliding_window.is_some() && cfg.sliding_window_pattern.is_some() {
                    (layer_idx + 1) % cfg.sliding_window_pattern.unwrap() > 0
                } else {
                    false
                };
            let layer = DecoderLayer::new(
                cfg,
                vb_l.pp(layer_idx),
                comm.clone(),
                if is_sliding_window {
                    rotary_emb.clone()
                } else {
                    sliding_emb.clone()
                },
                is_sliding_window.then_some(cfg.sliding_window.unwrap()),
            )?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = ReplicatedLinear::from_weight_bias(embed_tokens.embeddings().clone(), None)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            hidden_size: cfg.hidden_size,
            cfg: cfg.clone(),
        })
    }

    fn create_attention_masks(
        &self,
        seqlens: &Vec<u32>,
        input_positions: &Tensor,
        is_prefill: bool,
    ) -> Result<(Option<Vec<Tensor>>, Option<Vec<Tensor>>)> {
        //normal mask
        let mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            seqlens,
            None,
            is_prefill,
        );

        //sliding_mask
        let sliding_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            seqlens,
            self.cfg.sliding_window,
            is_prefill,
        );

        Ok((mask, sliding_mask))
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let seqlens = if input_metadata.cu_seqlens_q.is_some() {
            input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..]
                .into()
        } else {
            Vec::new()
        };
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(&seqlens, input_positions, input_metadata.is_prefill)?;

        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let mask = if layer.sliding_window.is_some() {
                    &sliding_attention_mask
                } else {
                    &attention_mask
                };

                xs = layer.forward(
                    &xs,
                    mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?
            }
        } else {
            for layer in self.layers.iter() {
                let mask = if layer.sliding_window.is_some() {
                    &sliding_attention_mask
                } else {
                    &attention_mask
                };
                xs = layer.forward(&xs, mask.as_ref(), input_positions, None, input_metadata)?
            }
        }

        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs)?.to_dtype(DType::F32)?;

        match self.cfg.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => (logits / sc)?.tanh()? * sc,
        }
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
