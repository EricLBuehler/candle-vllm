use super::{
    attention::Attention, mlp::Mlp, rotary_emb::ScalingRotaryEmbedding, Config, ScalingValue,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{embedding, rms_norm, Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::QuantConfig;
use crate::openai::models::TokenID;
use crate::InputMetadata;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, RmsNorm};
use either::Either;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

fn default_rope_theta() -> f64 {
    10000.0
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub(crate) rope_theta: f64,
    pub(crate) rope_type: Option<String>,
    pub(crate) original_max_position_embeddings: Option<usize>,
    pub(crate) factor: Option<f64>,
    pub(crate) beta_fast: Option<f64>,
    pub(crate) beta_slow: Option<f64>,
    pub(crate) mscale: Option<f64>,
    pub(crate) mscale_all_dim: Option<f64>,
    pub(crate) llama_4_scaling_beta: Option<f64>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct MistralTextConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) original_max_position_embeddings: Option<usize>,
    pub(crate) rms_norm_eps: f64,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
    /// Direct rope_theta field (for older config formats)
    #[serde(default)]
    pub(crate) rope_theta: Option<f64>,
    /// Nested rope_parameters (for newer Ministral/Mistral3 config formats)
    pub(crate) rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub(crate) attention_bias: bool,
    pub(crate) hidden_act: Option<Activation>,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) quantization_config: Option<QuantConfig>,
}

impl MistralTextConfig {
    /// Get rope_theta from either direct field or nested rope_parameters
    pub fn get_rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().map(|rp| rp.rope_theta))
            .unwrap_or(default_rope_theta())
    }

    /// Get original_max_position_embeddings from either direct field or nested rope_parameters
    pub fn get_original_max_position_embeddings(&self) -> Option<usize> {
        self.original_max_position_embeddings.or_else(|| {
            self.rope_parameters
                .as_ref()
                .and_then(|rp| rp.original_max_position_embeddings)
        })
    }
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Mistral3Config {
    pub architectures: Option<Vec<String>>,
    pub bos_token_id: Option<TokenID>,
    pub eos_token_id: Option<TokenID>,
    pub text_config: MistralTextConfig,
}

impl Mistral {
    pub fn load_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let mut config = Config::load_config(filename.clone())?;
        config.head_dim = Some(
            config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads),
        );
        config.num_key_value_heads = Some(
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
        );
        config.max_seq_len = config.max_position_embeddings.unwrap_or(config.max_seq_len);
        if config.quantization_config.is_some() {
            config.quant = Some(
                config
                    .quantization_config
                    .as_ref()
                    .unwrap()
                    .quant_method
                    .clone(),
            );
        } else if isq.is_some() {
            config.quant = Some(isq.unwrap().to_string());
        }
        Ok(config)
    }

    pub fn load_text_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let config = match std::fs::read(filename.clone()) {
            Ok(f) => {
                let config: Mistral3Config =
                    serde_json::from_slice(&f).map_err(candle_core::Error::wrap)?;
                config
            }
            Err(e) => panic!("Unable to load config file {:?}", e),
        };

        let bos_token_id = config
            .bos_token_id
            .unwrap_or(super::TokenID(Either::Left(Some(1))));

        let eos_token_id = config
            .eos_token_id
            .unwrap_or(super::TokenID(Either::Left(Some(2))));

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

        // Build rope_scaling from rope_parameters if present and rope_type is yarn
        let rope_scaling = config.text_config.rope_parameters.as_ref().and_then(|rp| {
            if rp.rope_type.as_deref() == Some("yarn") {
                let mut map = HashMap::<String, ScalingValue>::new();
                map.insert(
                    "rope_type".to_string(),
                    ScalingValue::String("yarn".to_string()),
                );
                if let Some(factor) = rp.factor {
                    map.insert("factor".to_string(), ScalingValue::Single(factor));
                }
                if let Some(beta_fast) = rp.beta_fast {
                    map.insert("beta_fast".to_string(), ScalingValue::Single(beta_fast));
                }
                if let Some(beta_slow) = rp.beta_slow {
                    map.insert("beta_slow".to_string(), ScalingValue::Single(beta_slow));
                }
                if let Some(orig_max_pos) = rp.original_max_position_embeddings {
                    map.insert(
                        "original_max_position_embeddings".to_string(),
                        ScalingValue::Single(orig_max_pos as f64),
                    );
                }
                if let Some(mscale) = rp.mscale {
                    map.insert("mscale".to_string(), ScalingValue::Single(mscale));
                }
                if let Some(mscale_all_dim) = rp.mscale_all_dim {
                    map.insert(
                        "mscale_all_dim".to_string(),
                        ScalingValue::Single(mscale_all_dim),
                    );
                }
                // Add attn_factor and extrapolation_factor with default values for yarn
                map.insert("attn_factor".to_string(), ScalingValue::Single(1.0));
                map.insert(
                    "extrapolation_factor".to_string(),
                    ScalingValue::Single(1.0),
                );
                Some(map)
            } else {
                None
            }
        });

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
            rope_theta: config.text_config.get_rope_theta(),
            rope_local_base_freq: None,
            bos_token_id: Some(bos_token_id),
            eos_token_id,
            max_seq_len: config.text_config.max_position_embeddings,
            sliding_window: config.text_config.sliding_window,
            sliding_window_pattern: None,
            hidden_act: Some(config.text_config.hidden_act.unwrap_or(Activation::Silu)),
            hidden_activation: None,
            tie_word_embeddings: config.text_config.tie_word_embeddings,
            rope_scaling,
            max_position_embeddings: Some(config.text_config.max_position_embeddings),
            original_max_position_embeddings: config
                .text_config
                .get_original_max_position_embeddings(),
            attention_bias: Some(config.text_config.attention_bias),
            partial_rotary_factor: None,
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: config.text_config.quantization_config.clone(),
            moe_config: None,
            quant,
            fp8_kvcache: None,
        };
        Ok(config)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            cfg.sliding_window,
        )?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"), comm.clone())?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
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
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

pub struct Mistral {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    cfg: Config,
}

impl Mistral {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = if cfg.architectures.is_some()
            && cfg.architectures.as_ref().unwrap()[0] == "Mistral3ForConditionalGeneration"
        {
            //text model in multimodal weights
            vb.pp("language_model.model")
        } else {
            vb.pp("model")
        };
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, cfg, device, true)?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), comm.clone())?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            // When tie_word_embeddings is true, reuse the embedding weights for lm_head
            ReplicatedLinear::from_weight_bias(embed_tokens.embeddings().clone(), None)?
        } else {
            ReplicatedLinear::load_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                if cfg.architectures.is_some()
                    && cfg.architectures.as_ref().unwrap()[0] == "Mistral3ForConditionalGeneration"
                {
                    vb.pp("language_model.lm_head")
                } else {
                    vb.pp("lm_head")
                },
                &None,
                &None,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
        })
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
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );
        let mut xs = self.embed_tokens.forward(input_ids)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?
            }
        } else {
            for layer in self.layers.iter() {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?
            }
        }
        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
