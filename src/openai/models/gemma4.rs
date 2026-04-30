use super::{
    attention::Attention, mlp::Mlp, rotary_emb::ScalingRotaryEmbedding, Config, MoEConfig,
    QuantConfig, QwenMoEConfig,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{embedding, Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::layers::moe::{
    FusedMoe, FusedMoeFp8, FusedMoeISQ, FusedMoeMxfp4, FusedMoeNvfp4,
};
use crate::openai::models::layers::others::{rms_norm, NormX};
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::rotary_emb::DefaultRotaryEmbedding;
use crate::openai::models::ScalingValue;
use crate::openai::models::TokenID;
use crate::InputMetadata;
use candle::{DType, Device, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::{Activation, Linear};
use either::Either;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

#[doc(hidden)]
#[macro_export]
macro_rules! gemma4_serde_default {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

#[derive(serde::Deserialize, Debug, Clone)]
pub(crate) struct Gemma4RopeScaling(
    #[serde(with = "either::serde_untagged")] pub Either<f64, String>,
);

#[allow(dead_code)]
#[derive(serde::Deserialize, Debug, Clone)]
pub struct Gemma4TextConfig {
    #[serde(default = "g4_vocab_size")]
    pub(crate) vocab_size: usize,
    #[serde(default = "g4_hidden_size")]
    pub(crate) hidden_size: usize,
    #[serde(default = "g4_intermediate_size")]
    pub(crate) intermediate_size: usize,
    #[serde(default = "g4_num_hidden_layers")]
    pub(crate) num_hidden_layers: usize,
    #[serde(default = "g4_num_attention_heads")]
    pub(crate) num_attention_heads: usize,
    #[serde(default = "g4_num_key_value_heads")]
    pub(crate) num_key_value_heads: usize,
    pub(crate) num_global_key_value_heads: Option<usize>,
    #[serde(default = "g4_head_dim")]
    pub(crate) head_dim: usize,
    pub(crate) global_head_dim: Option<usize>,
    pub(crate) swa_head_dim: Option<usize>,
    #[serde(default = "g4_hidden_activation")]
    pub(crate) hidden_activation: Activation,
    #[serde(default = "g4_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    pub(crate) original_max_position_embeddings: Option<usize>,
    #[serde(default = "g4_rms_norm_eps")]
    pub(crate) rms_norm_eps: f64,
    pub(crate) eos_token_id: Option<TokenID>,
    pub(crate) bos_token_id: Option<TokenID>,
    #[serde(default = "g4_tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    #[serde(default = "g4_rope_theta")]
    pub(crate) rope_theta: f64,
    #[serde(default = "g4_attention_bias")]
    pub(crate) attention_bias: bool,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) final_logit_softcapping: Option<f64>,
    pub(crate) attn_logit_softcapping: Option<f64>,
    #[serde(default = "g4_rope_local_base_freq")]
    pub(crate) rope_local_base_freq: f64,
    pub(crate) quantization_config: Option<QuantConfig>,
    pub(crate) rope_scaling: Option<HashMap<String, Gemma4RopeScaling>>,
    pub(crate) rope_parameters: Option<serde_json::Value>,
    pub(crate) layer_types: Option<Vec<String>>,
    pub(crate) attention_k_eq_v: Option<bool>,
    pub(crate) partial_rotary_factor: Option<f64>,
    pub(crate) enable_moe_block: Option<bool>,
    #[serde(alias = "n_routed_experts")]
    pub(crate) num_experts: Option<usize>,
    pub(crate) num_experts_per_tok: Option<usize>,
    pub(crate) top_k_experts: Option<usize>,
    pub(crate) moe_intermediate_size: Option<usize>,
    pub(crate) hidden_size_per_layer_input: Option<usize>,
    pub(crate) num_kv_shared_layers: Option<usize>,
    pub(crate) use_double_wide_mlp: Option<bool>,
}

gemma4_serde_default!(usize, g4_vocab_size, 262208);
gemma4_serde_default!(usize, g4_hidden_size, 3584);
gemma4_serde_default!(usize, g4_intermediate_size, 14336);
gemma4_serde_default!(usize, g4_num_hidden_layers, 36);
gemma4_serde_default!(usize, g4_num_attention_heads, 8);
gemma4_serde_default!(usize, g4_num_key_value_heads, 4);
gemma4_serde_default!(usize, g4_head_dim, 256);
gemma4_serde_default!(usize, g4_max_position_embeddings, 131072);
gemma4_serde_default!(f64, g4_rms_norm_eps, 1e-6);
gemma4_serde_default!(bool, g4_tie_word_embeddings, true);
gemma4_serde_default!(f64, g4_rope_theta, 1_000_000.0);
gemma4_serde_default!(bool, g4_attention_bias, false);
gemma4_serde_default!(f64, g4_rope_local_base_freq, 10_000.0);
gemma4_serde_default!(Activation, g4_hidden_activation, Activation::Silu);

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Gemma4Config {
    pub architectures: Option<Vec<String>>,
    pub bos_token_id: Option<TokenID>,
    pub eos_token_id: Option<TokenID>,
    pub text_config: Gemma4TextConfig,
}

struct Gemma4Router {
    scale: Tensor,
    proj: Linear,
    per_expert_scale: Tensor,
    hidden_size: usize,
    top_k: usize,
    eps: f64,
}

impl Gemma4Router {
    fn new(
        hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        eps: f64,
        vb: VarBuilder,
        dtype: DType,
    ) -> Result<Self> {
        let scale = vb.get(hidden_size, "scale")?.to_dtype(dtype)?;
        let proj_vb = vb.pp("proj");
        let proj_w = proj_vb
            .get((num_experts, hidden_size), "weight")?
            .to_dtype(dtype)?;
        let proj = Linear::new(proj_w, None);

        let per_expert_scale = vb
            .get(num_experts, "per_expert_scale")?
            .to_dtype(DType::F32)?;

        Ok(Self {
            scale,
            proj,
            per_expert_scale,
            hidden_size,
            top_k,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let rms = (xs_f32.sqr()?.mean_keepdim(candle::D::Minus1)? + self.eps)?.sqrt()?;
        let normed = xs_f32.broadcast_div(&rms)?;

        let root_size = (self.hidden_size as f64).powf(-0.5);
        let scaled = (normed * root_size)?;
        let scale_f32 = self.scale.to_dtype(DType::F32)?;
        let scaled = scaled.broadcast_mul(&scale_f32.unsqueeze(0)?)?;

        let logits = scaled
            .to_dtype(self.proj.weight().dtype())?
            .apply(&self.proj)?;
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?;

        let sorted_idx = probs.arg_sort_last_dim(false)?;
        let topk_indices = sorted_idx.narrow(1, 0, self.top_k)?.contiguous()?;
        let topk_weights = probs.contiguous()?.gather(&topk_indices, 1)?;

        let renorm = topk_weights.sum_keepdim(candle::D::Minus1)?;
        let topk_weights = topk_weights.broadcast_div(&renorm)?;

        let flat_idx = topk_indices.flatten_all()?.to_dtype(DType::U32)?;
        let scales = self
            .per_expert_scale
            .index_select(&flat_idx, 0)?
            .reshape(topk_indices.shape())?;
        let topk_weights = (topk_weights * scales)?;

        let topk_indices = topk_indices.to_dtype(DType::U32)?;
        Ok((topk_weights, topk_indices))
    }
}

enum Gemma4MoE {
    FusedMoe(FusedMoe),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
    FusedMoeMxfp4(FusedMoeMxfp4),
    FusedMoeNvfp4(FusedMoeNvfp4),
}

impl Gemma4MoE {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeFp8(m) => m.forward(xs, is_prefill),
            Self::FusedMoeMxfp4(m) => m.forward(xs, is_prefill),
            Self::FusedMoeNvfp4(m) => m.forward(xs, is_prefill),
        }
    }

    fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward_with_routing(xs, topk_weights, topk_ids, is_prefill),
            Self::FusedMoeISQ(m) => m.forward_with_routing(xs, topk_weights, topk_ids, is_prefill),
            Self::FusedMoeFp8(m) => m.forward_with_routing(xs, topk_weights, topk_ids, is_prefill),
            Self::FusedMoeMxfp4(m) => {
                m.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
            }
            Self::FusedMoeNvfp4(m) => {
                m.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
            }
        }
    }
}

struct Gemma4DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    moe: Option<Gemma4MoE>,
    gemma4_router: Option<Gemma4Router>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    pre_feedforward_layernorm: NormX,
    post_feedforward_layernorm: NormX,
    post_feedforward_layernorm_1: Option<NormX>,
    post_feedforward_layernorm_2: Option<NormX>,
    pre_feedforward_layernorm_2: Option<NormX>,
    post_per_layer_input_norm: Option<NormX>,
    per_layer_input_gate: Option<ReplicatedLinear>,
    per_layer_projection: Option<ReplicatedLinear>,
    layer_scalar: Tensor,
    is_sliding: bool,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    rotary_emb_local: Arc<ScalingRotaryEmbedding>,
}

impl Gemma4DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        moe_cfg: Option<&QwenMoEConfig>,
        vb: VarBuilder,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        rotary_emb_local: Arc<ScalingRotaryEmbedding>,
        is_sliding: bool,
        enable_moe: bool,
        global_head_dim: usize,
        swa_head_dim: usize,
        global_kv_heads: usize,
        dtype: DType,
        intermediate_size: usize,
        k_eq_v: bool,
        hidden_size_per_layer_input: Option<usize>,
    ) -> Result<Self> {
        let head_dim = if is_sliding {
            swa_head_dim
        } else {
            global_head_dim
        };
        let mut layer_cfg = cfg.clone();
        layer_cfg.head_dim = Some(head_dim);
        if !is_sliding {
            layer_cfg.num_key_value_heads = Some(global_kv_heads);
        }
        layer_cfg.intermediate_size = intermediate_size;

        let sliding_window = if is_sliding { cfg.sliding_window } else { None };

        let self_attn = Attention::new_with_option(
            rotary_emb.clone(),
            &layer_cfg,
            vb.pp("self_attn"),
            comm.clone(),
            sliding_window,
            k_eq_v,
            false,
            Some(1.0),
        )?;

        let mlp = Mlp::new(&layer_cfg, vb.pp("mlp"), comm.clone())?;

        let (moe, gemma4_router) = if enable_moe && moe_cfg.is_some() {
            let mc = moe_cfg.unwrap();
            let num_experts = mc.num_experts.unwrap();
            let top_k = mc.num_experts_per_tok;

            let m = if let Some(quant_cfg) = &cfg.quantization_config {
                if quant_cfg.quant_method == "nvfp4" {
                    Gemma4MoE::FusedMoeNvfp4(FusedMoeNvfp4::new_with_gate(
                        cfg,
                        vb.pp("router").pp("proj"),
                        vb.pp("experts"),
                        comm.clone(),
                        dtype,
                    )?)
                } else if quant_cfg.quant_method == "fp8" {
                    Gemma4MoE::FusedMoeFp8(FusedMoeFp8::new_with_gate(
                        cfg,
                        vb.pp("router").pp("proj"),
                        vb.pp("experts"),
                        comm.clone(),
                        dtype,
                        quant_cfg,
                    )?)
                } else if quant_cfg.quant_method == "mxfp4" {
                    Gemma4MoE::FusedMoeMxfp4(FusedMoeMxfp4::new_with_gate(
                        cfg,
                        vb.pp("router").pp("proj"),
                        vb.pp("experts"),
                        comm.clone(),
                        dtype,
                    )?)
                } else {
                    candle::bail!(
                        "Unsupported quantization for Gemma4 MoE: {}",
                        quant_cfg.quant_method
                    );
                }
            } else if cfg.isq_quant.is_some() {
                Gemma4MoE::FusedMoeISQ(FusedMoeISQ::new_with_gate(
                    cfg,
                    vb.pp("router").pp("proj"),
                    vb.pp("experts"),
                    comm.clone(),
                    dtype,
                )?)
            } else {
                Gemma4MoE::FusedMoe(FusedMoe::new_with_gate(
                    cfg,
                    vb.pp("router").pp("proj"),
                    vb.pp("experts"),
                    comm.clone(),
                    dtype,
                )?)
            };

            let router = Gemma4Router::new(
                cfg.hidden_size,
                num_experts,
                top_k,
                cfg.rms_norm_eps,
                vb.pp("router"),
                dtype,
            )?;

            (Some(m), Some(router))
        } else {
            (None, None)
        };

        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
            DType::F32,
            false,
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            DType::F32,
            false,
        )?;
        let pre_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
            DType::F32,
            false,
        )?;
        let post_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
            DType::F32,
            false,
        )?;

        let (
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
        ) = if enable_moe && moe_cfg.is_some() {
            (
                Some(rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_1"),
                    DType::F32,
                    false,
                )?),
                Some(rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_2"),
                    DType::F32,
                    false,
                )?),
                Some(rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("pre_feedforward_layernorm_2"),
                    DType::F32,
                    false,
                )?),
            )
        } else {
            (None, None, None)
        };

        let (post_per_layer_input_norm, per_layer_input_gate, per_layer_projection) =
            if let Some(pli_dim) = hidden_size_per_layer_input {
                let norm = rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_per_layer_input_norm"),
                    DType::F32,
                    false,
                )?;
                let gate = ReplicatedLinear::load_no_bias(
                    cfg.hidden_size,
                    pli_dim,
                    vb.pp("per_layer_input_gate"),
                    &cfg.isq_quant,
                    &cfg.quantization_config,
                )?;
                let proj = ReplicatedLinear::load_no_bias(
                    pli_dim,
                    cfg.hidden_size,
                    vb.pp("per_layer_projection"),
                    &cfg.isq_quant,
                    &cfg.quantization_config,
                )?;
                (Some(norm), Some(gate), Some(proj))
            } else {
                (None, None, None)
            };

        let layer_scalar = vb.get(1, "layer_scalar")?.to_dtype(dtype)?;

        Ok(Self {
            self_attn,
            mlp,
            moe,
            gemma4_router,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
            post_per_layer_input_norm,
            per_layer_input_gate,
            per_layer_projection,
            layer_scalar,
            is_sliding,
            rotary_emb,
            rotary_emb_local,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        sliding_mask: Option<&Vec<Tensor>>,
        full_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let normed = self.input_layernorm.forward(xs)?;

        let mask = if self.is_sliding {
            sliding_mask
        } else {
            full_mask
        };

        let rotary = if self.is_sliding {
            &self.rotary_emb_local
        } else {
            &self.rotary_emb
        };

        let attn_output = self.self_attn.forward_ext(
            &normed,
            Some(rotary.as_ref()),
            mask,
            input_positions,
            cache,
            input_metadata,
            None,
        )?;

        let mut xs = self.post_attention_layernorm.forward(&attn_output)?;
        xs = (xs + &residual)?;

        let residual = xs.clone();

        let mlp_input = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&mlp_input)?;

        let combined = if let Some(moe) = &self.moe {
            let mlp_normed = self
                .post_feedforward_layernorm_1
                .as_ref()
                .unwrap()
                .forward(&mlp_output)?;

            let residual_flat = residual.flatten(0, residual.rank() - 2)?;

            let moe_output = if let Some(router) = &self.gemma4_router {
                let (topk_weights, topk_ids) = router.forward(&residual_flat)?;
                let moe_input = self
                    .pre_feedforward_layernorm_2
                    .as_ref()
                    .unwrap()
                    .forward(&residual_flat)?;
                moe.forward_with_routing(
                    &moe_input,
                    topk_weights,
                    topk_ids,
                    input_metadata.is_prefill,
                )?
            } else {
                let moe_input = self
                    .pre_feedforward_layernorm_2
                    .as_ref()
                    .unwrap()
                    .forward(&residual_flat)?;
                moe.forward(&moe_input, input_metadata.is_prefill)?
            };
            let moe_output = moe_output.reshape(residual.shape())?;

            let moe_normed = self
                .post_feedforward_layernorm_2
                .as_ref()
                .unwrap()
                .forward(&moe_output)?;

            (mlp_normed + moe_normed)?
        } else {
            mlp_output
        };

        let combined = self.post_feedforward_layernorm.forward(&combined)?;
        let mut xs = (&residual + combined)?;

        if let (Some(gate), Some(proj), Some(norm), Some(pli)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual_ple = xs.clone();
            let gated = gate
                .forward(&xs)?
                .apply(&candle_nn::Activation::GeluPytorchTanh)?;
            let gated = (gated * pli)?;
            let projected = proj.forward(&gated)?;
            xs = (&residual_ple + norm.forward(&projected)?)?;
        }

        xs.broadcast_mul(&self.layer_scalar.to_dtype(xs.dtype())?)
    }
}

pub struct Gemma4 {
    embed_tokens: candle_nn::Embedding,
    embed_tokens_per_layer: Option<candle_nn::Embedding>,
    per_layer_model_projection: Option<Linear>,
    per_layer_projection_norm: Option<NormX>,
    layers: Vec<Gemma4DecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    cfg: Config,
    #[allow(dead_code)]
    layer_types: Vec<String>,
    hidden_size_per_layer_input: Option<usize>,
    num_hidden_layers: usize,
}

impl Gemma4 {
    pub fn load_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let (config, raw) = match std::fs::read(filename.clone()) {
            Ok(f) => {
                let config: Gemma4Config =
                    serde_json::from_slice(&f).map_err(candle_core::Error::wrap)?;
                let raw = String::from_utf8(f).map_err(candle_core::Error::wrap)?;
                (config, raw)
            }
            Err(e) => panic!("Unable to load config file {:?}", e),
        };

        let bos_token_id = config
            .text_config
            .bos_token_id
            .or(config.bos_token_id)
            .unwrap_or(super::TokenID(Either::Left(Some(2))));

        let eos_token_id = config
            .eos_token_id
            .or(config.text_config.eos_token_id)
            .unwrap_or(super::TokenID(Either::Right(Some(vec![1, 106]))));

        let ropescaling = if config.text_config.rope_scaling.is_some() {
            let mut ropescaling = HashMap::<String, ScalingValue>::new();
            for (key, value) in config.text_config.rope_scaling.as_ref().unwrap() {
                match value {
                    Gemma4RopeScaling(Either::Left(l)) => {
                        ropescaling.insert(key.to_string(), ScalingValue::Single(*l));
                    }
                    Gemma4RopeScaling(Either::Right(r)) => {
                        ropescaling.insert(key.to_string(), ScalingValue::String(r.to_string()));
                    }
                }
            }
            Some(ropescaling)
        } else {
            None
        };

        let quant = if config.text_config.quantization_config.is_some() {
            None
        } else {
            isq
        };

        let top_level_quant_config = serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .and_then(|root| root.get("quantization_config").cloned())
            .and_then(|v| serde_json::from_value::<QuantConfig>(v).ok())
            .map(|mut qcfg| {
                qcfg.normalize_compressed_tensors();
                qcfg
            });

        let moe_config = if config.text_config.enable_moe_block.unwrap_or(false)
            || config.text_config.num_experts.is_some()
        {
            Some(MoEConfig::QwenMoE(QwenMoEConfig {
                moe_intermediate_size: config
                    .text_config
                    .moe_intermediate_size
                    .unwrap_or(config.text_config.intermediate_size),
                shared_expert_intermediate_size: None,
                num_experts: config.text_config.num_experts,
                mlp_only_layers: None,
                decoder_sparse_step: None,
                norm_topk_prob: true,
                num_experts_per_tok: config
                    .text_config
                    .num_experts_per_tok
                    .or(config.text_config.top_k_experts)
                    .unwrap_or(8),
                routed_scaling_factor: None,
                first_k_dense_replace: None,
                n_shared_experts: None,
            }))
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
            eos_token_id: Some(eos_token_id),
            max_seq_len: config.text_config.max_position_embeddings,
            sliding_window: config.text_config.sliding_window,
            sliding_window_pattern: None,
            hidden_act: Some(config.text_config.hidden_activation),
            hidden_activation: None,
            tie_word_embeddings: config.text_config.tie_word_embeddings,
            rope_scaling: ropescaling,
            max_position_embeddings: Some(config.text_config.max_position_embeddings),
            original_max_position_embeddings: config.text_config.original_max_position_embeddings,
            attention_bias: Some(config.text_config.attention_bias),
            partial_rotary_factor: config.text_config.partial_rotary_factor.map(|f| f as f32),
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: config.text_config.attn_logit_softcapping,
            final_logit_softcapping: config.text_config.final_logit_softcapping,
            quantization_config: config
                .text_config
                .quantization_config
                .clone()
                .or(top_level_quant_config),
            moe_config,
            isq_quant: quant,
            fp8_kvcache: None,
            extra_config_json: Some(raw),
        };
        Ok(config)
    }

    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let extra_json = cfg.extra_config_json.as_ref();
        let parsed_extra: Option<serde_json::Value> =
            extra_json.and_then(|s| serde_json::from_str(s).ok());
        let text_cfg = parsed_extra
            .as_ref()
            .and_then(|v| v.get("text_config"))
            .unwrap_or_else(|| parsed_extra.as_ref().unwrap_or(&serde_json::Value::Null));

        let layer_types: Vec<String> = text_cfg
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or("sliding_attention").to_string())
                    .collect()
            })
            .unwrap_or_else(|| {
                (0..cfg.num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % 6 == 0 {
                            "full_attention".to_string()
                        } else {
                            "sliding_attention".to_string()
                        }
                    })
                    .collect()
            });

        let enable_moe = text_cfg
            .get("enable_moe_block")
            .and_then(|v| v.as_bool())
            .unwrap_or(cfg.moe_config.is_some());

        let global_head_dim = text_cfg
            .get("global_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| cfg.head_dim.unwrap_or(256));

        let swa_head_dim = text_cfg
            .get("swa_head_dim")
            .or_else(|| text_cfg.get("head_dim"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(256);

        let global_kv_heads = text_cfg
            .get("num_global_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| cfg.num_key_value_heads.unwrap_or(4));

        let rope_local_base_freq = text_cfg
            .get("rope_local_base_freq")
            .or_else(|| {
                text_cfg
                    .get("rope_parameters")
                    .and_then(|rp| rp.get("sliding_attention"))
                    .and_then(|sa| sa.get("rope_theta"))
            })
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);

        let k_eq_v = text_cfg
            .get("attention_k_eq_v")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let num_kv_shared_layers = text_cfg
            .get("num_kv_shared_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let use_double_wide_mlp = text_cfg
            .get("use_double_wide_mlp")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let hidden_size_per_layer_input: Option<usize> = text_cfg
            .get("hidden_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .filter(|&v| v > 0)
            .map(|v| v as usize);

        let first_kv_shared_layer = cfg.num_hidden_layers.saturating_sub(num_kv_shared_layers);

        let moe_cfg = cfg.moe_config.as_ref().and_then(|mc| match mc {
            MoEConfig::QwenMoE(c) => Some(c),
            _ => None,
        });

        let vb_m = vb.pp("model.language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if let Some(pli_dim) = hidden_size_per_layer_input {
                let total_dim = pli_dim * cfg.num_hidden_layers;
                let emb = embedding(cfg.vocab_size, total_dim, vb_m.pp("embed_tokens_per_layer"))?;

                let proj_vb = vb_m.pp("per_layer_model_projection");
                let proj_w = proj_vb.get((total_dim, cfg.hidden_size), "weight")?;
                let proj_w = if let Ok(scale) = proj_vb
                    .get((total_dim, 1), "weight_scale")
                    .or_else(|_| proj_vb.get((total_dim, 1), "weight_scale_inv"))
                {
                    let scale = scale.to_dtype(DType::F32)?;
                    proj_w
                        .to_dtype(DType::F32)?
                        .broadcast_mul(&scale)?
                        .to_dtype(dtype)?
                } else {
                    proj_w.to_dtype(dtype)?
                };
                let proj = Linear::new(proj_w, None);

                let norm = rms_norm(
                    pli_dim,
                    cfg.rms_norm_eps,
                    vb_m.pp("per_layer_projection_norm"),
                    DType::F32,
                    false,
                )?;

                (Some(emb), Some(proj), Some(norm))
            } else {
                (None, None, None)
            };

        let (global_rope_theta, partial_rotary_factor) = {
            let mut theta = cfg.rope_theta;
            let mut prf = cfg.partial_rotary_factor.unwrap_or(0.25) as f64;
            if let Some(extra) = &cfg.extra_config_json {
                let v: serde_json::Value =
                    serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
                let fa = v
                    .get("text_config")
                    .and_then(|tc| tc.get("rope_parameters"))
                    .and_then(|rp| rp.get("full_attention"));
                if let Some(fa) = fa {
                    if let Some(t) = fa.get("rope_theta").and_then(|v| v.as_f64()) {
                        theta = t;
                    }
                    if let Some(p) = fa.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                        prf = p;
                    }
                }
            }
            (theta, prf)
        };
        let rope_angles = (partial_rotary_factor * global_head_dim as f64 / 2.0) as usize;
        let half_dim = global_head_dim / 2;

        let mut inv_freq_vec: Vec<f32> = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(
                1.0f32 / (global_rope_theta as f32).powf((2 * i) as f32 / global_head_dim as f32),
            );
        }
        for _ in rope_angles..half_dim {
            inv_freq_vec.push(0.0f32);
        }

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), &vb.device())?;
        let t = Tensor::arange(
            0u32,
            cfg.max_position_embeddings.unwrap() as u32,
            &vb.device(),
        )?
        .to_dtype(DType::F32)?
        .reshape((cfg.max_position_embeddings.unwrap(), 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding {
            0: DefaultRotaryEmbedding {
                cos: freqs.cos()?.to_dtype(DType::F32)?,
                sin: freqs.sin()?.to_dtype(DType::F32)?,
                is_gpt_neox: true,
                rotary_dim: None,
            },
        });

        let swa_head_dim_for_rope = if let Some(extra) = &cfg.extra_config_json {
            let v: serde_json::Value =
                serde_json::from_str(extra).unwrap_or(serde_json::Value::Null);
            v.get("swa_head_dim")
                .or_else(|| v.get("text_config").and_then(|tc| tc.get("head_dim")))
                .and_then(|v| v.as_u64())
                .unwrap_or(256) as usize
        } else {
            256
        };

        let mut local_config = cfg.clone();
        local_config.head_dim = Some(swa_head_dim_for_rope);
        local_config.partial_rotary_factor = None;
        local_config.rope_theta = rope_local_base_freq;

        let rotary_emb_local = Arc::new(ScalingRotaryEmbedding {
            0: DefaultRotaryEmbedding::new(DType::F32, &local_config, &vb.device(), true)?,
        });

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();

        for layer_idx in 0..cfg.num_hidden_layers {
            let is_sliding = layer_types
                .get(layer_idx)
                .map(|t| t == "sliding_attention")
                .unwrap_or(true);

            let layer_intermediate = if use_double_wide_mlp
                && num_kv_shared_layers > 0
                && layer_idx >= first_kv_shared_layer
            {
                cfg.intermediate_size * 2
            } else {
                cfg.intermediate_size
            };

            let layer_k_eq_v = k_eq_v && !is_sliding;

            let layer = Gemma4DecoderLayer::new(
                cfg,
                moe_cfg,
                vb_l.pp(layer_idx),
                comm.clone(),
                rotary_emb.clone(),
                rotary_emb_local.clone(),
                is_sliding,
                enable_moe,
                global_head_dim,
                swa_head_dim,
                global_kv_heads,
                dtype,
                layer_intermediate,
                layer_k_eq_v,
                hidden_size_per_layer_input,
            )?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
        }

        let norm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_m.pp("norm"),
            DType::F32,
            false,
        )?;
        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            if cfg.tie_word_embeddings {
                vb_m.pp("embed_tokens")
            } else {
                vb.pp("lm_head")
            },
            &None,
            &None,
        )?;

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            hidden_size: cfg.hidden_size,
            cfg: cfg.clone(),
            layer_types,
            hidden_size_per_layer_input,
            num_hidden_layers: cfg.num_hidden_layers,
        })
    }

    fn create_attention_masks(
        &self,
        seqlens: &[u32],
        input_positions: &Tensor,
        is_prefill: bool,
    ) -> Result<(Option<Vec<Tensor>>, Option<Vec<Tensor>>)> {
        let full_mask = if is_prefill && !seqlens.is_empty() {
            let mut masks = Vec::new();
            let mut start = 0u32;
            for seq_offset in seqlens {
                let seq_len = (seq_offset - start) as usize;
                if seq_len > 1 {
                    let mask = Tensor::zeros((seq_len, seq_len), self.dtype, &self.device)?;
                    attention_rs::mask::causal_mask(&mask, None)?;
                    masks.push(mask.unsqueeze(0)?.unsqueeze(0)?);
                } else {
                    masks.push(Tensor::zeros((1, 1, 1, 1), self.dtype, &self.device)?);
                }
                start = *seq_offset;
            }
            Some(masks)
        } else {
            None
        };

        let seqlens_vec = seqlens.to_vec();
        let sliding_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens_vec,
            self.cfg.sliding_window,
            is_prefill,
        );

        Ok((full_mask, sliding_mask))
    }

    pub fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        xs * (self.hidden_size as f64).sqrt()
    }

    fn get_per_layer_embeddings(
        &self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Option<Vec<Tensor>>> {
        let (emb_per_layer, pli_dim, proj, norm) = match (
            &self.embed_tokens_per_layer,
            self.hidden_size_per_layer_input,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            (Some(e), Some(d), Some(p), Some(n)) => (e, d, p, n),
            _ => return Ok(None),
        };

        let embedded = emb_per_layer.forward(input_ids)?;
        let embedded = (embedded * (pli_dim as f64).sqrt())?;

        let projected = inputs_embeds.apply(proj)?;
        let projected = (projected * (self.hidden_size as f64).powf(-0.5))?;

        let seq_len = input_ids.dim(0)?;
        let projected = projected.reshape((seq_len, self.num_hidden_layers, pli_dim))?;
        let projected = norm.forward(&projected)?;

        let embedded = embedded.reshape((seq_len, self.num_hidden_layers, pli_dim))?;
        let combined = ((projected + embedded)? * std::f64::consts::FRAC_1_SQRT_2)?;

        let mut per_layer = Vec::with_capacity(self.num_hidden_layers);
        for i in 0..self.num_hidden_layers {
            per_layer.push(combined.narrow(1, i, 1)?.squeeze(1)?);
        }
        Ok(Some(per_layer))
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            input_positions,
            kv_caches,
            input_metadata,
            false,
            false,
        )
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            input_positions,
            kv_caches,
            input_metadata,
            false,
            true,
        )
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embedded_inputs: bool,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let seqlens: Vec<u32> = if input_metadata.cu_seqlens_q.is_some() {
            input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..]
                .into()
        } else {
            Vec::new()
        };

        let (full_mask, sliding_mask) =
            self.create_attention_masks(&seqlens, input_positions, input_metadata.is_prefill)?;

        let mut xs = if embedded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };

        let per_layer_inputs = self.get_per_layer_embeddings(input_ids, &xs)?;

        if let Some(kv_caches) = kv_caches {
            for (i, ((k_cache, v_cache), layer)) in
                zip(kv_caches.iter(), self.layers.iter()).enumerate()
            {
                let pli = per_layer_inputs.as_ref().map(|v| &v[i]);
                xs = layer.forward(
                    &xs,
                    pli,
                    sliding_mask.as_ref(),
                    full_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                let pli = per_layer_inputs.as_ref().map(|v| &v[i]);
                xs = layer.forward(
                    &xs,
                    pli,
                    sliding_mask.as_ref(),
                    full_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
            }
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }

        let xs = self.norm.forward(&xs)?;

        if return_hidden {
            return xs.to_dtype(DType::F32);
        }

        let logits = self
            .lm_head
            .forward(&xs.to_dtype(self.dtype)?)?
            .to_dtype(DType::F32)?;

        let logits = match self.cfg.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };

        Ok(logits)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn get_vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
}
