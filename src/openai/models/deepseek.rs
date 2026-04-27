#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use super::{Config, MoEConfig, QuantConfig, QwenMoEConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, Comm, ReplicatedLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
    VarBuilder,
};
use crate::openai::models::layers::mla_attention::{MlaAttention, MlaConfig};
use crate::openai::models::layers::moe::{
    FusedMoe, FusedMoeFp8, FusedMoeISQ, FusedMoeMxfp4, FusedMoeNvfp4,
};
use crate::openai::models::layers::others::{rms_norm, NormX};
use crate::openai::models::layers::rotary_emb::ScalingRotaryEmbedding;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::InputMetadata;
use candle::{DType, Device, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::Embedding;
use parking_lot::RwLock;
use serde::Deserialize;
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use super::TokenID;

struct Mlp {
    gate_proj: TensorParallelColumnLinear,
    up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, intermediate_size: usize, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let gate_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            intermediate_size,
            false,
            vb.pp("gate_proj"),
            comm.clone(),
            &cfg.isq_quant,
            &cfg.quantization_config,
        )?;
        let up_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            intermediate_size,
            false,
            vb.pp("up_proj"),
            comm.clone(),
            &cfg.isq_quant,
            &cfg.quantization_config,
        )?;
        let down_proj = TensorParallelRowLinear::load_with_hints(
            intermediate_size,
            hidden_sz,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.isq_quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap_or(candle_nn::Activation::Silu),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.act_fn.forward(&self.gate_proj.forward(xs)?)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(&lhs * &rhs)?)
    }
}

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
    FusedMoeMxfp4(FusedMoeMxfp4),
    FusedMoeNvfp4(FusedMoeNvfp4),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeFp8(m) => m.forward(xs, is_prefill),
            Self::FusedMoeMxfp4(m) => m.forward(xs, is_prefill),
            Self::FusedMoeNvfp4(m) => m.forward(xs, is_prefill),
        }
    }
}

pub struct DeepSeekDecoderLayer {
    self_attn: MlaAttention,
    mlp: MoeOrMlp,
    shared_expert: Option<Mlp>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
}

impl DeepSeekDecoderLayer {
    pub fn new(
        vb: VarBuilder,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        mla_cfg: &MlaConfig,
        dtype: DType,
        layer_idx: usize,
    ) -> Result<Self> {
        let moe_cfg = match &config.moe_config {
            Some(MoEConfig::QwenMoE(m)) => m.clone(),
            _ => candle::bail!("DeepSeek requires moe_config: QwenMoE"),
        };

        let self_attn =
            MlaAttention::new(vb.pp("self_attn"), comm.clone(), mla_cfg, config, dtype)?;

        let is_moe_layer = layer_idx >= moe_cfg.first_k_dense_replace.unwrap_or(0)
            && moe_cfg.num_experts.is_some();

        let mlp = if is_moe_layer {
            if let Some(ref quant_config) = config.quantization_config {
                if quant_config.quant_method == "fp8" {
                    MoeOrMlp::FusedMoeFp8(FusedMoeFp8::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                        quant_config,
                    )?)
                } else if quant_config.quant_method == "mxfp4" {
                    MoeOrMlp::FusedMoeMxfp4(FusedMoeMxfp4::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                } else if quant_config.quant_method == "nvfp4" {
                    MoeOrMlp::FusedMoeNvfp4(FusedMoeNvfp4::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                } else if config.isq_quant.is_some() {
                    MoeOrMlp::FusedMoeISQ(FusedMoeISQ::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                } else {
                    MoeOrMlp::FusedMoe(FusedMoe::new(
                        config,
                        vb.pp("mlp").clone(),
                        comm.clone(),
                        dtype,
                    )?)
                }
            } else if config.isq_quant.is_some() {
                MoeOrMlp::FusedMoeISQ(FusedMoeISQ::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            } else {
                MoeOrMlp::FusedMoe(FusedMoe::new(
                    config,
                    vb.pp("mlp").clone(),
                    comm.clone(),
                    dtype,
                )?)
            }
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                config,
                config.intermediate_size,
                vb.pp("mlp"),
                comm.clone(),
            )?)
        };

        let shared_expert = if is_moe_layer {
            if let Some(intermediate_size) = moe_cfg.shared_expert_intermediate_size {
                if intermediate_size > 0 {
                    let mlp = Mlp::new(
                        config,
                        intermediate_size * moe_cfg.n_shared_experts.unwrap_or(1),
                        vb.pp("mlp.shared_experts"),
                        comm.clone(),
                    )?;
                    Some(mlp)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
            DType::F32,
            false,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            DType::F32,
            false,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            shared_expert,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let rope = self.rotary_emb.clone();
        let attn_output = self.self_attn.forward(
            &xs,
            &Some(rope),
            attention_mask,
            positions,
            cache,
            input_metadata,
        )?;
        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;

        let shared_output = if let Some(shared_expert) = &self.shared_expert {
            Some(shared_expert.forward(&xs)?)
        } else {
            None
        };
        let mlp_output = self.mlp.forward(&xs, input_metadata.is_prefill)?;
        let out = if let Some(shared_output) = shared_output {
            (residual + (mlp_output + shared_output)?)?
        } else {
            (residual + mlp_output)?
        };
        Ok(out)
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub n_shared_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_layer_freq: Option<usize>,
    #[serde(default)]
    pub first_k_dense_replace: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: Option<f64>,
    pub hidden_act: Option<candle_nn::Activation>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub rope_theta: Option<f64>,
    pub rope_scaling: Option<serde_json::Value>,
    pub attention_bias: Option<bool>,
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub v_head_dim: Option<usize>,
    pub qk_nope_head_dim: Option<usize>,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    pub sliding_window: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
    pub bos_token_id: Option<TokenID>,
    pub eos_token_id: Option<TokenID>,
}

pub struct DeepSeek {
    embed_tokens: Embedding,
    layers: Vec<DeepSeekDecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
}

impl DeepSeek {
    pub fn load_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let f = std::fs::read(filename.clone()).map_err(candle::Error::wrap)?;
        let ds_cfg: DeepSeekConfig = serde_json::from_slice(&f).map_err(candle::Error::wrap)?;
        let raw = String::from_utf8(f.clone()).map_err(candle::Error::wrap)?;

        let hidden_act = ds_cfg.hidden_act.unwrap_or(candle_nn::Activation::Silu);

        let num_kv_heads = ds_cfg
            .num_key_value_heads
            .unwrap_or(ds_cfg.num_attention_heads);

        let rope_theta = ds_cfg.rope_theta.unwrap_or(10000.0);

        let rope_scaling = ds_cfg.rope_scaling.as_ref().and_then(|v| {
            serde_json::from_value::<
                std::collections::HashMap<String, crate::openai::models::ScalingValue>,
            >(v.clone())
            .ok()
        });

        let moe_intermediate_size = ds_cfg.moe_intermediate_size;
        let n_shared_experts = ds_cfg.n_shared_experts;
        let shared_expert_intermediate_size = n_shared_experts.map(|n| moe_intermediate_size * n);

        let qwen_moe = QwenMoEConfig {
            moe_intermediate_size,
            shared_expert_intermediate_size,
            num_experts: ds_cfg.n_routed_experts,
            mlp_only_layers: None,
            decoder_sparse_step: None,
            norm_topk_prob: ds_cfg.norm_topk_prob,
            num_experts_per_tok: ds_cfg.num_experts_per_tok.unwrap_or(8),
            routed_scaling_factor: ds_cfg.routed_scaling_factor,
            first_k_dense_replace: ds_cfg.first_k_dense_replace,
            n_shared_experts,
        };

        let isq_quant = if ds_cfg.quantization_config.is_some() {
            None
        } else {
            isq
        };

        let mut quant_config = ds_cfg.quantization_config.clone();
        if let Some(ref mut qcfg) = quant_config {
            qcfg.normalize_compressed_tensors();
        }

        let mut config = Config {
            architectures: Some(vec!["DeepseekV3ForCausalLM".to_string()]),
            hidden_size: ds_cfg.hidden_size,
            head_dim: Some(ds_cfg.hidden_size / ds_cfg.num_attention_heads),
            intermediate_size: ds_cfg.intermediate_size,
            vocab_size: ds_cfg.vocab_size,
            num_hidden_layers: ds_cfg.num_hidden_layers,
            num_attention_heads: ds_cfg.num_attention_heads,
            num_key_value_heads: Some(num_kv_heads),
            rms_norm_eps: ds_cfg.rms_norm_eps,
            rope_theta,
            rope_local_base_freq: None,
            bos_token_id: ds_cfg.bos_token_id,
            eos_token_id: ds_cfg.eos_token_id,
            max_seq_len: ds_cfg.max_position_embeddings,
            sliding_window: ds_cfg.sliding_window,
            sliding_window_pattern: None,
            hidden_act: Some(hidden_act),
            hidden_activation: None,
            tie_word_embeddings: ds_cfg.tie_word_embeddings,
            rope_scaling,
            max_position_embeddings: Some(ds_cfg.max_position_embeddings),
            original_max_position_embeddings: None,
            attention_bias: ds_cfg.attention_bias.or(Some(false)),
            partial_rotary_factor: None,
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: quant_config,
            moe_config: Some(MoEConfig::QwenMoE(qwen_moe)),
            isq_quant,
            fp8_kvcache: None,
            extra_config_json: Some(raw),
        };

        config.apply_rope_overrides();
        config.max_seq_len = config.effective_max_seq_len();
        Ok(config)
    }

    pub fn get_config(&self) -> &Config {
        &self.config
    }

    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let mla_cfg = MlaConfig::from_config(cfg);
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut mla_rope_cfg = cfg.clone();
        mla_rope_cfg.head_dim = Some(mla_cfg.qk_rope_head_dim);
        mla_rope_cfg.partial_rotary_factor = None;
        let is_qvar_builder = cfg.isq_quant.is_some();
        let rotary_dtype = if is_qvar_builder || cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            rotary_dtype,
            &mla_rope_cfg,
            device,
            false,
        )?);

        let reporter = progress_reporter.clone();
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let layer = DeepSeekDecoderLayer::new(
                vb_m.pp("layers").pp(i),
                comm.clone(),
                rotary_emb.clone(),
                cfg,
                &mla_cfg,
                dtype,
                i,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let norm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_m.pp("norm"),
            DType::F32,
            false,
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            ReplicatedLinear::from_weight_bias(embed_tokens.embeddings().clone(), None)?
        } else {
            ReplicatedLinear::load_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: cfg.clone(),
            dtype,
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(xs)?;
        let needs_f32 = if let Some(qcfg) = &self.config.quantization_config {
            !matches!(qcfg.quant_method.as_str(), "nvfp4" | "mxfp4" | "fp8")
        } else {
            false
        };
        if needs_f32 && xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)
        } else {
            Ok(xs)
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, positions, kv_caches, input_metadata, false)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        let seqlens = if let Some(seqlens) = input_metadata.seqlens.as_ref() {
            seqlens.clone()
        } else if input_metadata.cu_seqlens_q.is_some() {
            input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..]
                .to_vec()
        } else {
            Vec::new()
        };
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            &seqlens,
            self.config.sliding_window,
            input_metadata.is_prefill,
        );

        let mut xs = if embeded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        }

        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        let logits = self
            .lm_head
            .forward(&xs.to_dtype(self.dtype)?)?
            .to_dtype(DType::F32)?;
        Ok(logits)
    }

    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward(input_ids, positions, kv_caches, input_metadata)
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
