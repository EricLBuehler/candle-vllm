use super::{Config, MoEConfig, QwenMoEConfig};
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
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

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
            act_fn: cfg.hidden_act.unwrap(),
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

pub struct GLM4MoeLiteDecoderLayer {
    self_attn: MlaAttention,
    mlp: MoeOrMlp,
    shared_expert: Option<Mlp>,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
}

impl GLM4MoeLiteDecoderLayer {
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
            _ => candle::bail!("GLM4 MoE Lite requires moe_config: QwenMoE"),
        };

        let self_attn =
            MlaAttention::new(vb.pp("self_attn"), comm.clone(), mla_cfg, config, dtype)?;

        let mlp = if layer_idx >= moe_cfg.first_k_dense_replace.unwrap_or(0) {
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

        let is_moe_layer = layer_idx >= moe_cfg.first_k_dense_replace.unwrap_or(0);
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
        _layer_idx: usize,
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

pub struct GLM4MoeLiteForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<GLM4MoeLiteDecoderLayer>,
    norm: NormX,
    lm_head: ReplicatedLinear,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
}

impl GLM4MoeLiteForCausalLM {
    pub fn load_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let f = std::fs::read(filename.clone()).map_err(candle::Error::wrap)?;
        let mut config: Config = serde_json::from_slice(&f).map_err(candle::Error::wrap)?;
        let raw = String::from_utf8(f.clone()).map_err(candle::Error::wrap)?;
        config.extra_config_json = Some(raw);

        config.isq_quant = if config.quantization_config.is_some() {
            None
        } else {
            isq
        };

        if config.moe_config.is_none() {
            let from_root: Option<QwenMoEConfig> = serde_json::from_slice::<QwenMoEConfig>(&f).ok();
            if let Some(moe_cfg) = from_root {
                config.moe_config = Some(MoEConfig::QwenMoE(moe_cfg));
            }
        }

        if let Some(MoEConfig::QwenMoE(ref mut moe_cfg)) = config.moe_config {
            if moe_cfg.shared_expert_intermediate_size.is_none() {
                if let Some(n) = moe_cfg.n_shared_experts {
                    if n > 0 {
                        moe_cfg.shared_expert_intermediate_size =
                            Some(moe_cfg.moe_intermediate_size);
                    }
                }
            }
        }

        if let Some(ref mut qcfg) = config.quantization_config {
            qcfg.normalize_compressed_tensors();
        }

        config.apply_rope_overrides();
        config.max_seq_len = config.effective_max_seq_len();
        Ok(config)
    }

    pub fn get_config(&self) -> &Config {
        &self.config
    }

    pub fn new(
        vb: &VarBuilder,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_gpt_neox: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let mla_cfg = MlaConfig::from_config(config);
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb_m.pp("embed_tokens"),
        )?;

        let mut mla_rope_cfg = config.clone();
        mla_rope_cfg.head_dim = Some(mla_cfg.qk_rope_head_dim);
        mla_rope_cfg.partial_rotary_factor = None;
        let is_qvar_builder = config.isq_quant.is_some();
        let rotary_dtype = if is_qvar_builder || config.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            rotary_dtype,
            &mla_rope_cfg,
            device,
            is_gpt_neox,
        )?);

        let reporter = progress_reporter.clone();
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = GLM4MoeLiteDecoderLayer::new(
                vb_m.pp("layers").pp(i),
                comm.clone(),
                rotary_emb.clone(),
                config,
                &mla_cfg,
                dtype,
                i,
            )?;
            layers.push(layer);
            reporter.write().set_progress(i + 1);
        }

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb_m.pp("norm"),
            DType::F32,
            false,
        )?;

        let lm_head = ReplicatedLinear::load_no_bias(
            config.hidden_size,
            config.vocab_size,
            if config.tie_word_embeddings {
                vb_m.pp("embed_tokens")
            } else {
                vb.pp("lm_head")
            },
            &config.isq_quant,
            &config.quantization_config,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size: config.vocab_size,
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(xs)
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
            for (i, ((k_cache, v_cache), layer)) in
                zip(kv_caches.iter(), self.layers.iter()).enumerate()
            {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                    i,
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
