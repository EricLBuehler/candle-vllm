use super::{
    attention::Attention,
    layers::moe::{FusedMoe, FusedMoeFp8, FusedMoeISQ, FusedMoeMxfp4, FusedMoeNvfp4},
    rotary_emb::ScalingRotaryEmbedding,
    utils::{apply_rms_norm_fp32, resolve_input_seqlens},
    Config, MoEConfig, QwenMoEConfig,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{embedding, rms_norm_x, Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::mask::get_attention_causal_mask;
use crate::InputMetadata;
use candle::{DType, Device, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::RmsNorm;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

enum MoeVariant {
    FusedMoe(FusedMoe),
    FusedMoeISQ(FusedMoeISQ),
    FusedMoeFp8(FusedMoeFp8),
    FusedMoeMxfp4(FusedMoeMxfp4),
    FusedMoeNvfp4(FusedMoeNvfp4),
}

impl MoeVariant {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
            Self::FusedMoeISQ(m) => m.forward(xs, is_prefill),
            Self::FusedMoeFp8(m) => m.forward(xs, is_prefill),
            Self::FusedMoeMxfp4(m) => m.forward(xs, is_prefill),
            Self::FusedMoeNvfp4(m) => m.forward(xs, is_prefill),
        }
    }
}

struct MiniMaxDecoderLayer {
    self_attn: Attention,
    moe: MoeVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MiniMaxDecoderLayer {
    fn new(
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
        _layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            cfg.sliding_window,
        )?;

        let moe_vb = vb.pp("block_sparse_moe");

        let moe = if let Some(ref quant_cfg) = cfg.quantization_config {
            if quant_cfg.quant_method == "fp8" {
                MoeVariant::FusedMoeFp8(FusedMoeFp8::new(
                    cfg,
                    moe_vb.clone(),
                    comm.clone(),
                    dtype,
                    quant_cfg,
                )?)
            } else if quant_cfg.quant_method == "mxfp4" {
                MoeVariant::FusedMoeMxfp4(FusedMoeMxfp4::new(
                    cfg,
                    moe_vb.clone(),
                    comm.clone(),
                    dtype,
                )?)
            } else if quant_cfg.quant_method == "nvfp4" {
                let mut m = FusedMoeNvfp4::new(cfg, moe_vb.clone(), comm.clone(), dtype)?;
                m.set_sigmoid_routing();
                MoeVariant::FusedMoeNvfp4(m)
            } else {
                MoeVariant::FusedMoe(FusedMoe::new(cfg, moe_vb.clone(), comm.clone(), dtype)?)
            }
        } else if cfg.isq_quant.is_some() {
            MoeVariant::FusedMoeISQ(FusedMoeISQ::new(cfg, moe_vb.clone(), comm.clone(), dtype)?)
        } else {
            MoeVariant::FusedMoe(FusedMoe::new(cfg, moe_vb.clone(), comm.clone(), dtype)?)
        };

        let input_layernorm = rms_norm_x(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
            DType::F32,
            false,
        )?;
        let post_attention_layernorm = rms_norm_x(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
            DType::F32,
            false,
        )?;

        Ok(Self {
            self_attn,
            moe,
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
        let xs = apply_rms_norm_fp32(&self.input_layernorm, xs)?;
        let attn_output =
            self.self_attn
                .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = (attn_output + residual)?;
        let residual = &xs;
        let xs = apply_rms_norm_fp32(&self.post_attention_layernorm, &xs)?;
        let mlp_output = self.moe.forward(&xs, input_metadata.is_prefill)?;
        residual + mlp_output
    }
}

pub struct MiniMaxForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<MiniMaxDecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    cfg: Config,
    vocab_size: usize,
}

impl MiniMaxForCausalLM {
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
        config.max_seq_len = config.effective_max_seq_len();
        config.attention_bias = Some(
            config
                .use_qkv_bias
                .or(config.attention_bias)
                .unwrap_or(false),
        );

        config.isq_quant = if config.quantization_config.is_some() {
            None
        } else {
            isq
        };

        if config.moe_config.is_none() {
            let f = std::fs::read(filename).map_err(candle::Error::wrap)?;
            let mut moe_cfg: Option<QwenMoEConfig> = Self::parse_minimax_moe_config(&f);
            if let Some(ref mut m) = moe_cfg {
                m.norm_topk_prob = true;
            }
            if let Some(moe) = moe_cfg {
                config.moe_config = Some(MoEConfig::QwenMoE(moe));
            }
        } else {
            if let Some(MoEConfig::QwenMoE(ref mut moe_cfg)) = config.moe_config {
                moe_cfg.norm_topk_prob = true;
            }
        }

        Ok(config)
    }

    fn parse_minimax_moe_config(raw_cfg: &[u8]) -> Option<QwenMoEConfig> {
        if let Ok(cfg) = serde_json::from_slice::<QwenMoEConfig>(raw_cfg) {
            if cfg.num_experts.unwrap_or(0) > 0 {
                return Some(cfg);
            }
        }
        let mut raw_cfg_json: serde_json::Value = serde_json::from_slice(raw_cfg).ok()?;
        let raw_cfg_obj = raw_cfg_json.as_object_mut()?;

        if !raw_cfg_obj.contains_key("moe_intermediate_size") {
            let intermediate_size = raw_cfg_obj.get("intermediate_size")?.clone();
            raw_cfg_obj.insert("moe_intermediate_size".to_string(), intermediate_size);
        }

        serde_json::from_value(raw_cfg_json).ok()
    }

    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, cfg, device, true)?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();

        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = MiniMaxDecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                comm.clone(),
                dtype,
                layer_idx,
            )?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
        }

        let norm = rms_norm_x(
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
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
            vocab_size: cfg.vocab_size,
        })
    }

    pub fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
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
        let seqlens = resolve_input_seqlens(input_metadata)?;

        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );

        let mut xs = if embedded_inputs {
            input_ids.to_owned()
        } else {
            self.embed_forward(input_ids)?
        };

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in kv_caches.iter().zip(self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }

        let xs = apply_rms_norm_fp32(&self.norm, &xs)?;

        if return_hidden {
            return xs.to_dtype(DType::F32);
        }
        self.lm_head
            .forward(&xs.to_dtype(self.dtype)?)?
            .to_dtype(DType::F32)
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

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
