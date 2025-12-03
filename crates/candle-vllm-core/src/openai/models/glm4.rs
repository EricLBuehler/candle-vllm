use super::{attention::Attention, rotary_emb::ScalingRotaryEmbedding, Config};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::distributed::{
    embedding, rms_norm, Comm, MergedParallelColumnLinear, ReplicatedLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::mask::get_attention_causal_mask;
use crate::InputMetadata;
use candle::{DType, Device, Result, Tensor};
use candle_core as candle;
use candle_nn::{Embedding, Module, RmsNorm};
use parking_lot::RwLock;
use std::iter::zip;
use std::path::PathBuf;
pub use std::rc::Rc;
use std::sync::Arc;

impl GLM4 {
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
        config.num_hidden_layers = config.num_hidden_layers;
        config.max_seq_len = config.max_position_embeddings.unwrap_or(32768);
        config.attention_bias = Some(config.attention_bias.unwrap_or(false));
        config.bos_token_id = Some(
            config
                .bos_token_id
                .unwrap_or(super::TokenID(either::Either::Left(Some(128256)))),
        );
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
}

#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_up_proj: MergedParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let gate_up_proj = MergedParallelColumnLinear::load_merged_with_hints(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            2,
            vb.pp("gate_up_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;

        let down_proj = TensorParallelRowLinear::load_with_hints(
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.quant,
            &cfg.quantization_config,
        )?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap(),
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up_states = self.gate_up_proj.forward(xs)?;
        let up_states = (&gate_up_states[1] * self.act_fn.forward(&gate_up_states[0])?)?;
        self.down_proj.forward(&up_states)
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    post_mlp_layernorm: RmsNorm,
    post_self_attn_layernorm: RmsNorm,
    mlp: MLP,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let post_self_attn_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_self_attn_layernorm"),
        )?;
        let post_mlp_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_mlp_layernorm"),
        )?;
        let self_attn = Attention::new(
            rotary_emb.clone(),
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            cfg.sliding_window,
        )?;
        let mlp = MLP::new(cfg, vb.pp("mlp"), comm.clone())?;
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            post_self_attn_layernorm,
            post_mlp_layernorm,
            self_attn,
            mlp,
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
        let hidden_states = self.input_layernorm.forward(xs)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            input_positions,
            cache,
            input_metadata,
        )?;
        let hidden_states = self.post_self_attn_layernorm.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_mlp_layernorm.forward(&hidden_states)?;
        residual + hidden_states
    }
}

pub struct GLM4 {
    embedding: Embedding,
    layers: Vec<DecoderLayer>,
    lm_head: ReplicatedLinear,
    norm: RmsNorm,
    dtype: DType,
    device: Device,
    cfg: Config,
}

impl GLM4 {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let reporter = progress_reporter.clone();

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, cfg, device, false)?);
        let embedding = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_index in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_index), comm.clone())?;
            layers.push(layer);
            reporter.write().set_progress(layer_index + 1);
        }

        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("lm_head"),
            &None,
            &None,
        )?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embedding,
            layers,
            lm_head,
            norm,
            dtype,
            device: device.clone(),
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
        let mut xs = self.embedding.forward(input_ids)?;

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
