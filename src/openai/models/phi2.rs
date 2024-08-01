use super::Config;
use crate::openai::models::linear::{linear_no_bias as linear, Linear};
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::with_tracing::{layer_norm, Embedding, LayerNorm};

use either::Either;
use serde::Deserialize;
use std::iter::zip;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Phi2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub partial_rotary_factor: f32,
    pub qk_layernorm: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: Option<usize>,
}

impl Phi2Config {
    pub fn into_config(self, use_flash_attn: bool, kv_cache_dtype: DType) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.layer_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: super::TokenID(Either::Left(self.bos_token_id)),
            eos_token_id: super::TokenID(Either::Left(self.eos_token_id)),
            max_seq_len: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            hidden_act: Some(self.hidden_act),
            tie_word_embeddings: false,
            rope_scaling: None,
            use_flash_attn,
            original_max_position_embeddings: self.original_max_position_embeddings,
            attention_bias: false,
            partial_rotary_factor: Some(self.partial_rotary_factor),
            qk_layer_rms_norm: Some(self.qk_layernorm),
            kv_cache_dtype,
            use_qkv_bias: None,
            custom_stop_tokens: None,
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    dim: usize,
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, _dtype: DType, dev: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let dim = (cfg.partial_rotary_factor.unwrap() * head_dim as f32) as usize;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| (1f64 / cfg.rope_theta.powf(i as f64 / dim as f64)) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, cfg.max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            dim,
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb(&self, xs: &Tensor, input_positions: &Vec<Vec<usize>>) -> Result<Tensor> {
        let (b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        let mut embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let cos = self.cos.narrow(0, seqlen_offset[0], seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offset[0], seq_len)?;
            let xs_rot = xs.i((b, .., .., ..self.dim))?.contiguous()?;
            let xs_pass = xs.i((b, .., .., self.dim..))?;
            let xs_rot = candle_nn::rotary_emb::rope(&xs_rot, &cos, &sin).unwrap();
            let embed = Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()?;
            embeds.push(embed);
        }
        Tensor::cat(&embeds, 0)
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            // This does not match the mixformers implementation where Gelu is used rather than
            // GeluNew.
            act: cfg.hidden_act.unwrap_or(Activation::Silu),
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    q_layernorm: Option<LayerNorm>,
    k_layernorm: Option<LayerNorm>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    attn: PagedAttention,
}

impl Attention {
    fn new(cfg: &Config, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let q_proj = linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let dense = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("dense"))?;
        // Alternative rope scalings are not supported.
        let rotary_emb = RotaryEmbedding::new(cfg, dtype, vb.device())?;
        let (q_layernorm, k_layernorm) = if cfg.qk_layer_rms_norm.unwrap() {
            let q_layernorm = layer_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_layernorm"))?;
            let k_layernorm = layer_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_layernorm"))?;
            (Some(q_layernorm), Some(k_layernorm))
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            q_layernorm,
            k_layernorm,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size: cfg.hidden_size,
            attn: PagedAttention::new(
                cfg.num_attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(cfg.num_key_value_heads),
                None,
                vb.device().clone(),
                None,
            )?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &Vec<Vec<usize>>,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = match &self.q_layernorm {
            None => query_states,
            Some(ln) => query_states.apply(ln)?,
        };
        let key_states = match &self.k_layernorm {
            None => key_states,
            Some(ln) => key_states.apply(ln)?,
        };
        let dtype = value_states.dtype();
        let (q, k, v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = query_states.reshape((b_size, self.num_heads, seq_len, self.head_dim))?;
            let k = key_states.reshape((b_size, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = value_states.reshape((b_size, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = query_states
                .reshape((b_size, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = key_states
                .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = value_states
                .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        };

        let q = self
            .rotary_emb
            .apply_rotary_emb(&q.to_dtype(DType::F32)?, input_positions)?;
        let k = self
            .rotary_emb
            .apply_rotary_emb(&k.to_dtype(DType::F32)?, input_positions)?;
        let v = v.to_dtype(DType::F32)?;

        let y = self.attn.forward(
            &q,
            &k,
            &v,
            attention_mask,
            cache.map(|(k_, _)| k_.clone()),
            cache.map(|(_, v_)| v_.clone()),
            input_metadata,
        )?;

        let y = if attention_mask.is_some() {
            y.transpose(1, 2)?
                .reshape(&[b_size, seq_len, self.hidden_size])?
        } else {
            y.reshape(&[b_size, seq_len, self.hidden_size])?
        };
        y.to_dtype(dtype)?.apply(&self.dense)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
    span: tracing::Span,
}

impl DecoderLayer {
    fn new(cfg: &Config, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, dtype, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            span: tracing::span!(tracing::Level::TRACE, "block"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        input_positions: &Vec<Vec<usize>>,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = xs;
        let xs = xs.apply(&self.input_layernorm)?;
        let attn_outputs =
            self.self_attn
                .forward(&xs, mask, input_positions, cache, input_metadata)?;
        let feed_forward_hidden_states = self.mlp.forward(&xs)?;
        attn_outputs + feed_forward_hidden_states + residual
    }
}

pub struct Phi2 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_layernorm: LayerNorm,
    lm_head: Linear,
    cfg: Config,
    device: Device,
}

impl Phi2 {
    pub fn new(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let final_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_m.pp("final_layernorm"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_m = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, dtype, vb_m.pp(layer_idx))?;
            layers.push(layer)
        }
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
            cfg: cfg.clone(),
            device: device.clone(),
        })
    }

    fn prepare_decoder_attention_mask(&self, b_size: usize, tgt_len: usize) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        mask.expand((b_size, 1, tgt_len, tgt_len))?
            .to_dtype(DType::F32)
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        input_positions: &Vec<Vec<usize>>,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.embed_tokens)?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len)?;
            Some(mask)
        };
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter_mut()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?
            }
        } else {
            for layer in self.layers.iter_mut() {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?
            }
        }
        xs.apply(&self.final_layernorm)?
            .i((.., seq_len - 1, ..))?
            .apply(&self.lm_head)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
