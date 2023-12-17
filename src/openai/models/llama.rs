/// Llama LLM, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama.rs
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear};
use serde::Deserialize;
use std::iter::zip;
use std::sync::Arc;

use crate::bindings::rotary_embedding;
use crate::openai::responses::APIError;
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use crate::{convert_candle_to_tch, convert_tch_to_ptr};

use super::ConfigLike;

pub const MAX_SEQ_LEN: usize = 4096;

#[derive(Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
}

impl ConfigLike for LlamaConfig {
    fn get_num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn get_num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn get_num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn get_sliding_window(&self) -> Option<usize> {
        None
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
        }
    }
}

#[derive(Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl ConfigLike for Config {
    fn get_num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }
    fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn get_num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn get_num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn get_sliding_window(&self) -> Option<usize> {
        None
    }
}

impl Config {
    pub fn config_7b_v1() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    pub fn config_7b_v2() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

fn embedding(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Embedding> {
    let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, cfg.hidden_size))
}

struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    attn: PagedAttention,
    cos_sin_cache: Tensor,
}

impl CausalSelfAttention {
    fn compute_cos_sin_cache(
        config: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor, APIError> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device).map_err(APIError::from)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)
            .map_err(APIError::from)?
            .to_dtype(DType::F32)
            .map_err(APIError::from)?
            .reshape((MAX_SEQ_LEN, 1))
            .map_err(APIError::from)?
            .matmul(
                &theta
                    .reshape((1, theta.elem_count()))
                    .map_err(APIError::from)?,
            )
            .map_err(APIError::from)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let idx_theta =
            Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1).map_err(APIError::from)?;
        let cos = idx_theta
            .cos()
            .map_err(APIError::from)?
            .to_dtype(dtype)
            .map_err(APIError::from)?;
        let sin = idx_theta
            .sin()
            .map_err(APIError::from)?
            .to_dtype(dtype)
            .map_err(APIError::from)?;
        let last = cos.dims().len() - 1;
        Tensor::cat(&[cos, sin], last).map_err(APIError::from)
    }

    fn apply_rotary_emb(&mut self, q: &mut Tensor, k: &mut Tensor, mut positions: Tensor) {
        let mut q_tch = convert_candle_to_tch(q);
        let (q_ptr, _) = convert_tch_to_ptr(&mut q_tch);

        let mut k_tch = convert_candle_to_tch(k);
        let (k_ptr, _) = convert_tch_to_ptr(&mut k_tch);

        let mut pos_tch = convert_candle_to_tch(&mut positions);
        let (pos_ptr, _) = convert_tch_to_ptr(&mut pos_tch);

        let mut cache_tch = convert_candle_to_tch(&mut self.cos_sin_cache);
        let (cache_ptr, _) = convert_tch_to_ptr(&mut cache_tch);

        unsafe {
            rotary_embedding(
                pos_ptr,
                q_ptr,
                k_ptr,
                self.head_dim.try_into().unwrap(),
                cache_ptr,
                false,
            );
        }
    }

    fn forward(
        &mut self,
        x: &Tensor,
        positions: &Tensor,
        input_metadata: &mut InputMetadata,
        cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor, APIError> {
        let (b_sz, seq_len, _) = x.dims3().map_err(APIError::from)?;
        let q = self.q_proj.forward(x).map_err(APIError::from)?;
        let k = self.k_proj.forward(x).map_err(APIError::from)?;
        let v = self.v_proj.forward(x).map_err(APIError::from)?;

        let mut q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))
            .map_err(APIError::from)?
            .transpose(1, 2)
            .map_err(APIError::from)?;
        let mut k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))
            .map_err(APIError::from)?
            .transpose(1, 2)
            .map_err(APIError::from)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))
            .map_err(APIError::from)?
            .transpose(1, 2)
            .map_err(APIError::from)?;

        self.apply_rotary_emb(&mut q, &mut k, positions.clone());

        let dtype = q.dtype();
        let device = q.device().clone();
        let attn_output = self.attn.forward(
            q,
            k,
            v,
            cache.map(|(k, _)| k.clone()),
            cache.map(|(_, v)| v.clone()),
            input_metadata,
            dtype,
            device,
        )?;

        let y = self.o_proj.forward(&attn_output).map_err(APIError::from)?;
        Ok(y)
    }

    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self, APIError> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj")).map_err(APIError::from)?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj")).map_err(APIError::from)?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj")).map_err(APIError::from)?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj")).map_err(APIError::from)?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            attn: PagedAttention::new(
                cfg.num_attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(cfg.num_key_value_heads),
                None,
                vb.device().clone(),
                None,
            )
            .map_err(APIError::from)?,
            cos_sin_cache: Self::compute_cos_sin_cache(cfg, device, dtype)?,
        })
    }
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> candle_core::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &mut self,
        x: &Tensor,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor, APIError> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x).map_err(APIError::from)?;
        let x = (self
            .attn
            .forward(&x, positions, input_metadata, cache)
            .map_err(APIError::from)?
            + residual)
            .map_err(APIError::from)?;
        let residual = &x;
        let x = (self
            .mlp
            .forward(&self.rms_2.forward(&x).map_err(APIError::from)?)
            .map_err(APIError::from)?
            + residual)
            .map_err(APIError::from)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self, APIError> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, dtype, device)
            .map_err(APIError::from)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg).map_err(APIError::from)?;
        let rms_1 = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
            .map_err(APIError::from)?;
        let rms_2 = RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(APIError::from)?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cfg: Config,
}

impl Llama {
    pub fn forward(
        &mut self,
        x: &Tensor,
        positions: &Tensor,
        kv_caches: Option<Arc<Vec<(Tensor, Tensor)>>>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor, APIError> {
        let (_b_sz, seq_len) = x.dims2().map_err(APIError::from)?;
        let mut x = self.wte.forward(x).map_err(APIError::from)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), block) in zip(kv_caches.iter(), &mut self.blocks) {
                x = block.forward(&x, positions, Some((k_cache, v_cache)), input_metadata)?;
            }
        } else {
            for block in &mut self.blocks {
                x = block.forward(&x, positions, None, input_metadata)?;
            }
        }
        let x = self.ln_f.forward(&x).map_err(APIError::from)?;
        let x = x.i((.., seq_len - 1, ..)).map_err(APIError::from)?;
        let logits = self.lm_head.forward(&x).map_err(APIError::from)?;
        logits.to_dtype(DType::F32).map_err(APIError::from)
    }

    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let wte = embedding(cfg, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(&format!("model.layers.{i}")), cfg, dtype, device).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cfg: cfg.clone(),
        })
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
