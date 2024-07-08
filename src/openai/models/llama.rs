use super::Config;
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_core as candle;
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
pub const MAX_SEQ_LEN: usize = 4096;
use std::iter::zip;
#[derive(Debug, Clone, serde::Deserialize)]
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
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta as f64,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            max_seq_len: MAX_SEQ_LEN,
            sliding_window: None,
            hidden_act: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            original_max_position_embeddings: None,
            attention_bias: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,
}

impl Cache {
    pub fn new(dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f64 / n_elem as f64) as f32)
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, config.max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
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
    span: tracing::Span,
    span_rot: tracing::Span,
    attn: PagedAttention,
    cos_sin_cache: Cache,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = self.cos_sin_cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.cos_sin_cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &mut self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        index_pos: usize,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let (q, k, v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = q.reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            (q, k, v)
        };

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

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
            y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?
        } else {
            y.reshape(&[b_sz, seq_len, hidden_size])?
        };
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }

    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: head_dim,
            span,
            span_rot,
            attn: PagedAttention::new(
                cfg.num_attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(cfg.num_key_value_heads),
                None,
                vb.device().clone(),
                None,
            )?,
            cos_sin_cache: Cache::new(dtype, &cfg, device)?,
        })
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
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
        attention_mask: Option<&Tensor>,
        index_pos: usize,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self
            .attn
            .forward(&x, attention_mask, index_pos, cache, input_metadata)?
            + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, dtype, device)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
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
    dtype: DType,
    device: Device,
}

impl Llama {
    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(_b_sz, seq_len, index_pos)?;
            Some(mask)
        };
        let mut x = self.wte.forward(x)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), block) in zip(kv_caches.iter(), &mut self.blocks) {
                x = block.forward(
                    &x,
                    attention_mask.as_ref(),
                    index_pos,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
            }
        } else {
            for block in &mut self.blocks {
                x = block.forward(&x, attention_mask.as_ref(), index_pos, None, input_metadata)?;
            }
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(&format!("model.layers.{i}")), cfg, dtype, device).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cfg: cfg.clone(),
            dtype: dtype,
            device: device.clone(),
        })
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
