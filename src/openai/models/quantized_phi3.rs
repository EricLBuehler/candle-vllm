use super::Config;
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use crate::SpecificConfig;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, RmsNorm};
use either::Either;
use std::iter::zip;
use std::sync::{Arc, RwLock};
#[derive(Debug, Clone)]
struct QLinear {
    inner: candle_core::quantized::QMatMul,
}

impl QLinear {
    fn new<R: std::io::Read + std::io::Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let w = ct.tensor(r, &format!("{name}.weight"), device)?;
        let inner = candle_core::quantized::QMatMul::from_qtensor(w)?;
        Ok(Self { inner })
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    ffn_up: QLinear,
    ffn_down: QLinear,
    i_size: usize,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = xs.apply(&self.ffn_up)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.silu()?)?;
        up_states.apply(&self.ffn_down)
    }
}

fn rms_norm(w: QTensor, eps: f64) -> Result<RmsNorm> {
    let w = w.dequantize(&w.device())?;
    let rms = RmsNorm::new(w, eps);
    Ok(rms)
}

struct LayerWeights {
    attn_qkv: QLinear,
    attn_output: QLinear,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    attn: PagedAttention,
    dtype: DType,
}

impl LayerWeights {
    fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &[Vec<usize>],
    ) -> Result<(Tensor, Tensor)> {
        let (b_size, _h, seq_len, _n_embd) = q.dims4()?;
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let cos = self.cos.narrow(0, seqlen_offset[0], seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offset[0], seq_len)?;
            let x_q = q.narrow(0, b, 1)?;
            let x_k = k.narrow(0, b, 1)?;
            let q_embed = candle_nn::rotary_emb::rope(&x_q, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&x_k, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }

    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let qkv = self.attn_qkv.forward(x)?;

        let query_pos = self.n_head * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?.to_dtype(self.dtype)?;
        let k = qkv
            .narrow(D::Minus1, query_pos, self.n_kv_head * self.head_dim)?
            .to_dtype(self.dtype)?;
        let v = qkv
            .narrow(
                D::Minus1,
                query_pos + self.n_kv_head * self.head_dim,
                self.n_kv_head * self.head_dim,
            )?
            .to_dtype(self.dtype)?;

        let (q, k, v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q.contiguous()?, k.contiguous()?, v.contiguous()?)
        };

        let (q, k) = self.apply_rotary_emb(&q, &k, input_positions)?;

        let y = self
            .attn
            .forward(
                &q,
                &k,
                &v,
                mask,
                cache.map(|(k_, _)| k_.clone()),
                cache.map(|(_, v_)| v_.clone()),
                input_metadata,
                None,
            )?
            .reshape((b_sz, seq_len, ()))?;

        let y = self.attn_output.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

pub struct GGUFPhi3 {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: RmsNorm,
    output: QLinear,
    cfg: Config,
    dtype: DType,
    device: Device,
}

fn precomput_freqs_cis(
    head_dim: usize,
    max_seq_len: usize,
    freq_base: f32,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?.to_dtype(dtype)?;
    let sin = idx_theta.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

impl GGUFPhi3 {
    pub fn into_config(
        embedding_length: usize,
        i_size: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rms_eps: f64,
        max_seq_len: usize,
        kv_cache_dtype: DType,
        s_cfg: SpecificConfig,
    ) -> Config {
        Config {
            hidden_size: embedding_length,
            head_dim: Some(embedding_length / head_count),
            intermediate_size: i_size,
            vocab_size: 0,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: head_count_kv,
            rms_norm_eps: rms_eps,
            rope_theta: 0.,
            rope_local_base_freq: None,
            use_flash_attn: false,
            bos_token_id: super::TokenID(Either::Left(Some(1))),
            eos_token_id: super::TokenID(Either::Left(Some(2))),
            max_seq_len,
            sliding_window: None,
            sliding_window_pattern: None,
            hidden_act: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            original_max_position_embeddings: Some(max_seq_len),
            attention_bias: false,
            partial_rotary_factor: None,
            qk_layer_rms_norm: None,
            kv_cache_dtype,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            specific_config: s_cfg,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: None,
            moe_config: None,
        }
    }

    pub fn get_num_of_layers(ct: gguf_file::Content) -> Result<usize> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        Ok(md_get("phi3.block_count")?.to_u32()? as usize)
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
        s_cfg: SpecificConfig,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let reporter = progress_reporter.clone();

        // Parameter extraction from metadata.
        let head_count = md_get("phi3.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("phi3.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("phi3.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("phi3.embedding_length")?.to_u32()? as usize;
        let max_seq_len = md_get("phi3.context_length")?.to_u32()? as usize;
        let head_dim = embedding_length / head_count;
        let i_size = md_get("phi3.feed_forward_length")?.to_u32()? as usize;
        let rope_dim = md_get("phi3.rope.dimension_count")?.to_u32()? as usize;
        let rms_eps = md_get("phi3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let (cos, sin) = precomput_freqs_cis(rope_dim, max_seq_len, 10_000., device, dtype)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let output_norm = rms_norm(ct.tensor(reader, "output_norm.weight", device)?, rms_eps)?;
        let output = QLinear::new(ct, reader, "output", device)?;

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let ffn_up = QLinear::new(ct, reader, &format!("{prefix}.ffn_up"), device)?;
            let ffn_down = QLinear::new(ct, reader, &format!("{prefix}.ffn_down"), device)?;
            let mlp = Mlp {
                ffn_up,
                ffn_down,
                i_size,
            };
            let attn_norm = rms_norm(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_eps,
            )?;
            let ffn_norm = rms_norm(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_eps,
            )?;
            layers.push(LayerWeights {
                attn_qkv: QLinear::new(ct, reader, &format!("{prefix}.attn_qkv"), device)?,
                attn_output: QLinear::new(ct, reader, &format!("{prefix}.attn_output"), device)?,
                attn_norm,
                ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                attn: PagedAttention::new(
                    head_count,
                    head_dim,
                    1. / ((head_dim as f32).sqrt()),
                    Some(head_count_kv),
                    None,
                    device.clone(),
                    None,
                )?,
                dtype,
            });
            reporter.write().unwrap().set_progress(layer_idx + 1);
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            cfg: GGUFPhi3::into_config(
                embedding_length,
                i_size,
                block_count,
                head_count,
                head_count_kv,
                rms_eps,
                max_seq_len,
                dtype,
                s_cfg,
            ),
            dtype,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = xs.dims2()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            super::get_attention_casual_mask(
                &self.device,
                self.dtype,
                b_sz,
                seq_len,
                input_positions,
                self.cfg.sliding_window,
            )
        };
        let mut xs = self.tok_embeddings.forward(xs)?;

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let residual = &xs;
                let ys = xs.apply(&layer.attn_norm)?;
                let ys = layer.forward_attn(
                    &ys,
                    mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
                let ys = (ys + residual)?;
                let residual = &ys;
                let ys = ys.apply(&layer.ffn_norm)?;
                let ys = layer.mlp.forward(&ys)?;
                xs = (ys + residual)?
            }
        } else {
            for layer in self.layers.iter() {
                let residual = &xs;
                let ys = xs.apply(&layer.attn_norm)?;
                let ys = layer.forward_attn(
                    &ys,
                    mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
                let ys = (ys + residual)?;
                let residual = &ys;
                let ys = ys.apply(&layer.ffn_norm)?;
                let ys = layer.mlp.forward(&ys)?;
                xs = (ys + residual)?
            }
        }
        let xs = xs
            .i((.., seq_len - 1, ..))?
            .contiguous()?
            .apply(&self.output_norm)?;
        self.output.forward(&xs)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
