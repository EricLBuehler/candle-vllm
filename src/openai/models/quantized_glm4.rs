use super::Config;
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::models::glm4::RotaryEmbedding;
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use crate::SpecificConfig;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use either::Either;
use std::iter::zip;
use std::sync::{Arc, RwLock};
#[derive(Debug, Clone)]
struct Mlp {
    ffn_gate_up: QMatMul,
    ffn_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.ffn_gate_up.forward(xs)?;
        let dim = w.dims().len() - 1;
        let gate = w.narrow(dim, 0, w.dim(dim)? / 2)?.contiguous()?;
        let up_states = w
            .narrow(dim, w.dim(dim)? / 2, w.dim(dim)? / 2)?
            .contiguous()?;
        self.ffn_down
            .forward(&(candle_nn::ops::silu(&gate)? * up_states)?)
    }
}

struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Option<Tensor>,
    attention_bk: Option<Tensor>,
    attention_bv: Option<Tensor>,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    post_ffw_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    rotary_emb: Arc<RotaryEmbedding>,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    attn: PagedAttention,
    dtype: DType,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = if self.attention_bq.is_some() {
            q.broadcast_add(self.attention_bq.as_ref().unwrap())?
        } else {
            q
        };

        let k = if self.attention_bk.is_some() {
            k.broadcast_add(self.attention_bk.as_ref().unwrap())?
        } else {
            k
        };

        let v = if self.attention_bv.is_some() {
            v.broadcast_add(self.attention_bv.as_ref().unwrap())?
        } else {
            v
        };

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

        let q = self
            .rotary_emb
            .apply_rotary_emb(&q.to_dtype(DType::F32)?, input_positions)?;
        let k = self
            .rotary_emb
            .apply_rotary_emb(&k.to_dtype(DType::F32)?, input_positions)?;
        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

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

        let y = self.attention_wo.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

pub struct GGUFGLM4 {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl GGUFGLM4 {
    pub fn into_config(
        embedding_length: usize,
        head_dim: usize,
        i_size: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rope_theta: f32,
        rms_eps: f64,
        max_seq_len: usize,
        kv_cache_dtype: DType,
        s_cfg: SpecificConfig,
    ) -> Config {
        Config {
            hidden_size: embedding_length,
            head_dim: Some(head_dim),
            intermediate_size: i_size,
            vocab_size: 0,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: head_count_kv,
            rms_norm_eps: rms_eps,
            rope_theta: rope_theta as f64,
            rope_local_base_freq: None,
            use_flash_attn: false,
            bos_token_id: super::TokenID(Either::Left(None)),
            eos_token_id: super::TokenID(Either::Left(None)),
            max_seq_len,
            sliding_window: None,
            sliding_window_pattern: None,
            hidden_act: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            original_max_position_embeddings: Some(max_seq_len),
            attention_bias: false,
            partial_rotary_factor: Some(0.5),
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
        Ok(md_get("glm4.block_count")?.to_u32()? as usize)
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

        let head_count = md_get("glm4.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("glm4.attention.head_count_kv")?.to_u32()? as usize;

        let head_dim = md_get("glm4.attention.key_length");
        let head_dim = if head_dim.is_ok() {
            Some(head_dim.unwrap().to_u32()? as usize)
        } else {
            None
        };
        let embedding_length = md_get("glm4.embedding_length")?.to_u32()? as usize;
        let context_length = md_get("glm4.context_length")?.to_u32()? as usize;
        let block_count = md_get("glm4.block_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("glm4.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("glm4.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let head_dim = head_dim.unwrap_or(embedding_length / head_count);
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => {
                // use tie_word_embeddings
                QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?
            }
        };

        let cfg = GGUFGLM4::into_config(
            embedding_length,
            head_dim,
            0,
            block_count,
            head_count,
            head_count_kv,
            rope_freq_base,
            rms_norm_eps,
            context_length,
            dtype,
            s_cfg,
        );
        let rotary_emb = Arc::new(RotaryEmbedding::new(&cfg, dtype, device)?);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;

            let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device);
            let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device);
            let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device);

            let attention_bq = if attention_bq.is_ok() {
                Some(
                    attention_bq
                        .unwrap()
                        .dequantize(device)?
                        .to_dtype(DType::F32)?,
                )
            } else {
                None
            };

            let attention_bk = if attention_bk.is_ok() {
                Some(
                    attention_bk
                        .unwrap()
                        .dequantize(device)?
                        .to_dtype(DType::F32)?,
                )
            } else {
                None
            };

            let attention_bv = if attention_bv.is_ok() {
                Some(
                    attention_bv
                        .unwrap()
                        .dequantize(device)?
                        .to_dtype(DType::F32)?,
                )
            } else {
                None
            };

            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp = {
                let ffn_gate_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                let ffn_down = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                Mlp {
                    ffn_gate_up: QMatMul::from_qtensor(ffn_gate_up)?,
                    ffn_down: QMatMul::from_qtensor(ffn_down)?,
                }
            };

            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let post_ffw_norm =
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?;
            let post_attention_norm = ct.tensor(
                reader,
                &format!("{prefix}.post_attention_norm.weight"),
                device,
            )?;
            let post_ffw_norm = RmsNorm::from_qtensor(post_ffw_norm, rms_norm_eps)?;
            let post_attention_norm = RmsNorm::from_qtensor(post_attention_norm, rms_norm_eps)?;

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq,
                attention_bk,
                attention_bv,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                post_ffw_norm,
                post_attention_norm,
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary_emb: rotary_emb.clone(),
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
            norm,
            output,
            cfg: cfg.clone(),
            dtype,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        assert!(
            seq_len < self.cfg.max_seq_len,
            "Input token length exceed maximum context allowed for this model."
        );
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
        let mut layer_in = self.tok_embeddings.forward(x)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let x = layer_in;
                let residual = &x;
                let x = layer.attention_norm.forward(&x)?;
                let attn = layer.forward_attn(
                    &x,
                    mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
                let attn = layer.post_attention_norm.forward(&attn)?;
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let x = layer.mlp.forward(&x)?;
                let x = layer.post_ffw_norm.forward(&x)?;
                let x = (x + residual)?;
                layer_in = x
            }
        } else {
            for layer in self.layers.iter() {
                let x = layer_in;
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let attn =
                    layer.forward_attn(&x, mask.as_ref(), input_positions, None, input_metadata)?;
                let attn = layer.attention_norm.forward(&attn)?;
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.post_attention_norm.forward(&x)?;
                let x = layer.mlp.forward(&x)?;
                let x = layer.post_ffw_norm.forward(&x)?;
                let x = (x + residual)?;
                layer_in = x
            }
        }
        let x = layer_in
            .i((.., seq_len - 1, ..))?
            .contiguous()?
            .apply(&self.norm)?;
        self.output.forward(&x)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
