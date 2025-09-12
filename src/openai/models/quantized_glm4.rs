use super::{attention::QuantizedAttention, rotary_emb::ScalingRotaryEmbedding, Config};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::models::mask::get_attention_casual_mask;
use crate::InputMetadata;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Result, Tensor};
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
    self_attn: QuantizedAttention,
    attention_norm: RmsNorm,
    post_ffw_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.self_attn
            .forward(x, mask, input_positions, cache, input_metadata)
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
        rope_theta: f64,
        rms_eps: f64,
        max_seq_len: usize,
        partial_rotary_factor: Option<f32>,
    ) -> Config {
        Config {
            architectures: Some(vec!["glm4".to_string()]),
            hidden_size: embedding_length,
            head_dim: Some(head_dim),
            intermediate_size: i_size,
            vocab_size: 0,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: Some(head_count_kv),
            rms_norm_eps: rms_eps,
            rope_theta,
            rope_local_base_freq: None,
            bos_token_id: None,
            eos_token_id: super::TokenID(Either::Left(None)),
            max_seq_len,
            sliding_window: None,
            sliding_window_pattern: None,
            hidden_act: None,
            hidden_activation: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: Some(max_seq_len),
            original_max_position_embeddings: None,
            attention_bias: Some(false),
            partial_rotary_factor,
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: None,
            moe_config: None,
            quant: Some("gguf".to_string()),
        }
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let reporter = progress_reporter.clone();

        let arch = md_get("general.architecture")?.to_string()?;

        let head_count =
            md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
        let head_count_kv =
            md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;

        let head_dim = md_get(format!("{arch}.attention.key_length").as_str());
        let embedding_length =
            md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
        let head_dim = if head_dim.is_ok() {
            head_dim.unwrap().to_u32()? as usize
        } else {
            embedding_length / head_count
        };
        let context_length = md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
        let block_count = md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
        let rms_norm_eps =
            md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
        let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

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

        let rope_dim = md_get(format!("{arch}.rope.dimension_count").as_str());
        let partial_rotary_factor = if rope_dim.is_ok() {
            let rope_dim = rope_dim.unwrap().to_u32()? as usize;
            if rope_dim != head_dim {
                Some(rope_dim as f32 / head_dim as f32)
            } else {
                None
            }
        } else {
            None
        };

        let cfg = GGUFGLM4::into_config(
            embedding_length,
            head_dim,
            0,
            block_count,
            head_count,
            head_count_kv,
            rope_freq_base as f64,
            rms_norm_eps,
            context_length,
            partial_rotary_factor,
        );
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            DType::F32,
            &cfg,
            device,
            false,
        )?);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
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
            let self_attn = QuantizedAttention::new(
                &cfg,
                ct,
                reader,
                &prefix,
                device,
                dtype,
                rotary_emb.clone(),
                cfg.sliding_window,
            )?;

            layers.push(LayerWeights {
                self_attn,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                post_ffw_norm,
                post_attention_norm,
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
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
        let attention_mask = get_attention_casual_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );
        let mut xs = self.tok_embeddings.forward(x)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let x = xs;
                let residual = &x;
                let x = layer.attention_norm.forward(&x)?;
                let attn = layer.forward_attn(
                    &x,
                    attention_mask.as_ref(),
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
                xs = x
            }
        } else {
            for layer in self.layers.iter() {
                let x = xs;
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let attn = layer.forward_attn(
                    &x,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
                let attn = layer.attention_norm.forward(&attn)?;
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.post_attention_norm.forward(&x)?;
                let x = layer.mlp.forward(&x)?;
                let x = layer.post_ffw_norm.forward(&x)?;
                let x = (x + residual)?;
                xs = x
            }
        }
        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
