use super::layers::quantized_var_builder::VarBuilder as QVarBuilder;
use super::{
    attention::QuantizedAttention, rotary_emb::ScalingRotaryEmbedding, Config, KvCacheDtype,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
#[cfg(feature = "nccl")]
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, Rc, VocabParallelLinear};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::InputMetadata;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use either::Either;
use parking_lot::RwLock;
use std::iter::zip;
use std::sync::Arc;

struct Mlp {
    ffn_gate_up: QMatMul,
    ffn_down: QMatMul,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    #[cfg(feature = "nccl")]
    dtype: DType,
}

impl Mlp {
    #[allow(unused_mut)]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.ffn_gate_up.forward(xs)?;
        let dim = w.dims().len() - 1;
        let gate = w.narrow(dim, 0, w.dim(dim)? / 2)?.contiguous()?;
        let up_states = w
            .narrow(dim, w.dim(dim)? / 2, w.dim(dim)? / 2)?
            .contiguous()?;
        let mut y = self
            .ffn_down
            .forward(&(candle_nn::ops::silu(&gate)? * up_states)?)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            y = all_reduce.apply(&y.to_dtype(self.dtype)?)?;
            y = y.to_dtype(DType::F32)?;
        }
        Ok(y)
    }
}

struct LayerWeights {
    self_attn: QuantizedAttention,
    attention_norm: QRmsNorm,
    post_ffw_norm: QRmsNorm,
    post_attention_norm: QRmsNorm,
    mlp: Mlp,
    ffn_norm: QRmsNorm,
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
    norm: QRmsNorm,
    output: VocabParallelLinear,
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
        _kv_cache_dtype: DType,
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
            eos_token_id: Some(super::TokenID(Either::Left(None))),
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
            isq_quant: None,
            kvcache_dtype: KvCacheDtype::Auto,
            extra_config_json: None,
            is_f16_mode: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_gguf(
        vb: &QVarBuilder,
        device: &Device,
        dtype: DType,
        kv_cache_dtype: DType,
        yarn_scaling_factor: Option<f64>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
        rank: usize,
        world_size: usize,
        #[allow(unused_variables)] comm: Rc<Comm>,
    ) -> Result<Self> {
        let metadata = vb.first_content_metadata();
        let md_get = |s: &str| match metadata.get(s) {
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

        let tok_embeddings = vb.get_no_shape("token_embd.weight")?;
        let vocab_size = tok_embeddings.shape().dims()[0];
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm =
            QRmsNorm::from_arc_qtensor(vb.get_no_shape("output_norm.weight")?, rms_norm_eps)?;
        let output_tensor_name = if vb.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output = VocabParallelLinear::load_from_gguf(
            vb,
            output_tensor_name,
            vocab_size,
            comm.clone(),
            dtype,
        )?;

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

        let mut cfg = GGUFGLM4::into_config(
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
            kv_cache_dtype,
        );
        cfg.apply_runtime_rope_overrides(yarn_scaling_factor);
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            DType::F32,
            &cfg,
            device,
            false,
        )?);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let prefix_vb = vb.pp(&prefix);
            let mlp = {
                let ffn_gate_up =
                    prefix_vb.get_sharded_no_shape("ffn_up.weight", 0, rank, world_size)?;
                let ffn_down =
                    prefix_vb.get_sharded_no_shape("ffn_down.weight", 1, rank, world_size)?;
                Mlp {
                    ffn_gate_up: QMatMul::from_arc(ffn_gate_up)?,
                    ffn_down: QMatMul::from_arc(ffn_down)?,
                    #[cfg(feature = "nccl")]
                    all_reduce: if world_size > 1 {
                        Some(AllReduce::new(comm.clone()))
                    } else {
                        None
                    },
                    #[cfg(feature = "nccl")]
                    dtype,
                }
            };

            let attention_norm = prefix_vb.get_no_shape("attn_norm.weight")?;
            let ffn_norm = prefix_vb.get_no_shape("ffn_norm.weight")?;
            let post_ffw_norm = prefix_vb.get_no_shape("post_ffw_norm.weight")?;
            let post_attention_norm = prefix_vb.get_no_shape("post_attention_norm.weight")?;
            let post_ffw_norm = QRmsNorm::from_arc_qtensor(post_ffw_norm, rms_norm_eps)?;
            let post_attention_norm =
                QRmsNorm::from_arc_qtensor(post_attention_norm, rms_norm_eps)?;
            let self_attn = QuantizedAttention::new(
                &cfg,
                vb,
                &prefix,
                device,
                dtype,
                rotary_emb.clone(),
                cfg.sliding_window,
                rank,
                world_size,
                comm.clone(),
            )?;

            layers.push(LayerWeights {
                self_attn,
                attention_norm: QRmsNorm::from_arc_qtensor(attention_norm, rms_norm_eps)?,
                post_ffw_norm,
                post_attention_norm,
                mlp,
                ffn_norm: QRmsNorm::from_arc_qtensor(ffn_norm, rms_norm_eps)?,
            });
            reporter.write().set_progress(layer_idx + 1);
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
        self.forward_inner(x, input_positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, true)
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
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
        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;

        if return_hidden {
            return Ok(xs);
        }
        self.output.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
