use super::layers::quantized_var_builder::VarBuilder as QVarBuilder;
use super::rotary_emb::ScalingRotaryEmbedding;
use super::Config;
use super::KvCacheDtype;
use crate::backend::progress::{ProgressLike, ProgressReporter};
#[cfg(feature = "nccl")]
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, Rc, VocabParallelLinear};
use crate::openai::models::mask::get_attention_causal_mask;
use crate::{InputMetadata, PagedAttention};
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, RmsNorm};
use either::Either;
use parking_lot::RwLock;
use std::iter::zip;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct QLinear {
    inner: candle_core::quantized::QMatMul,
}

impl QLinear {
    #[allow(dead_code)]
    fn new(vb: &QVarBuilder, name: &str) -> Result<Self> {
        let w = vb.get_no_shape(&format!("{name}.weight"))?;
        let inner = candle_core::quantized::QMatMul::from_arc(w)?;
        Ok(Self { inner })
    }

    fn new_sharded(
        vb: &QVarBuilder,
        name: &str,
        _device: &Device,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        let w = vb.get_sharded_no_shape(&format!("{name}.weight"), dim, rank, world_size)?;
        let inner = candle_core::quantized::QMatMul::from_arc(w)?;
        Ok(Self { inner })
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

struct Mlp {
    ffn_up: QLinear,
    ffn_down: QLinear,
    i_size: usize,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    #[cfg(feature = "nccl")]
    dtype: DType,
}

impl Mlp {
    #[allow(unused_mut)]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = xs.apply(&self.ffn_up)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let mut y = (up_states * gate.silu()?)?;
        y = y.apply(&self.ffn_down)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            y = all_reduce.apply(&y.to_dtype(self.dtype)?)?;
            y = y.to_dtype(DType::F32)?;
        }
        Ok(y)
    }
}

fn rms_norm(w: Arc<QTensor>, eps: f64) -> Result<RmsNorm> {
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
    attn: PagedAttention,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    dtype: DType,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
}

impl LayerWeights {
    #[allow(unused_mut)]
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let (seq_len, _) = x.dims2()?;
        let qkv = self.attn_qkv.forward(x)?;

        let query_pos = self.n_head * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?.to_dtype(DType::F32)?;
        let k = qkv
            .narrow(D::Minus1, query_pos, self.n_kv_head * self.head_dim)?
            .to_dtype(DType::F32)?;
        let v = qkv
            .narrow(
                D::Minus1,
                query_pos + self.n_kv_head * self.head_dim,
                self.n_kv_head * self.head_dim,
            )?
            .to_dtype(self.dtype)?;

        let q = q
            .reshape((1, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((1, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, input_positions)?;
        let (q, k) = (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?);
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
            .reshape((seq_len, ()))?;

        let mut y = self.attn_output.forward(&y.to_dtype(x.dtype())?)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            y = all_reduce.apply(&y.to_dtype(self.dtype)?)?;
            y = y.to_dtype(DType::F32)?;
        }
        Ok(y)
    }
}

pub struct GGUFPhi3 {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    output_norm: RmsNorm,
    output: VocabParallelLinear,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl GGUFPhi3 {
    pub fn into_config(
        embedding_length: usize,
        i_size: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rms_eps: f64,
        rope_theta: f64,
        max_seq_len: usize,
        original_max_position_embeddings: Option<usize>,
        partial_rotary_factor: Option<f32>,
        _kv_cache_dtype: DType,
    ) -> Config {
        Config {
            architectures: Some(vec!["phi3".to_string()]),
            hidden_size: embedding_length,
            head_dim: Some(embedding_length / head_count),
            intermediate_size: i_size,
            vocab_size: 0,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: Some(head_count_kv),
            rms_norm_eps: rms_eps,
            rope_theta,
            rope_local_base_freq: None,
            bos_token_id: Some(super::TokenID(Either::Left(Some(1)))),
            eos_token_id: Some(super::TokenID(Either::Left(Some(2)))),
            max_seq_len,
            sliding_window: None,
            sliding_window_pattern: None,
            hidden_act: None,
            hidden_activation: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: Some(max_seq_len),
            original_max_position_embeddings,
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

        // Parameter extraction from metadata.
        let head_count =
            md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
        let head_count_kv =
            md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;
        let block_count = md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
        let embedding_length =
            md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
        let max_seq_len = md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
        let head_dim = md_get(format!("{arch}.attention.key_length").as_str());
        let head_dim = if head_dim.is_ok() {
            head_dim.unwrap().to_u32()? as usize
        } else {
            embedding_length / head_count
        };
        let i_size = md_get(format!("{arch}.feed_forward_length").as_str())?.to_u32()? as usize;
        let rms_eps =
            md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
        let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let tok_embeddings = vb.get_no_shape("token_embd.weight")?;
        let vocab_size = tok_embeddings.shape().dims()[0];
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let output_norm = rms_norm(vb.get_no_shape("output_norm.weight")?, rms_eps)?;
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

        let original_max_position_embeddings =
            md_get(format!("{arch}.rope.scaling.original_context_length").as_str());
        let original_max_position_embeddings = if original_max_position_embeddings.is_ok() {
            Some(original_max_position_embeddings.unwrap().to_u32()? as usize)
        } else {
            None
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

        let mut cfg = GGUFPhi3::into_config(
            embedding_length,
            i_size,
            block_count,
            head_count,
            head_count_kv,
            rms_eps,
            rope_freq_base as f64,
            max_seq_len,
            original_max_position_embeddings,
            partial_rotary_factor,
            kv_cache_dtype,
        );
        cfg.apply_runtime_rope_overrides(yarn_scaling_factor);
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, &cfg, device, true)?);

        let n_head = head_count / world_size;
        let n_kv_head = if head_count_kv >= world_size {
            head_count_kv / world_size
        } else {
            head_count_kv
        };
        let local_i_size = i_size / world_size;

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let prefix_vb = vb.pp(&prefix);
            let ffn_up = QLinear::new_sharded(&prefix_vb, "ffn_up", device, 0, rank, world_size)?;
            let ffn_down =
                QLinear::new_sharded(&prefix_vb, "ffn_down", device, 1, rank, world_size)?;
            let mlp = Mlp {
                ffn_up,
                ffn_down,
                i_size: local_i_size,
                #[cfg(feature = "nccl")]
                all_reduce: if world_size > 1 {
                    Some(AllReduce::new(comm.clone()))
                } else {
                    None
                },
                #[cfg(feature = "nccl")]
                dtype,
            };
            let attn_norm = rms_norm(prefix_vb.get_no_shape("attn_norm.weight")?, rms_eps)?;
            let ffn_norm = rms_norm(prefix_vb.get_no_shape("ffn_norm.weight")?, rms_eps)?;
            layers.push(LayerWeights {
                attn_qkv: QLinear::new_sharded(
                    &prefix_vb, "attn_qkv", device, 0, rank, world_size,
                )?,
                attn_output: QLinear::new_sharded(
                    &prefix_vb,
                    "attn_output",
                    device,
                    1,
                    rank,
                    world_size,
                )?,
                attn_norm,
                ffn_norm,
                mlp,
                n_head,
                n_kv_head,
                head_dim,
                attn: PagedAttention::new(
                    n_head,
                    head_dim,
                    1. / ((head_dim as f32).sqrt()),
                    Some(n_kv_head),
                    None,
                    device.clone(),
                    None,
                    cfg.kvcache_dtype.is_fp8_keys(),
                )?,
                rotary_emb: rotary_emb.clone(),
                dtype,
                #[cfg(feature = "nccl")]
                all_reduce: if world_size > 1 {
                    Some(AllReduce::new(comm.clone()))
                } else {
                    None
                },
            });
            reporter.write().set_progress(layer_idx + 1);
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            output_norm,
            output,
            cfg,
            dtype,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(xs, input_positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        xs: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(xs, input_positions, kv_caches, input_metadata, true)
    }

    fn forward_inner(
        &self,
        xs: &Tensor,
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
        let mut xs = self.tok_embeddings.forward(xs)?;

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let residual = &xs;
                let ys = xs.apply(&layer.attn_norm)?;
                let ys = layer.forward_attn(
                    &ys,
                    attention_mask.as_ref(),
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
                    attention_mask.as_ref(),
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
        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.output_norm.forward(&xs)?;

        if return_hidden {
            return Ok(xs);
        }
        self.output.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
