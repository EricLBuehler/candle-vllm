use super::rotary_emb::ScalingRotaryEmbedding;
use super::{attention::QuantizedAttention, Config, MoEConfig, QwenMoEConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::models::linear::Linear;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::InputMetadata;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use either::Either;
use parking_lot::RwLock;
use std::iter::zip;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

struct FusedMoe {
    gate: QMatMul,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl FusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let router_logits = self.gate.forward(&xs)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        //last dim size 128
        let indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut scores = routing_weights.gather(&indices, D::Minus1)?;

        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            let xs = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = self.gate_experts.indexed_moe_forward(&xs, &indices)?;
            let up = self.up_experts.indexed_moe_forward(&xs, &indices)?;
            let xs = self
                .down_experts
                .indexed_moe_forward(&(up * gate.apply(&self.act)?)?, &indices)?;
            xs
        };
        ys.broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((num_tokens, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

struct LayerWeights {
    self_attn: QuantizedAttention,
    attention_norm: RmsNorm,
    mlp: MoeOrMlp,
    shared_gate: Option<Linear>,
    shared_expert: Option<Mlp>,
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

pub struct GGUFQWenMoE {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl GGUFQWenMoE {
    pub fn into_config(
        arch: String,
        embedding_length: usize,
        head_dim: usize,
        i_size: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rms_eps: f64,
        rope_theta: f64,
        max_seq_len: usize,
        original_max_position_embeddings: Option<usize>,
        partial_rotary_factor: Option<f32>,
        moe_cfg: &QwenMoEConfig,
        kv_cache_dtype: DType,
    ) -> Config {
        Config {
            architectures: Some(vec![arch]),
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
            bos_token_id: Some(super::TokenID(Either::Left(Some(151644)))),
            eos_token_id: super::TokenID(Either::Left(Some(151645))),
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
            moe_config: Some(MoEConfig::QwenMoE(moe_cfg.clone())),
            quant: Some("gguf".to_string()),
            fp8_kvcache: Some(kv_cache_dtype == DType::U8),
        }
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
        kv_cache_dtype: DType,
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
        let expert_shared_feed_forward_length =
            md_get(format!("{arch}.expert_shared_feed_forward_length").as_str());
        let shared_expert_intermediate_size = match expert_shared_feed_forward_length {
            Ok(length) => {
                if length.to_u32()? > 0 {
                    Some(length.to_u32()? as usize)
                } else {
                    None
                }
            }
            _ => None,
        };

        let moe_cfg = QwenMoEConfig {
            moe_intermediate_size: md_get(format!("{arch}.expert_feed_forward_length").as_str())?
                .to_u32()? as usize,
            shared_expert_intermediate_size,
            num_experts: Some(md_get(format!("{arch}.expert_count").as_str())?.to_u32()? as usize),
            mlp_only_layers: Some(vec![]),
            decoder_sparse_step: Some(1),
            norm_topk_prob: shared_expert_intermediate_size.is_none(),
            num_experts_per_tok: md_get(format!("{arch}.expert_used_count").as_str())?.to_u32()?
                as usize,
        };

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
        let cfg = GGUFQWenMoE::into_config(
            arch.clone(),
            embedding_length,
            head_dim,
            0,
            block_count,
            head_count,
            head_count_kv,
            rms_norm_eps,
            rope_freq_base as f64,
            context_length,
            original_max_position_embeddings,
            partial_rotary_factor,
            &moe_cfg,
            kv_cache_dtype,
        );
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, &cfg, device, true)?);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let mlp = if !moe_cfg
                .mlp_only_layers
                .as_ref()
                .unwrap_or(&Vec::<usize>::new())
                .contains(&layer_idx)
                && (moe_cfg.num_experts.unwrap_or(0) > 0
                    && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap_or(1) == 0)
            {
                let gate = ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let gate_experts =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate_exps.weight"), device)?;
                let up_experts =
                    ct.tensor(reader, &format!("{prefix}.ffn_up_exps.weight"), device)?;
                let down_experts =
                    ct.tensor(reader, &format!("{prefix}.ffn_down_exps.weight"), device)?;
                let moe = FusedMoe {
                    gate: QMatMul::from_qtensor(gate)?,
                    gate_experts: QMatMul::from_qtensor(gate_experts)?,
                    up_experts: QMatMul::from_qtensor(up_experts)?,
                    down_experts: QMatMul::from_qtensor(down_experts)?,
                    act: candle_nn::Activation::Silu,
                    norm_topk_prob: moe_cfg.norm_topk_prob,
                    num_experts_per_tok: moe_cfg.num_experts_per_tok,
                };

                MoeOrMlp::FusedMoe(moe)
            } else {
                let mlp = {
                    let feed_forward_w1 =
                        ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                    Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    }
                };
                MoeOrMlp::Mlp(mlp)
            };

            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            //shared experts weights in Qwen2 MoE models
            let (shared_gate, shared_expert) =
                if let Some(_) = moe_cfg.shared_expert_intermediate_size {
                    let ws = ct
                        .tensor(
                            reader,
                            &format!("{prefix}.ffn_gate_inp_shexp.weight"),
                            device,
                        )?
                        .dequantize(device)?
                        .reshape((1, cfg.hidden_size))?; //weight must be 2d+

                    let shared_gate = Linear::new(ws, None);

                    let mlp = {
                        let feed_forward_w1 =
                            ct.tensor(reader, &format!("{prefix}.ffn_gate_shexp.weight"), device)?;
                        let feed_forward_w2 =
                            ct.tensor(reader, &format!("{prefix}.ffn_down_shexp.weight"), device)?;
                        let feed_forward_w3 =
                            ct.tensor(reader, &format!("{prefix}.ffn_up_shexp.weight"), device)?;
                        Mlp {
                            feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                            feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                            feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                        }
                    };

                    (Some(shared_gate), Some(mlp))
                } else {
                    (None, None)
                };

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
                mlp,
                shared_gate,
                shared_expert,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
            });
            reporter.write().set_progress(layer_idx + 1);
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            cfg,
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
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;

                //shared experts for Qwen2 MoE models
                let shared_output = match (&layer.shared_gate, &layer.shared_expert) {
                    (Some(shared_gate), Some(shared_expert)) => {
                        let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&x)?)?;
                        let shared_output = shared_expert.forward(&x)?;
                        Some(gate.broadcast_mul(&shared_output)?)
                    }
                    _ => None,
                };
                let x = layer.mlp.forward(&x)?;
                let x = if let Some(shared_output) = shared_output {
                    (residual + (x + shared_output)?)?
                } else {
                    (x + residual)?
                };
                xs = x
            }
        } else {
            for layer in self.layers.iter() {
                let x = xs;
                let residual = &x;
                let x = layer.attention_norm.forward(&x)?;
                let attn = layer.forward_attn(
                    &x,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                //shared experts for Qwen2 MoE models
                let shared_output = match (&layer.shared_gate, &layer.shared_expert) {
                    (Some(shared_gate), Some(shared_expert)) => {
                        let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&x)?)?;
                        let shared_output = shared_expert.forward(&x)?;
                        Some(gate.broadcast_mul(&shared_output)?)
                    }
                    _ => None,
                };
                let x = layer.mlp.forward(&x)?;
                let x = if let Some(shared_output) = shared_output {
                    (residual + (x + shared_output)?)?
                } else {
                    (x + residual)?
                };
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
