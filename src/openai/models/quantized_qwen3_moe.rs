use super::layers::quantized_var_builder::VarBuilder as QVarBuilder;
use super::rotary_emb::ScalingRotaryEmbedding;
use super::{attention::QuantizedAttention, Config, KvCacheDtype, MoEConfig, QwenMoEConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
#[cfg(feature = "nccl")]
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, Rc, VocabParallelLinear};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::linear::Linear;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::InputMetadata;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use either::Either;
use parking_lot::RwLock;
use std::iter::zip;
use std::sync::Arc;

struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    #[cfg(feature = "nccl")]
    dtype: DType,
}

impl Mlp {
    #[allow(unused_mut)]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        let mut y = self
            .feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            y = all_reduce.apply(&y.to_dtype(self.dtype)?)?;
            y = y.to_dtype(DType::F32)?;
        }
        Ok(y)
    }
}

struct FusedMoe {
    gate: QMatMul,
    gate_experts: Arc<QTensor>,
    up_experts: Arc<QTensor>,
    down_experts: Arc<QTensor>,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    dtype: DType,
    #[cfg(not(feature = "nccl"))]
    #[allow(dead_code)]
    world_size: usize,
    #[cfg(feature = "nccl")]
    world_size: usize,
}

impl FusedMoe {
    #[allow(unused_mut, unused_variables)]
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) =
            attention_rs::topk::topk_softmax(&router_logits, self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        if let Some(factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * factor)?;
        }

        let flat = topk_ids.flatten_all()?;
        let (expert_ids, sorted_token_ids) = flat.sort_last_dim(true)?;

        let ys = {
            let gate = attention_rs::moe::moe_gemm_gguf(
                &xs,
                &self.gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let up = attention_rs::moe::moe_gemm_gguf(
                &xs,
                &self.up_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let down_inputs = (up * gate.apply(&self.act)?)?;
            attention_rs::moe::moe_gemm_gguf(
                &down_inputs,
                &self.down_experts,
                &Some(topk_weights),
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?
        };
        let mut ys = ys.reshape((num_tokens, (), hidden_dim))?.sum(D::Minus2)?;
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        #[cfg(feature = "nccl")]
        if self.world_size > 1 {
            if let Some(all_reduce) = &self.all_reduce {
                ys = all_reduce.apply(&ys)?;
            }
        }
        ys.to_dtype(original_dtype)
    }
}

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs, is_prefill),
        }
    }
}

struct LayerWeights {
    self_attn: QuantizedAttention,
    attention_norm: QRmsNorm,
    mlp: MoeOrMlp,
    shared_gate: Option<Linear>,
    shared_expert: Option<Mlp>,
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

pub struct GGUFQWenMoE {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: VocabParallelLinear,
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
        _kv_cache_dtype: DType,
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
            eos_token_id: Some(super::TokenID(Either::Left(Some(151645)))),
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
            routed_scaling_factor: None,
            first_k_dense_replace: None,
            n_shared_experts: None,
        };

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
        let mut cfg = GGUFQWenMoE::into_config(
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
        cfg.apply_runtime_rope_overrides(yarn_scaling_factor);
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, &cfg, device, true)?);

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let prefix_vb = vb.pp(&prefix);
            let mlp = if !moe_cfg
                .mlp_only_layers
                .as_ref()
                .unwrap_or(&Vec::<usize>::new())
                .contains(&layer_idx)
                && (moe_cfg.num_experts.unwrap_or(0) > 0
                    && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap_or(1) == 0)
            {
                let gate = prefix_vb.get_no_shape("ffn_gate_inp.weight")?;
                let gate_experts =
                    prefix_vb.get_sharded_no_shape("ffn_gate_exps.weight", 1, rank, world_size)?;
                let up_experts =
                    prefix_vb.get_sharded_no_shape("ffn_up_exps.weight", 1, rank, world_size)?;
                let down_experts =
                    prefix_vb.get_sharded_no_shape("ffn_down_exps.weight", 2, rank, world_size)?;
                let moe = FusedMoe {
                    gate: QMatMul::from_arc(gate)?,
                    gate_experts,
                    up_experts,
                    down_experts,
                    act: candle_nn::Activation::Silu,
                    norm_topk_prob: moe_cfg.norm_topk_prob,
                    routed_scaling_factor: moe_cfg.routed_scaling_factor,
                    num_experts_per_tok: moe_cfg.num_experts_per_tok,
                    #[cfg(feature = "nccl")]
                    all_reduce: if world_size > 1 {
                        Some(AllReduce::new(comm.clone()))
                    } else {
                        None
                    },
                    dtype,
                    world_size,
                };

                MoeOrMlp::FusedMoe(moe)
            } else {
                let mlp = {
                    let feed_forward_w1 =
                        prefix_vb.get_sharded_no_shape("ffn_gate.weight", 0, rank, world_size)?;
                    let feed_forward_w2 =
                        prefix_vb.get_sharded_no_shape("ffn_down.weight", 1, rank, world_size)?;
                    let feed_forward_w3 =
                        prefix_vb.get_sharded_no_shape("ffn_up.weight", 0, rank, world_size)?;
                    Mlp {
                        feed_forward_w1: QMatMul::from_arc(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_arc(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_arc(feed_forward_w3)?,
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
                MoeOrMlp::Mlp(mlp)
            };

            let attention_norm = prefix_vb.get_no_shape("attn_norm.weight")?;
            let ffn_norm = prefix_vb.get_no_shape("ffn_norm.weight")?;

            let (shared_gate, shared_expert) =
                if let Some(_) = moe_cfg.shared_expert_intermediate_size {
                    let ws = prefix_vb
                        .get_no_shape("ffn_gate_inp_shexp.weight")?
                        .dequantize(device)?
                        .reshape((1, cfg.hidden_size))?;

                    let shared_gate = Linear::new(ws, None);

                    let mlp = {
                        let feed_forward_w1 = prefix_vb.get_sharded_no_shape(
                            "ffn_gate_shexp.weight",
                            0,
                            rank,
                            world_size,
                        )?;
                        let feed_forward_w2 = prefix_vb.get_sharded_no_shape(
                            "ffn_down_shexp.weight",
                            1,
                            rank,
                            world_size,
                        )?;
                        let feed_forward_w3 = prefix_vb.get_sharded_no_shape(
                            "ffn_up_shexp.weight",
                            0,
                            rank,
                            world_size,
                        )?;
                        Mlp {
                            feed_forward_w1: QMatMul::from_arc(feed_forward_w1)?,
                            feed_forward_w2: QMatMul::from_arc(feed_forward_w2)?,
                            feed_forward_w3: QMatMul::from_arc(feed_forward_w3)?,
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

                    (Some(shared_gate), Some(mlp))
                } else {
                    (None, None)
                };

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
                mlp,
                shared_gate,
                shared_expert,
                ffn_norm: QRmsNorm::from_arc_qtensor(ffn_norm, rms_norm_eps)?,
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
                let x = layer.mlp.forward(&x, input_metadata.is_prefill)?;
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
                let x = layer.mlp.forward(&x, input_metadata.is_prefill)?;
                let x = if let Some(shared_output) = shared_output {
                    (residual + (x + shared_output)?)?
                } else {
                    (x + residual)?
                };
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
