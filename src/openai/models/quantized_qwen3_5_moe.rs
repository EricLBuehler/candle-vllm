use super::layers::quantized_var_builder::VarBuilder as QVarBuilder;
use super::quantized_qwen3_5::{parse_gguf_hybrid_config, QuantizedGatedDeltaNet};
use super::rotary_emb::ScalingRotaryEmbedding;
use super::{attention::QuantizedAttention, Config, KvCacheDtype, MoEConfig, QwenMoEConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
#[cfg(feature = "nccl")]
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, Rc, VocabParallelLinear};
use crate::openai::models::layers::moe::sort_expert_assignments;
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::linear::Linear;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::quantized_qwen3_5::build_extra_config_json;
use crate::openai::models::utils::{resolve_input_seqlens, resolve_mamba_seq_slots};
use crate::{InputMetadata, InputMetadataExt};
use attention_rs::mamba_cache::MambaCache;
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use either::Either;
use parking_lot::{RwLock, RwLockWriteGuard};
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
    e_score_correction_bias: Option<Tensor>,
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

        let mut router_logits = self.gate.forward(&xs)?;

        if let Some(bias) = &self.e_score_correction_bias {
            router_logits = router_logits.broadcast_add(&bias.to_dtype(DType::F32)?)?;
        }

        let (mut topk_weights, topk_ids) =
            attention_rs::topk::topk_softmax(&router_logits, self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        if let Some(factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * factor)?;
        }

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

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

fn try_load_e_score_correction_bias(
    vb: &QVarBuilder,
    prefix: &str,
    _num_experts: usize,
    device: &Device,
) -> Option<Tensor> {
    let prefix_vb = vb.pp(prefix);
    prefix_vb
        .get_no_shape("ffn_gate_inp.e_score_correction_bias")
        .ok()
        .and_then(|qt| qt.dequantize(device).ok())
        .or_else(|| {
            prefix_vb
                .get_no_shape("e_score_correction_bias")
                .ok()
                .and_then(|qt| qt.dequantize(device).ok())
        })
        .or_else(|| {
            prefix_vb
                .get_no_shape("exp_probs_b.bias")
                .ok()
                .and_then(|qt| qt.dequantize(device).ok())
        })
        .map(|t| {
            let t = t.to_dtype(DType::F32).unwrap_or_else(|_| t.clone());
            t.flatten_all().unwrap_or(t)
        })
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

enum AttnType {
    FullAttention(QuantizedAttention),
    LinearAttention(QuantizedGatedDeltaNet),
}

struct LayerWeights {
    attn: AttnType,
    attention_norm: QRmsNorm,
    mlp: MoeOrMlp,
    shared_gate: Option<Linear>,
    shared_expert: Option<Mlp>,
    ffn_norm: QRmsNorm,
}

impl LayerWeights {
    #[allow(dead_code)]
    fn is_full_attention(&self) -> bool {
        matches!(self.attn, AttnType::FullAttention(_))
    }
}

pub struct GGUFQWen3_5MoE {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: VocabParallelLinear,
    mamba_cache: RwLock<MambaCache>,
    cfg: Config,
    dtype: DType,
    device: Device,
}

impl GGUFQWen3_5MoE {
    pub fn into_config(
        arch: String,
        embedding_length: usize,
        head_dim: usize,
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
        extra_config_json: Option<String>,
    ) -> Config {
        Config {
            architectures: Some(vec![arch]),
            hidden_size: embedding_length,
            head_dim: Some(head_dim),
            intermediate_size: 0,
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
            extra_config_json,
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

        let expert_weights_norm = md_get(format!("{arch}.expert_weights_norm").as_str())
            .ok()
            .and_then(|v| v.to_bool().ok());
        let expert_weights_scale = md_get(format!("{arch}.expert_weights_scale").as_str())
            .ok()
            .and_then(|v| v.to_f64().ok());

        let moe_cfg = QwenMoEConfig {
            moe_intermediate_size: md_get(format!("{arch}.expert_feed_forward_length").as_str())?
                .to_u32()? as usize,
            shared_expert_intermediate_size,
            num_experts: Some(md_get(format!("{arch}.expert_count").as_str())?.to_u32()? as usize),
            mlp_only_layers: Some(vec![]),
            decoder_sparse_step: Some(1),
            norm_topk_prob: expert_weights_norm.unwrap_or(true),
            num_experts_per_tok: md_get(format!("{arch}.expert_used_count").as_str())?.to_u32()?
                as usize,
            routed_scaling_factor: expert_weights_scale,
            first_k_dense_replace: None,
            n_shared_experts: None,
            n_group: None,
            topk_group: None,
            scoring_func: None,
            topk_method: None,
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

        let hybrid = parse_gguf_hybrid_config(&metadata, &arch, block_count);
        let extra_config_json = build_extra_config_json(&hybrid, &arch);

        let mut cfg = GGUFQWen3_5MoE::into_config(
            arch.clone(),
            embedding_length,
            head_dim,
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
            Some(extra_config_json),
        );
        cfg.apply_runtime_rope_overrides(yarn_scaling_factor);
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, &cfg, device, true)?);

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

        let layer_types = &hybrid.layer_types;
        let mut layers = Vec::with_capacity(block_count);
        let mut gdn_layer_idx = 0usize;

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let layer_type = layer_types
                .get(layer_idx)
                .map(String::as_str)
                .unwrap_or("full_attention");

            let attn = if layer_type == "full_attention" {
                AttnType::FullAttention(QuantizedAttention::new(
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
                )?)
            } else {
                let cur_gdn_idx = gdn_layer_idx;
                gdn_layer_idx += 1;
                AttnType::LinearAttention(QuantizedGatedDeltaNet::new(
                    vb,
                    &prefix,
                    device,
                    &hybrid,
                    cur_gdn_idx,
                    rms_norm_eps,
                    rank,
                    world_size,
                )?)
            };

            let mlp = if !moe_cfg
                .mlp_only_layers
                .as_ref()
                .unwrap_or(&Vec::<usize>::new())
                .contains(&layer_idx)
                && (moe_cfg.num_experts.unwrap_or(0) > 0
                    && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap_or(1) == 0)
            {
                let prefix_vb = vb.pp(&prefix);
                let gate = prefix_vb.get_no_shape("ffn_gate_inp.weight")?;
                let gate_experts =
                    prefix_vb.get_sharded_no_shape("ffn_gate_exps.weight", 1, rank, world_size)?;
                let up_experts =
                    prefix_vb.get_sharded_no_shape("ffn_up_exps.weight", 1, rank, world_size)?;
                let down_experts =
                    prefix_vb.get_sharded_no_shape("ffn_down_exps.weight", 2, rank, world_size)?;
                let bias = try_load_e_score_correction_bias(
                    vb,
                    &prefix,
                    moe_cfg.num_experts.unwrap_or(0),
                    device,
                );
                let moe = FusedMoe {
                    gate: QMatMul::from_arc(gate)?,
                    gate_experts,
                    up_experts,
                    down_experts,
                    act: candle_nn::Activation::Silu,
                    norm_topk_prob: moe_cfg.norm_topk_prob,
                    routed_scaling_factor: moe_cfg.routed_scaling_factor,
                    num_experts_per_tok: moe_cfg.num_experts_per_tok,
                    e_score_correction_bias: bias,
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
                    let prefix_vb = vb.pp(&prefix);
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

            let prefix_vb = vb.pp(&prefix);
            let attention_norm = prefix_vb.get_no_shape("attn_norm.weight")?;
            let ffn_norm = prefix_vb.get_no_shape("post_attention_norm.weight")?;

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

            layers.push(LayerWeights {
                attn,
                attention_norm: QRmsNorm::from_arc_qtensor(attention_norm, rms_norm_eps)?,
                mlp,
                shared_gate,
                shared_expert,
                ffn_norm: QRmsNorm::from_arc_qtensor(ffn_norm, rms_norm_eps)?,
            });
            reporter.write().set_progress(layer_idx + 1);
        }

        let num_gdn_layers = gdn_layer_idx;
        let num_v_heads = hybrid.num_v_heads;
        let num_k_heads = hybrid.num_k_heads;
        let d_conv = num_k_heads * hybrid.key_head_dim * 2 + num_v_heads * hybrid.value_head_dim;

        let mamba_cache = if num_gdn_layers > 0 {
            MambaCache::new(
                num_gdn_layers,
                1,
                d_conv,
                hybrid.conv_kernel_size,
                num_v_heads,
                hybrid.key_head_dim,
                hybrid.value_head_dim,
                DType::F32,
                DType::F32,
                device,
            )?
        } else {
            MambaCache::new(0, 1, 1, 2, 1, 1, 1, DType::F32, DType::F32, device)?
        };

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            mamba_cache: RwLock::new(mamba_cache),
            cfg,
            dtype,
            device: device.clone(),
        })
    }

    pub fn embed_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(input_ids)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, false, false)
    }

    pub fn forward_embedding(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, true, false)
    }

    pub fn forward_with_deepstack(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embedded_inputs: bool,
        _visual_pos_masks: &Option<Tensor>,
        _deepstack_visual_embeds: &Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        self.forward_inner(
            x,
            input_positions,
            kv_caches,
            input_metadata,
            false,
            embedded_inputs,
        )
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
        embedded_inputs: bool,
    ) -> Result<Tensor> {
        let seqlens = resolve_input_seqlens(input_metadata)?;
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );
        let mut xs = if embedded_inputs {
            x.clone()
        } else {
            self.tok_embeddings.forward(x)?
        };

        let mut mamba_cache = self.mamba_cache.write();
        let seq_slots = resolve_mamba_seq_slots(
            "Qwen3.5 MoE GGUF",
            &self.device,
            input_metadata,
            xs.dim(0)?,
            &mut mamba_cache,
        )?;

        let mut kv_cache_idx = 0usize;
        for layer in self.layers.iter() {
            let residual = &xs;
            let x = layer.attention_norm.forward(&xs)?;
            let x = match &layer.attn {
                AttnType::FullAttention(attn) => {
                    let cache = if let Some(kv_caches) = kv_caches {
                        let c = &kv_caches[kv_cache_idx];
                        kv_cache_idx += 1;
                        Some((&c.0, &c.1))
                    } else {
                        None
                    };
                    attn.forward(
                        &x,
                        attention_mask.as_ref(),
                        input_positions,
                        cache,
                        input_metadata,
                    )?
                }
                AttnType::LinearAttention(gdn) => {
                    gdn.forward(&x, &mut mamba_cache, input_metadata, &seq_slots)?
                }
            };
            let x = (x + residual)?;

            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;

            let shared_output = match (&layer.shared_gate, &layer.shared_expert) {
                (Some(shared_gate), Some(shared_expert)) => {
                    let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&x)?)?;
                    let shared_output = shared_expert.forward(&x)?;
                    Some(gate.broadcast_mul(&shared_output)?)
                }
                _ => None,
            };
            let x = layer.mlp.forward(&x, input_metadata.moe_is_prefill())?;
            xs = if let Some(shared_output) = shared_output {
                (residual + (x + shared_output)?)?
            } else {
                (x + residual)?
            };
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

    pub fn release_sequence_state(&self, sequence_id: usize) {
        self.mamba_cache.write().free_slot(sequence_id);
    }

    pub fn ensure_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .ensure_slots_for_sequences(sequence_ids)
    }

    pub fn get_mamba_slots_for_sequences(&self, sequence_ids: &[usize]) -> Result<Vec<usize>> {
        self.mamba_cache
            .write()
            .get_slots_for_sequences(sequence_ids)
    }

    pub fn has_mamba_slot_for_sequence(&self, sequence_id: usize) -> bool {
        self.mamba_cache.read().get_slot(sequence_id).is_some()
    }

    pub fn lock_mamba_cache_for_graph(&self) -> RwLockWriteGuard<'_, MambaCache> {
        self.mamba_cache.write()
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        self.mamba_cache.write().reserve_capacity(max_num_seqs)
    }

    pub fn set_mamba_prefix_cache_capacity(&self, capacity: usize) {
        self.mamba_cache.write().set_prefix_cache_capacity(capacity);
    }

    pub fn capture_mamba_prefix_state(
        &self,
        seq_id: usize,
        hash: u64,
        preserve: bool,
    ) -> Result<bool> {
        self.mamba_cache
            .write()
            .capture_prefix_state(seq_id, hash, preserve)
    }

    pub fn has_mamba_prefix_state(&self, hash: u64) -> bool {
        self.mamba_cache.write().has_prefix_state(hash)
    }

    pub fn restore_mamba_prefix_state(&self, seq_id: usize, hash: u64) -> Result<bool> {
        self.mamba_cache.write().restore_prefix_state(seq_id, hash)
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        self.mamba_cache.write().reset_all()
    }
}
