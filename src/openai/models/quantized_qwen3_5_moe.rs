use super::quantized_qwen3_5::{parse_gguf_hybrid_config, QuantizedGatedDeltaNet};
use super::rotary_emb::ScalingRotaryEmbedding;
use super::{attention::QuantizedAttention, Config, MoEConfig, QwenMoEConfig};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::linear::Linear;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::quantized_qwen3_5::build_extra_config_json;
use crate::openai::models::utils::{resolve_input_seqlens, resolve_mamba_seq_slots};
use crate::InputMetadata;
use attention_rs::mamba_cache::MambaCache;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use either::Either;
use parking_lot::{RwLock, RwLockWriteGuard};
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
    output: QMatMul,
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
        kv_cache_dtype: DType,
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
            fp8_kvcache: Some(kv_cache_dtype == DType::U8),
            extra_config_json,
        }
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
        kv_cache_dtype: DType,
        yarn_scaling_factor: Option<f64>,
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

        let hybrid = parse_gguf_hybrid_config(ct, &arch, block_count);
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

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?,
        };

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
                    ct,
                    reader,
                    &prefix,
                    device,
                    dtype,
                    rotary_emb.clone(),
                    cfg.sliding_window,
                )?)
            } else {
                let cur_gdn_idx = gdn_layer_idx;
                gdn_layer_idx += 1;
                AttnType::LinearAttention(QuantizedGatedDeltaNet::new(
                    ct,
                    reader,
                    &prefix,
                    device,
                    &hybrid,
                    cur_gdn_idx,
                    rms_norm_eps,
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
            let ffn_norm = ct.tensor(
                reader,
                &format!("{prefix}.post_attention_norm.weight"),
                device,
            )?;

            let (shared_gate, shared_expert) =
                if let Some(_) = moe_cfg.shared_expert_intermediate_size {
                    let ws = ct
                        .tensor(
                            reader,
                            &format!("{prefix}.ffn_gate_inp_shexp.weight"),
                            device,
                        )?
                        .dequantize(device)?
                        .reshape((1, cfg.hidden_size))?;

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

            layers.push(LayerWeights {
                attn,
                attention_norm: QRmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp,
                shared_gate,
                shared_expert,
                ffn_norm: QRmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
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
            let x = layer.mlp.forward(&x)?;
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
