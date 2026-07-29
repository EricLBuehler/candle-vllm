#[cfg(feature = "nccl")]
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, ReplicatedLinear, VarBuilder};
use crate::openai::models::layers::attention::Attention;
use crate::openai::models::layers::attention::QuantizedAttention;
use crate::openai::models::layers::mlp::Mlp;
use crate::openai::models::layers::moe::sort_expert_assignments;
use crate::openai::models::layers::moe::{FusedMoe, FusedMoeFp8};
use crate::openai::models::layers::others::{rms_norm, NormX};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::layers::quantized_var_builder::VarBuilder as QVarBuilder;
use crate::openai::models::layers::rotary_emb::ScalingRotaryEmbedding;
use crate::openai::models::linear::LinearX as Linear;
use crate::openai::models::{Config, MoEConfig, QuantConfig};
use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub static MTP_TOTAL_PROPOSED: AtomicUsize = AtomicUsize::new(0);
pub static MTP_TOTAL_ACCEPTED: AtomicUsize = AtomicUsize::new(0);
pub static MTP_TOTAL_STEPS: AtomicUsize = AtomicUsize::new(0);
pub const MTP_STATS_LOG_INTERVAL_STEPS: usize = 256;

#[derive(Debug, Clone)]
pub struct MtpVerifyResult {
    pub accepted_tokens: Vec<u32>,
    pub continuation_token: u32,
    pub num_accepted: usize,
    pub num_proposed: usize,
}

pub fn verify_draft_greedy(
    verify_logits: &Tensor,
    draft_tokens: &[u32],
) -> Result<MtpVerifyResult> {
    let num_positions = verify_logits.dim(0)?;
    let num_proposed = draft_tokens.len();
    let verify_logits = verify_logits.to_dtype(DType::F32)?;
    let target_tokens = verify_logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let compare_len = num_proposed.min(num_positions);
    let mut num_accepted = 0usize;
    for i in 0..compare_len {
        if target_tokens[i] == draft_tokens[i] {
            num_accepted += 1;
        } else {
            break;
        }
    }
    let continuation_token = if num_accepted < num_positions {
        target_tokens[num_accepted]
    } else {
        target_tokens[num_positions - 1]
    };
    Ok(MtpVerifyResult {
        accepted_tokens: draft_tokens[..num_accepted].to_vec(),
        continuation_token,
        num_accepted,
        num_proposed,
    })
}

pub fn mtp_stats_update(proposed: usize, accepted: usize) {
    MTP_TOTAL_PROPOSED.fetch_add(proposed, Ordering::Relaxed);
    MTP_TOTAL_ACCEPTED.fetch_add(accepted, Ordering::Relaxed);
    MTP_TOTAL_STEPS.fetch_add(1, Ordering::Relaxed);
}

pub fn mtp_stats_should_log(step: usize) -> bool {
    step > 0 && step % MTP_STATS_LOG_INTERVAL_STEPS == 0
}

pub fn mtp_stats_summary() -> String {
    let proposed = MTP_TOTAL_PROPOSED.load(Ordering::Relaxed);
    let accepted = MTP_TOTAL_ACCEPTED.load(Ordering::Relaxed);
    let steps = MTP_TOTAL_STEPS.load(Ordering::Relaxed);
    format!(
        "MTP Stats: proposed={}, accepted={}, acceptance_rate={:.2}%, avg_tokens/step={:.2}",
        proposed,
        accepted,
        if proposed > 0 {
            accepted as f64 / proposed as f64 * 100.0
        } else {
            0.0
        },
        if steps > 0 {
            (accepted + 2 * steps) as f64 / steps as f64
        } else {
            1.0
        }
    )
}

fn has_any_key(vb: &VarBuilder, keys: &[&str]) -> bool {
    keys.iter().any(|key| vb.contains_tensor(key))
}

fn mtp_quant_config(vb: &VarBuilder, main_quant: Option<&QuantConfig>) -> Option<QuantConfig> {
    if !has_any_key(
        vb,
        &[
            "fc.weight_scale",
            "fc.weight_scale_inv",
            "layers.0.self_attn.q_proj.weight_scale",
            "layers.0.self_attn.q_proj.weight_scale_inv",
            "layers.0.self_attn.k_proj.weight_scale",
            "layers.0.self_attn.k_proj.weight_scale_inv",
            "layers.0.self_attn.v_proj.weight_scale",
            "layers.0.self_attn.v_proj.weight_scale_inv",
            "layers.0.self_attn.o_proj.weight_scale",
            "layers.0.self_attn.o_proj.weight_scale_inv",
            "layers.0.mlp.gate_proj.weight_scale",
            "layers.0.mlp.gate_proj.weight_scale_inv",
            "layers.0.mlp.up_proj.weight_scale",
            "layers.0.mlp.up_proj.weight_scale_inv",
            "layers.0.mlp.down_proj.weight_scale",
            "layers.0.mlp.down_proj.weight_scale_inv",
            "layers.0.mlp.experts.gate_up_proj_scale_inv",
            "layers.0.mlp.experts.down_proj_scale_inv",
        ],
    ) {
        return None;
    }
    let mut cfg = main_quant.cloned().unwrap_or(QuantConfig {
        quant_method: "fp8".to_string(),
        activation_scheme: None,
        weight_per_tensor: None,
        act_per_tensor: None,
        modules_to_not_convert: None,
        bits: 0,
        group_size: 0,
        sym: None,
        desc_act: None,
        checkpoint_format: None,
        weight_block_size: None,
        format: None,
        config_groups: None,
        quantized_layers: None,
        quant_algo: None,
        mode: None,
        ignore: None,
        is_mlx_nvfp4: false,
        is_compressed_tensors: false,
    });
    cfg.quant_method = "fp8".to_string();
    cfg.weight_block_size = Some(vec![128, 128]);
    cfg.modules_to_not_convert = None;
    Some(cfg)
}

enum MtpFusedMoe {
    BF16(FusedMoe),
    FP8(FusedMoeFp8),
}

impl MtpFusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::BF16(m) => m.forward(xs, false),
            Self::FP8(m) => m.forward(xs, false),
        }
    }
}

enum MtpMlp {
    Dense(Mlp),
    Moe {
        fused_moe: MtpFusedMoe,
        shared_gate: Option<Linear>,
        shared_expert: Option<Mlp>,
    },
}

impl MtpMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(xs),
            Self::Moe {
                fused_moe,
                shared_gate,
                shared_expert,
            } => {
                let shared_output = match (shared_gate, shared_expert) {
                    (Some(gate), Some(expert)) => {
                        let gate = candle_nn::ops::sigmoid(&gate.forward(xs)?)?;
                        Some(gate.broadcast_mul(&expert.forward(xs)?)?)
                    }
                    _ => None,
                };
                let moe = fused_moe.forward(xs)?;
                if let Some(shared) = shared_output {
                    moe + shared
                } else {
                    Ok(moe)
                }
            }
        }
    }
}

struct Qwen3_5MtpDecoderLayer {
    attn: Attention,
    mlp: MtpMlp,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl Qwen3_5MtpDecoderLayer {
    fn forward_single_token(
        &self,
        xs: &Tensor,
        positions: &Tensor,
        rotary_emb: &ScalingRotaryEmbedding,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = (self
            .attn
            .forward_single_token_no_cache(&xs, rotary_emb, positions)?
            + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        residual + self.mlp.forward(&xs)?
    }
}

pub(crate) struct SafetensorMtpHead {
    pre_fc_norm_hidden: NormX,
    pre_fc_norm_embedding: NormX,
    fc: ReplicatedLinear,
    layer: Qwen3_5MtpDecoderLayer,
    norm: NormX,
    rotary_emb: Arc<ScalingRotaryEmbedding>,
    device: Device,
    dtype: DType,
}

impl SafetensorMtpHead {
    pub fn has_mtp_weights(vb: &VarBuilder) -> bool {
        let mtp = vb.pp("mtp");
        mtp.contains_tensor("fc.weight")
            || mtp.contains_tensor("layers.0.mlp.gate_proj.weight")
            || mtp.contains_tensor("layers.0.mlp.gate.weight")
    }

    fn new(
        vb: VarBuilder,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let mtp_vb = vb.pp("mtp");
        let mtp_quant_config = mtp_quant_config(&mtp_vb, config.quantization_config.as_ref());
        let mut mtp_config = config.clone();
        mtp_config.quantization_config = mtp_quant_config.clone();
        mtp_config.isq_quant = None;

        let pre_fc_norm_hidden = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp("mtp.pre_fc_norm_hidden"),
            DType::F32,
            true,
        )?;
        let pre_fc_norm_embedding = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp("mtp.pre_fc_norm_embedding"),
            DType::F32,
            true,
        )?;
        let fc = ReplicatedLinear::load_no_bias(
            hidden_size * 2,
            hidden_size,
            vb.pp("mtp.fc"),
            &None,
            &mtp_quant_config,
        )?;
        let norm = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            vb.pp("mtp.norm"),
            DType::F32,
            true,
        )?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if mtp_config.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            &mtp_config,
            device,
            true,
        )?);
        let layer_vb = vb.pp("mtp.layers.0");
        let attn = Attention::new(
            rotary_emb.clone(),
            &mtp_config,
            layer_vb.pp("self_attn"),
            comm.clone(),
            config.sliding_window,
        )?;

        let mlp_vb = layer_vb.pp("mlp");
        let is_moe = config.moe_config.is_some() && mlp_vb.contains_tensor("gate.weight");
        let mlp = if is_moe {
            let fused_moe = if let Some(quant_config) = &mtp_quant_config {
                MtpFusedMoe::FP8(FusedMoeFp8::new(
                    &mtp_config,
                    mlp_vb.clone(),
                    comm.clone(),
                    dtype,
                    quant_config,
                )?)
            } else {
                MtpFusedMoe::BF16(FusedMoe::new(
                    &mtp_config,
                    mlp_vb.clone(),
                    comm.clone(),
                    dtype,
                )?)
            };
            let moe_cfg = match config.moe_config.as_ref() {
                Some(MoEConfig::QwenMoE(cfg)) => cfg,
                _ => candle_core::bail!("Qwen3.5 MTP MoE requires Qwen MoE config"),
            };
            let (shared_gate, shared_expert) =
                if let Some(intermediate_size) = moe_cfg.shared_expert_intermediate_size {
                    if intermediate_size > 0 {
                        let ws = mlp_vb.pp("shared_expert_gate").get_with_hints_dtype(
                            (1, hidden_size),
                            "weight",
                            Default::default(),
                            dtype,
                        )?;
                        let mut shared_cfg = mtp_config.clone();
                        shared_cfg.intermediate_size = intermediate_size;
                        (
                            Some(Linear::new(ws, None, &None, &None)),
                            Some(Mlp::new(
                                &shared_cfg,
                                mlp_vb.pp("shared_expert").clone(),
                                comm.clone(),
                            )?),
                        )
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };
            MtpMlp::Moe {
                fused_moe,
                shared_gate,
                shared_expert,
            }
        } else {
            MtpMlp::Dense(Mlp::new(&mtp_config, mlp_vb, comm.clone())?)
        };

        let input_layernorm = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            layer_vb.pp("input_layernorm"),
            DType::F32,
            true,
        )?;
        let post_attention_layernorm = rms_norm(
            hidden_size,
            config.rms_norm_eps,
            layer_vb.pp("post_attention_layernorm"),
            DType::F32,
            true,
        )?;

        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layer: Qwen3_5MtpDecoderLayer {
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            },
            norm,
            rotary_emb,
            device: device.clone(),
            dtype,
        })
    }

    fn forward_step(
        &self,
        backbone_hidden: &Tensor,
        token_embedding: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let norm_hidden = self.pre_fc_norm_hidden.forward(backbone_hidden)?;
        let norm_embed = self.pre_fc_norm_embedding.forward(token_embedding)?;
        let norm_embed = norm_embed.to_dtype(norm_hidden.dtype())?;
        let fused = Tensor::cat(&[norm_embed, norm_hidden], D::Minus1)?.to_dtype(self.dtype)?;
        let xs = self.fc.forward(&fused)?;
        let xs = self
            .layer
            .forward_single_token(&xs, positions, &self.rotary_emb)?;
        self.norm.forward(&xs)
    }

    pub fn draft_tokens_gpu(
        &self,
        initial_hidden: &Tensor,
        anchor_token_tensor: &Tensor,
        num_tokens: usize,
        embed_weight: &Tensor,
        lm_head_fn: impl Fn(&Tensor) -> Result<Tensor>,
        positions_base: usize,
    ) -> Result<(Vec<u32>, Tensor)> {
        let mut gpu_draft_tokens = Vec::with_capacity(num_tokens);
        let mut current_hidden = if initial_hidden.dims().len() == 1 {
            initial_hidden.unsqueeze(0)?
        } else {
            initial_hidden.clone()
        };
        let mut current_token_t = anchor_token_tensor.reshape((1,))?;

        for step in 0..num_tokens {
            let token_embed = embed_weight.index_select(&current_token_t, 0)?;
            let positions =
                Tensor::from_vec(vec![(positions_base + step) as i64], (1,), &self.device)?;
            let hidden = self.forward_step(&current_hidden, &token_embed, &positions)?;
            let logits = lm_head_fn(&hidden.to_dtype(self.dtype)?)?;
            let logits = if logits.dims().len() == 2 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };
            let next_token = logits.to_dtype(DType::F32)?.argmax(D::Minus1)?;
            gpu_draft_tokens.push(next_token.clone());
            current_hidden = if hidden.dims().len() == 2 {
                hidden.get(hidden.dim(0)? - 1)?.unsqueeze(0)?
            } else {
                hidden
            };
            current_token_t = next_token.reshape((1,))?;
        }

        let draft_tokens = if gpu_draft_tokens.is_empty() {
            Vec::new()
        } else {
            Tensor::stack(&gpu_draft_tokens, 0)?.to_vec1::<u32>()?
        };
        Ok((draft_tokens, current_hidden.squeeze(0)?))
    }
}

enum GgufMtpMlp {
    Dense(GgufDenseMlp),
    Moe {
        fused_moe: GgufFusedMoe,
        shared_gate: Option<Linear>,
        shared_expert: Option<GgufDenseMlp>,
    },
}

impl GgufMtpMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(xs),
            Self::Moe {
                fused_moe,
                shared_gate,
                shared_expert,
            } => {
                let shared_output = match (shared_gate, shared_expert) {
                    (Some(gate), Some(expert)) => {
                        let gate = candle_nn::ops::sigmoid(&gate.forward(xs)?)?;
                        Some(gate.broadcast_mul(&expert.forward(xs)?)?)
                    }
                    _ => None,
                };
                let moe = fused_moe.forward(xs, false)?;
                if let Some(shared) = shared_output {
                    moe + shared
                } else {
                    Ok(moe)
                }
            }
        }
    }
}

struct GgufDenseMlp {
    gate: QMatMul,
    down: QMatMul,
    up: QMatMul,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    #[cfg(feature = "nccl")]
    dtype: DType,
}

impl GgufDenseMlp {
    fn new(
        vb: &QVarBuilder,
        rank: usize,
        world_size: usize,
        #[allow(unused_variables)] comm: Rc<Comm>,
        #[allow(unused_variables)] dtype: DType,
        suffix: &str,
    ) -> Result<Self> {
        Ok(Self {
            gate: QMatMul::from_arc(vb.get_sharded_no_shape(
                &format!("ffn_gate{suffix}.weight"),
                0,
                rank,
                world_size,
            )?)?,
            down: QMatMul::from_arc(vb.get_sharded_no_shape(
                &format!("ffn_down{suffix}.weight"),
                1,
                rank,
                world_size,
            )?)?,
            up: QMatMul::from_arc(vb.get_sharded_no_shape(
                &format!("ffn_up{suffix}.weight"),
                0,
                rank,
                world_size,
            )?)?,
            #[cfg(feature = "nccl")]
            all_reduce: if world_size > 1 {
                Some(AllReduce::new(comm))
            } else {
                None
            },
            #[cfg(feature = "nccl")]
            dtype,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let up = self.up.forward(xs)?;
        let mut y = self.down.forward(&(candle_nn::ops::silu(&gate)? * up)?)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            y = all_reduce.apply(&y.to_dtype(self.dtype)?)?;
            y = y.to_dtype(DType::F32)?;
        }
        Ok(y)
    }
}

struct GgufFusedMoe {
    gate: QMatMul,
    gate_experts: Arc<QTensor>,
    up_experts: Arc<QTensor>,
    down_experts: Arc<QTensor>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    dtype: DType,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    #[cfg(feature = "nccl")]
    world_size: usize,
}

impl GgufFusedMoe {
    fn new(
        vb: &QVarBuilder,
        config: &Config,
        rank: usize,
        world_size: usize,
        #[allow(unused_variables)] comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let moe_cfg = match config.moe_config.as_ref() {
            Some(MoEConfig::QwenMoE(cfg)) => cfg,
            _ => candle_core::bail!("Qwen3.5 GGUF MTP MoE requires Qwen MoE config"),
        };
        Ok(Self {
            gate: QMatMul::from_arc(vb.get_no_shape("ffn_gate_inp.weight")?)?,
            gate_experts: vb.get_sharded_no_shape("ffn_gate_exps.weight", 1, rank, world_size)?,
            up_experts: vb.get_sharded_no_shape("ffn_up_exps.weight", 1, rank, world_size)?,
            down_experts: vb.get_sharded_no_shape("ffn_down_exps.weight", 2, rank, world_size)?,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            dtype,
            #[cfg(feature = "nccl")]
            all_reduce: if world_size > 1 {
                Some(AllReduce::new(comm))
            } else {
                None
            },
            #[cfg(feature = "nccl")]
            world_size,
        })
    }

    fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.clone()
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
        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;
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
        let down_inputs = (up * gate.apply(&candle_nn::Activation::Silu)?)?;
        let mut ys = attention_rs::moe::moe_gemm_gguf(
            &down_inputs,
            &self.down_experts,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
            self.dtype,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;
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

struct GgufMtpDecoderLayer {
    attn: QuantizedAttention,
    mlp: GgufMtpMlp,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
}

impl GgufMtpDecoderLayer {
    fn forward_single_token(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = (self.attn.forward_single_token_no_cache(&xs, positions)? + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        residual + self.mlp.forward(&xs)?
    }
}

pub(crate) struct GgufMtpHead {
    pre_fc_norm_hidden: QRmsNorm,
    pre_fc_norm_embedding: QRmsNorm,
    fc: QMatMul,
    layer: GgufMtpDecoderLayer,
    norm: QRmsNorm,
    device: Device,
    dtype: DType,
}

impl GgufMtpHead {
    fn block_prefix(config: &Config) -> String {
        format!("blk.{}", config.num_hidden_layers)
    }

    fn has_mtp_weights_at(vb: &QVarBuilder, block_idx: usize) -> bool {
        let mtp_vb = vb.pp(format!("blk.{block_idx}"));
        mtp_vb.contains_key("nextn.eh_proj.weight")
            || mtp_vb.contains_key("attn_q.weight")
            || mtp_vb.contains_key("ffn_gate.weight")
            || mtp_vb.contains_key("ffn_gate_inp.weight")
    }

    fn has_mtp_weights(vb: &QVarBuilder, config: &Config) -> bool {
        Self::has_mtp_weights_at(vb, config.num_hidden_layers)
    }

    fn new(
        vb: &QVarBuilder,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        let layer_prefix = Self::block_prefix(config);
        let mtp_vb = vb.pp(&layer_prefix);
        let pre_fc_norm_hidden = QRmsNorm::from_arc_qtensor(
            mtp_vb.pp("nextn.hnorm").get_no_shape("weight")?,
            config.rms_norm_eps,
        )?;
        let pre_fc_norm_embedding = QRmsNorm::from_arc_qtensor(
            mtp_vb.pp("nextn.enorm").get_no_shape("weight")?,
            config.rms_norm_eps,
        )?;
        let fc = QMatMul::from_arc(mtp_vb.pp("nextn.eh_proj").get_no_shape("weight")?)?;
        let norm = QRmsNorm::from_arc_qtensor(
            mtp_vb.pp("nextn.shared_head_norm").get_no_shape("weight")?,
            config.rms_norm_eps,
        )?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            DType::F32,
            config,
            device,
            true,
        )?);
        let attn = QuantizedAttention::new(
            config,
            vb,
            &layer_prefix,
            device,
            dtype,
            rotary_emb,
            config.sliding_window,
            rank,
            world_size,
            comm.clone(),
        )?;
        let is_moe = config.moe_config.is_some() && mtp_vb.contains_key("ffn_gate_inp.weight");
        let mlp = if is_moe {
            let fused_moe =
                GgufFusedMoe::new(&mtp_vb, config, rank, world_size, comm.clone(), dtype)?;
            let moe_cfg = match config.moe_config.as_ref() {
                Some(MoEConfig::QwenMoE(cfg)) => cfg,
                _ => candle_core::bail!("Qwen3.5 GGUF MTP MoE requires Qwen MoE config"),
            };
            let (shared_gate, shared_expert) =
                if let Some(intermediate_size) = moe_cfg.shared_expert_intermediate_size {
                    if intermediate_size > 0 && mtp_vb.contains_key("ffn_gate_inp_shexp.weight") {
                        let ws = mtp_vb
                            .pp("ffn_gate_inp_shexp")
                            .get_no_shape("weight")?
                            .dequantize(device)?
                            .reshape((1, config.hidden_size))?;
                        (
                            Some(Linear::new(ws, None, &None, &None)),
                            Some(GgufDenseMlp::new(
                                &mtp_vb,
                                rank,
                                world_size,
                                comm.clone(),
                                dtype,
                                "_shexp",
                            )?),
                        )
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };
            GgufMtpMlp::Moe {
                fused_moe,
                shared_gate,
                shared_expert,
            }
        } else {
            GgufMtpMlp::Dense(GgufDenseMlp::new(
                &mtp_vb,
                rank,
                world_size,
                comm.clone(),
                dtype,
                "",
            )?)
        };
        let input_layernorm = QRmsNorm::from_arc_qtensor(
            mtp_vb.pp("attn_norm").get_no_shape("weight")?,
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = QRmsNorm::from_arc_qtensor(
            mtp_vb.pp("post_attention_norm").get_no_shape("weight")?,
            config.rms_norm_eps,
        )?;
        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layer: GgufMtpDecoderLayer {
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            },
            norm,
            device: device.clone(),
            dtype,
        })
    }

    fn forward_step(
        &self,
        backbone_hidden: &Tensor,
        token_embedding: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let norm_hidden = self.pre_fc_norm_hidden.forward(backbone_hidden)?;
        let norm_embed = self.pre_fc_norm_embedding.forward(token_embedding)?;
        let norm_embed = norm_embed.to_dtype(norm_hidden.dtype())?;
        let fused = Tensor::cat(&[norm_embed, norm_hidden], D::Minus1)?.to_dtype(DType::F32)?;
        let xs = self.fc.forward(&fused)?.to_dtype(self.dtype)?;
        let xs = self.layer.forward_single_token(&xs, positions)?;
        self.norm.forward(&xs)
    }

    fn draft_tokens_gpu(
        &self,
        initial_hidden: &Tensor,
        anchor_token_tensor: &Tensor,
        num_tokens: usize,
        embed_weight: &Tensor,
        lm_head_fn: impl Fn(&Tensor) -> Result<Tensor>,
        positions_base: usize,
    ) -> Result<(Vec<u32>, Tensor)> {
        let mut gpu_draft_tokens = Vec::with_capacity(num_tokens);
        let mut current_hidden = if initial_hidden.dims().len() == 1 {
            initial_hidden.unsqueeze(0)?
        } else {
            initial_hidden.clone()
        };
        let mut current_token_t = anchor_token_tensor.reshape((1,))?;

        for step in 0..num_tokens {
            let token_embed = embed_weight.index_select(&current_token_t, 0)?;
            let positions =
                Tensor::from_vec(vec![(positions_base + step) as i64], (1,), &self.device)?;
            let hidden = self.forward_step(&current_hidden, &token_embed, &positions)?;
            let logits = lm_head_fn(&hidden.to_dtype(self.dtype)?)?;
            let logits = if logits.dims().len() == 2 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };
            let next_token = logits.to_dtype(DType::F32)?.argmax(D::Minus1)?;
            gpu_draft_tokens.push(next_token.clone());
            current_hidden = if hidden.dims().len() == 2 {
                hidden.get(hidden.dim(0)? - 1)?.unsqueeze(0)?
            } else {
                hidden
            };
            current_token_t = next_token.reshape((1,))?;
        }

        let draft_tokens = if gpu_draft_tokens.is_empty() {
            Vec::new()
        } else {
            Tensor::stack(&gpu_draft_tokens, 0)?.to_vec1::<u32>()?
        };
        Ok((draft_tokens, current_hidden.squeeze(0)?))
    }
}

enum MtpHeadInner {
    Safetensor(SafetensorMtpHead),
    Gguf(GgufMtpHead),
}

pub struct Qwen3_5MtpHead {
    inner: MtpHeadInner,
}

impl Qwen3_5MtpHead {
    pub fn has_mtp_weights(vb: &VarBuilder) -> bool {
        SafetensorMtpHead::has_mtp_weights(vb)
    }

    pub fn has_gguf_mtp_weights(vb: &QVarBuilder, config: &Config) -> bool {
        GgufMtpHead::has_mtp_weights(vb, config)
    }

    pub fn has_gguf_mtp_weights_at(vb: &QVarBuilder, block_idx: usize) -> bool {
        GgufMtpHead::has_mtp_weights_at(vb, block_idx)
    }

    pub fn new(
        vb: VarBuilder,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            inner: MtpHeadInner::Safetensor(SafetensorMtpHead::new(
                vb, comm, config, dtype, device,
            )?),
        })
    }

    pub fn new_gguf(
        vb: &QVarBuilder,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: MtpHeadInner::Gguf(GgufMtpHead::new(
                vb, comm, config, dtype, device, rank, world_size,
            )?),
        })
    }

    pub fn draft_tokens_gpu(
        &self,
        initial_hidden: &Tensor,
        anchor_token_tensor: &Tensor,
        num_tokens: usize,
        embed_weight: &Tensor,
        lm_head_fn: impl Fn(&Tensor) -> Result<Tensor>,
        positions_base: usize,
    ) -> Result<(Vec<u32>, Tensor)> {
        match &self.inner {
            MtpHeadInner::Safetensor(head) => head.draft_tokens_gpu(
                initial_hidden,
                anchor_token_tensor,
                num_tokens,
                embed_weight,
                lm_head_fn,
                positions_base,
            ),
            MtpHeadInner::Gguf(head) => head.draft_tokens_gpu(
                initial_hidden,
                anchor_token_tensor,
                num_tokens,
                embed_weight,
                lm_head_fn,
                positions_base,
            ),
        }
    }
}
