use crate::candle::quantized::QTensor;
use crate::openai::distributed::{shard, AllReduce, Comm, VarBuilder};
use crate::openai::models::linear::{linear_no_bias, linear_no_bias_x, Linear, LinearX};
use crate::openai::models::{Config, MoEConfig, QuantConfig, QwenMoEConfig};
use attention_rs::moe;
use attention_rs::moe::moe_gemm_fp8;
use attention_rs::silu_and_mul::silu_and_mul;
use attention_rs::sort::ArgSortOp;
use candle::{DType, Module, Result, Tensor, D};
use candle_core as candle;
use candle_core::quantized::GgmlDType;
use candle_nn::var_builder::Shard;
use candle_nn::Activation;
use std::borrow::Cow;

use std::rc::Rc;

/// Apply gated activation on fused gate_up tensor.
/// Uses optimized `silu_and_mul` kernel for SiLU activation, falls back to
/// generic tensor operations for other activations (e.g., GeluPytorchTanh).
fn gated_activation(gate_up: &Tensor, half_dim: usize, act: &Activation) -> Result<Tensor> {
    if matches!(act, Activation::Silu) {
        silu_and_mul(gate_up, half_dim)
    } else {
        let gate = gate_up
            .narrow(candle::D::Minus1, 0, half_dim)?
            .contiguous()?;
        let up = gate_up
            .narrow(candle::D::Minus1, half_dim, half_dim)?
            .contiguous()?;
        (up * gate.apply(act)?)?.contiguous()
    }
}

pub(crate) fn sort_expert_assignments(
    topk_ids: &Tensor,
    is_prefill: bool,
) -> Result<(Tensor, Tensor)> {
    let flat = topk_ids.flatten_all()?;
    if is_prefill {
        flat.sort(true)
    } else {
        flat.sort_last_dim(true)
    }
}

fn presorted_expert_assignments(
    topk_ids: &Tensor,
    is_prefill: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    if !is_prefill {
        return Ok(None);
    }
    let (expert_ids, sorted_token_ids) = sort_expert_assignments(topk_ids, true)?;
    Ok(Some((sorted_token_ids, expert_ids)))
}

fn select_topk_indices(scores: &Tensor, topk: usize, is_prefill: bool) -> Result<Tensor> {
    let sorted_idx = if is_prefill {
        scores.contiguous()?.arg_sort(false)?
    } else {
        scores.arg_sort_last_dim(false)?
    };
    sorted_idx.narrow(D::Minus1, 0, topk)?.contiguous()
}

#[derive(Clone, Copy, Debug)]
enum PackedGateUpLayout {
    // [experts, hidden, 2*intermediate]
    HiddenPacked,
    // [experts, 2*intermediate, hidden]
    InterPacked,
}

#[derive(Clone, Copy, Debug)]
enum PackedDownLayout {
    // [experts, intermediate, hidden] -> transpose to [experts, hidden, intermediate]
    InterHidden,
    // [experts, hidden, intermediate] -> already in expected GEMM layout
    HiddenInter,
}

fn qwen_moe_cfg(cfg: &Config) -> Result<QwenMoEConfig> {
    if let Some(MoEConfig::QwenMoE(moe_cfg)) = &cfg.moe_config {
        Ok(moe_cfg.clone())
    } else {
        candle::bail!("Expected QwenMoEConfig")
    }
}

/// Resolve per-expert projection sub-prefix. Falls back from standard
/// `gate_proj`/`up_proj`/`down_proj` to MiniMax-style `w1`/`w3`/`w2`.
pub fn resolve_expert_proj_prefix(
    expert_vb: &candle_nn::var_builder::ShardedVarBuilder,
) -> (&'static str, &'static str, &'static str) {
    if expert_vb.contains_tensor("gate_proj.weight")
        || expert_vb.contains_tensor("gate_proj.weight_packed")
        || expert_vb.contains_tensor("gate_proj.blocks")
    {
        ("gate_proj", "up_proj", "down_proj")
    } else {
        ("w1", "w3", "w2")
    }
}

fn arch_name(cfg: &Config) -> &str {
    cfg.architectures
        .as_ref()
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("")
}

fn resolve_packed_gate_up_layout(cfg: &Config) -> Result<PackedGateUpLayout> {
    let arch = arch_name(cfg);

    // Qwen3.5 MoE / Qwen3-Next / Gemma4 checkpoints store gate_up as [experts, 2*intermediate, hidden].
    if matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
            | "Gemma4ForConditionalGeneration"
            | "Gemma4ForCausalLM"
    ) {
        return Ok(PackedGateUpLayout::InterPacked);
    }

    let moe_cfg = qwen_moe_cfg(cfg)?;
    if cfg.hidden_size == moe_cfg.moe_intermediate_size * 2 {
        candle::bail!(
            "Ambiguous packed gate_up_proj layout for arch {:?}: hidden_size ({}) == 2 * moe_intermediate_size ({}). Please add architecture-specific mapping.",
            arch,
            cfg.hidden_size,
            moe_cfg.moe_intermediate_size
        );
    }

    Ok(PackedGateUpLayout::HiddenPacked)
}

fn resolve_packed_down_layout(cfg: &Config) -> PackedDownLayout {
    let arch = arch_name(cfg);

    // Qwen3.5 MoE / Qwen3-Next / Gemma4 checkpoints store down_proj as [experts, hidden, intermediate].
    if matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
            | "Gemma4ForConditionalGeneration"
            | "Gemma4ForCausalLM"
    ) {
        PackedDownLayout::HiddenInter
    } else {
        PackedDownLayout::InterHidden
    }
}

fn has_packed_gate_up(experts_vb: &VarBuilder) -> bool {
    experts_vb.contains_tensor("gate_up_proj.weight") || experts_vb.contains_tensor("gate_up_proj")
}

fn get_packed_weight_3d(
    experts_vb: &VarBuilder,
    tensor_name: &str,
    shape: (usize, usize, usize),
    sh: Shard,
) -> Result<Tensor> {
    experts_vb
        .pp(tensor_name)
        .get_with_hints(shape, "weight", sh)
        .or_else(|_| experts_vb.get_with_hints(shape, tensor_name, sh))
}

fn load_packed_experts(
    cfg: &Config,
    experts_vb: VarBuilder,
    comm: Rc<Comm>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let moe_cfg = qwen_moe_cfg(cfg)?;
    let num_experts = moe_cfg.num_experts.unwrap_or(0);
    if num_experts == 0 {
        candle::bail!("num_experts must be > 0")
    }

    if has_packed_gate_up(&experts_vb) {
        let gate_up_layout = resolve_packed_gate_up_layout(cfg)?;

        let (gate_w, up_w) = match gate_up_layout {
            // [experts, hidden, 2*intermediate]
            PackedGateUpLayout::HiddenPacked => {
                let gate = get_packed_weight_3d(
                    &experts_vb,
                    "gate_up_proj",
                    (
                        num_experts,
                        cfg.hidden_size,
                        moe_cfg.moe_intermediate_size * 2,
                    ),
                    shard(2, comm.rank(), comm.world_size() * 2),
                )?
                .t()?
                .contiguous()?;

                let up = get_packed_weight_3d(
                    &experts_vb,
                    "gate_up_proj",
                    (
                        num_experts,
                        cfg.hidden_size,
                        moe_cfg.moe_intermediate_size * 2,
                    ),
                    shard(2, comm.rank() + comm.world_size(), comm.world_size() * 2),
                )?
                .t()?
                .contiguous()?;
                (gate, up)
            }
            // [experts, 2*intermediate, hidden]
            PackedGateUpLayout::InterPacked => {
                let gate = get_packed_weight_3d(
                    &experts_vb,
                    "gate_up_proj",
                    (
                        num_experts,
                        moe_cfg.moe_intermediate_size * 2,
                        cfg.hidden_size,
                    ),
                    shard(1, comm.rank(), comm.world_size() * 2),
                )?
                .contiguous()?;

                let up = get_packed_weight_3d(
                    &experts_vb,
                    "gate_up_proj",
                    (
                        num_experts,
                        moe_cfg.moe_intermediate_size * 2,
                        cfg.hidden_size,
                    ),
                    shard(1, comm.rank() + comm.world_size(), comm.world_size() * 2),
                )?
                .contiguous()?;
                (gate, up)
            }
        };

        let down_w = match resolve_packed_down_layout(cfg) {
            PackedDownLayout::InterHidden => get_packed_weight_3d(
                &experts_vb,
                "down_proj",
                (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                shard(1, comm.rank(), comm.world_size()),
            )?
            .t()?
            .contiguous()?,
            PackedDownLayout::HiddenInter => get_packed_weight_3d(
                &experts_vb,
                "down_proj",
                (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                shard(2, comm.rank(), comm.world_size()),
            )?
            .contiguous()?,
        };

        let (_, gate_n, gate_k) = gate_w.dims3()?;
        let (_, up_n, up_k) = up_w.dims3()?;
        let (_, down_n, down_k) = down_w.dims3()?;
        if gate_n != up_n
            || gate_k != up_k
            || gate_k != cfg.hidden_size
            || down_n != cfg.hidden_size
            || down_k != gate_n
        {
            candle::bail!(
                "Invalid packed MoE tensor shapes after loading: gate={:?}, up={:?}, down={:?}, hidden_size={}, arch={:?}. This usually means packed down_proj / gate_up_proj layout was interpreted incorrectly.",
                gate_w.shape(),
                up_w.shape(),
                down_w.shape(),
                cfg.hidden_size,
                cfg.architectures
            );
        }

        return Ok((gate_w, up_w, down_w));
    }

    // Legacy per-expert layout.
    let mut gate_experts = Vec::with_capacity(num_experts);
    let mut up_experts = Vec::with_capacity(num_experts);
    let mut down_experts = Vec::with_capacity(num_experts);

    for i in 0..num_experts {
        let expert_vb = experts_vb.pp(format!("{}", i).as_str());
        let (gate_name, up_name, down_name) = resolve_expert_proj_prefix(&expert_vb);
        let gate = expert_vb.pp(gate_name).get_with_hints(
            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
            "weight",
            shard(0, comm.rank(), comm.world_size()),
        )?;
        let up = expert_vb.pp(up_name).get_with_hints(
            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
            "weight",
            shard(0, comm.rank(), comm.world_size()),
        )?;
        let down = expert_vb.pp(down_name).get_with_hints(
            (cfg.hidden_size, moe_cfg.moe_intermediate_size),
            "weight",
            shard(1, comm.rank(), comm.world_size()),
        )?;

        gate_experts.push(gate);
        up_experts.push(up);
        down_experts.push(down);
    }

    Ok((
        Tensor::stack(&gate_experts, 0)?,
        Tensor::stack(&up_experts, 0)?,
        Tensor::stack(&down_experts, 0)?,
    ))
}

fn get_hidden_act(cfg: &Config) -> Activation {
    cfg.hidden_act
        .or(cfg.hidden_activation)
        .unwrap_or(Activation::Silu)
}

#[allow(dead_code)]
pub struct FusedMoe {
    gate: Linear,
    gate_up_w: Tensor,
    w_size_n: usize,
    down_w: Tensor,
    act: Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoe {
    pub fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        assert!(
            cfg.quantization_config.is_none(),
            "Invalid quantization format!"
        );
        let gate = linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
        )?;

        let (gate_w, up_w, down_w) = load_packed_experts(cfg, vb.pp("experts"), comm.clone())?;
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 1)?;
        let world_size = comm.world_size();
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_w,
            w_size_n,
            down_w,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilder,
        experts_vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let gate = linear_no_bias(cfg.hidden_size, num_experts, gate_vb, Shard::default())?;

        let (gate_w, up_w, down_w) = load_packed_experts(cfg, experts_vb, comm.clone())?;
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 1)?;
        let world_size = comm.world_size();
        let w_size_n = gate_up_w.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_w,
            w_size_n,
            down_w,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let router_logits = self.gate.forward(&xs)?;

        let (mut topk_weights, topk_ids) = attention_rs::topk::topk_softmax(
            &router_logits.to_dtype(DType::F32)?,
            self.num_experts_per_tok,
        )?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }

        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let gate_up = moe::moe_gemm(
            &xs,
            &self.gate_up_w,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?;

        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;

        let mut ys = moe::moe_gemm(
            &down_inputs,
            &self.down_w,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}

pub struct FusedMoeISQ {
    gate: Linear,
    gate_experts: QTensor,
    up_experts: QTensor,
    down_experts: QTensor,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeISQ {
    pub fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let mut quant_type = match cfg.isq_quant.as_ref().unwrap().as_str() {
            "q40" | "q4_0" => GgmlDType::Q4_0,
            "q4" | "q41" | "q4_1" => GgmlDType::Q4_1,
            "q50" | "q5_0" => GgmlDType::Q5_0,
            "q5" | "q51" | "q5_1" => GgmlDType::Q5_1,
            "q8" | "q80" | "q8_0" => GgmlDType::Q8_0,
            "q2k" | "q2_k" => GgmlDType::Q2K,
            "q3k" | "q3_k" => GgmlDType::Q3K,
            "q4k" | "q4_k" => GgmlDType::Q4K,
            "q5k" | "q5_k" => GgmlDType::Q5K,
            "q6k" | "q6_k" => GgmlDType::Q6K,
            _ => panic!("Unsupported GGML data type!"),
        };

        let get_moe_intermediate_chunk = |blk_size: usize| -> usize {
            let base = moe_cfg.moe_intermediate_size / comm.world_size();
            if base % blk_size != 0 {
                ((base + blk_size - 1) / blk_size) * blk_size
            } else {
                base
            }
        };

        let mut block_size = quant_type.block_size();
        if comm.world_size() > 1
            && moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0
        {
            // In case experts cannot be split cleanly under QK formats, fallback to q8_0.
            let chunk = get_moe_intermediate_chunk(block_size);
            if (moe_cfg.moe_intermediate_size - chunk) % (comm.world_size() - 1) != 0 {
                quant_type = GgmlDType::Q8_0;
                block_size = quant_type.block_size();
            }
        }

        let gate_ws = vb.pp("gate").get_with_hints_dtype(
            (num_experts, cfg.hidden_size),
            "weight",
            Shard::default(),
            DType::F32,
        )?;
        let gate = Linear::new(gate_ws, None);

        let (gate_experts, up_experts, down_experts) = if moe_cfg.moe_intermediate_size
            / comm.world_size()
            % block_size
            == 0
        {
            load_packed_experts(cfg, vb.pp("experts"), comm.clone())?
        } else {
            let experts_vb = vb.pp("experts");
            let mut gate_experts = Vec::with_capacity(num_experts);
            let mut up_experts = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);
            let moe_intermediate_chunk = get_moe_intermediate_chunk(block_size);

            let (gate_experts, up_experts, down_experts) = if has_packed_gate_up(&experts_vb) {
                let gate_up_layout = resolve_packed_gate_up_layout(cfg)?;
                let (gate_expert, up_expert) = match gate_up_layout {
                    PackedGateUpLayout::HiddenPacked => {
                        let gate = get_packed_weight_3d(
                            &experts_vb,
                            "gate_up_proj",
                            (
                                num_experts,
                                cfg.hidden_size,
                                moe_cfg.moe_intermediate_size * 2,
                            ),
                            shard(2, 0, 2),
                        )?
                        .t()?
                        .contiguous()?;
                        let up = get_packed_weight_3d(
                            &experts_vb,
                            "gate_up_proj",
                            (
                                num_experts,
                                cfg.hidden_size,
                                moe_cfg.moe_intermediate_size * 2,
                            ),
                            shard(2, 1, 2),
                        )?
                        .t()?
                        .contiguous()?;
                        (gate, up)
                    }
                    PackedGateUpLayout::InterPacked => {
                        let gate = get_packed_weight_3d(
                            &experts_vb,
                            "gate_up_proj",
                            (
                                num_experts,
                                moe_cfg.moe_intermediate_size * 2,
                                cfg.hidden_size,
                            ),
                            shard(1, 0, 2),
                        )?
                        .contiguous()?;
                        let up = get_packed_weight_3d(
                            &experts_vb,
                            "gate_up_proj",
                            (
                                num_experts,
                                moe_cfg.moe_intermediate_size * 2,
                                cfg.hidden_size,
                            ),
                            shard(1, 1, 2),
                        )?
                        .contiguous()?;
                        (gate, up)
                    }
                };

                let down_expert = match get_packed_weight_3d(
                    &experts_vb,
                    "down_proj",
                    (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    Shard::default(),
                ) {
                    Ok(w) => w.t()?.contiguous()?,
                    Err(_) => get_packed_weight_3d(
                        &experts_vb,
                        "down_proj",
                        (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                        Shard::default(),
                    )?
                    .contiguous()?,
                };
                (gate_expert, up_expert, down_expert)
            } else {
                for i in 0..num_experts {
                    let expert_vb = experts_vb.pp(format!("{}", i).as_str());
                    let (gate_name, up_name, down_name) = resolve_expert_proj_prefix(&expert_vb);
                    let gate = expert_vb.pp(gate_name).get_with_hints(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                        Shard::default(),
                    )?;
                    let up = expert_vb.pp(up_name).get_with_hints(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                        Shard::default(),
                    )?;
                    let down = expert_vb.pp(down_name).get_with_hints(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                        "weight",
                        Shard::default(),
                    )?;
                    gate_experts.push(gate);
                    up_experts.push(up);
                    down_experts.push(down);
                }

                (
                    Tensor::stack(&gate_experts, 0)?,
                    Tensor::stack(&up_experts, 0)?,
                    Tensor::stack(&down_experts, 0)?,
                )
            };

            let mut last_remain_size = moe_intermediate_chunk;
            if comm.rank() * moe_intermediate_chunk + moe_intermediate_chunk
                >= moe_cfg.moe_intermediate_size
            {
                last_remain_size =
                    moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk;
                assert!(
                    last_remain_size > 0 && last_remain_size % block_size == 0,
                    "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}!",
                    moe_cfg.moe_intermediate_size,
                    comm.world_size(),
                    block_size
                );
            }

            let gate_experts =
                gate_experts.narrow(1, comm.rank() * moe_intermediate_chunk, last_remain_size)?;
            let up_experts =
                up_experts.narrow(1, comm.rank() * moe_intermediate_chunk, last_remain_size)?;
            let down_experts =
                down_experts.narrow(2, comm.rank() * moe_intermediate_chunk, last_remain_size)?;
            (gate_experts, up_experts, down_experts)
        };

        let gate_last_dim = gate_experts.dim(candle_core::D::Minus1)?;
        let gate_up_quant_type = if gate_last_dim % quant_type.block_size() == 0 {
            quant_type
        } else if gate_last_dim % GgmlDType::Q8_0.block_size() == 0 {
            GgmlDType::Q8_0
        } else {
            candle::bail!(
                "ISQ MoE gate/up last dim {} incompatible with any GGUF block size",
                gate_last_dim
            );
        };
        let down_last_dim = down_experts.dim(candle_core::D::Minus1)?;
        let hp_dtype =
            crate::openai::models::layers::isq_high_precision_dtype(cfg.isq_quant.as_deref());
        let down_quant_type = if down_last_dim % hp_dtype.block_size() == 0 {
            hp_dtype
        } else if down_last_dim % quant_type.block_size() == 0 {
            quant_type
        } else if down_last_dim % GgmlDType::Q8_0.block_size() == 0 {
            GgmlDType::Q8_0
        } else {
            candle::bail!(
                "ISQ MoE down_experts last dim {} incompatible with any GGUF block size",
                down_last_dim
            );
        };
        let gate_experts = QTensor::quantize(&gate_experts, gate_up_quant_type)?;
        let up_experts = QTensor::quantize(&up_experts, gate_up_quant_type)?;
        let down_experts = QTensor::quantize(&down_experts, down_quant_type)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let xs_f32 = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let router_logits = self.gate.forward(&xs_f32)?;

        let (mut topk_weights, topk_ids) =
            attention_rs::topk::topk_softmax(&router_logits, self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }

        let mut ys = self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)?;
        if ys.dtype() != original_dtype {
            ys = ys.to_dtype(original_dtype)?;
        }
        Ok(ys)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let xs = if xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)?
        } else {
            xs.to_owned()
        };

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let ys = {
            let gate = moe::moe_gemm_gguf(
                &xs,
                &self.gate_experts,
                &None,
                &sorted_token_ids,
                &expert_ids,
                self.num_experts_per_tok,
                is_prefill,
                self.dtype,
            )?;
            let up = moe::moe_gemm_gguf(
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
            moe::moe_gemm_gguf(
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
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        Ok(ys)
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilder,
        experts_vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let mut quant_type = match cfg.isq_quant.as_ref().unwrap().as_str() {
            "q40" | "q4_0" => GgmlDType::Q4_0,
            "q4" | "q41" | "q4_1" => GgmlDType::Q4_1,
            "q50" | "q5_0" => GgmlDType::Q5_0,
            "q5" | "q51" | "q5_1" => GgmlDType::Q5_1,
            "q8" | "q80" | "q8_0" => GgmlDType::Q8_0,
            "q2k" | "q2_k" => GgmlDType::Q2K,
            "q3k" | "q3_k" => GgmlDType::Q3K,
            "q4k" | "q4_k" => GgmlDType::Q4K,
            "q5k" | "q5_k" => GgmlDType::Q5K,
            "q6k" | "q6_k" => GgmlDType::Q6K,
            _ => panic!("Unsupported GGML data type!"),
        };

        let block_size = quant_type.block_size();
        if comm.world_size() > 1
            && moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0
        {
            quant_type = GgmlDType::Q8_0;
        }

        let gate = linear_no_bias(cfg.hidden_size, num_experts, gate_vb, Shard::default())?;

        let (gate_experts, up_experts, down_experts) =
            load_packed_experts(cfg, experts_vb, comm.clone())?;

        let gate_last_dim = gate_experts.dim(candle_core::D::Minus1)?;
        let gate_up_quant_type = if gate_last_dim % quant_type.block_size() == 0 {
            quant_type
        } else if gate_last_dim % GgmlDType::Q8_0.block_size() == 0 {
            GgmlDType::Q8_0
        } else {
            candle::bail!(
                "ISQ MoE gate/up last dim {} incompatible with any GGUF block size",
                gate_last_dim
            );
        };
        let down_last_dim = down_experts.dim(candle_core::D::Minus1)?;
        let hp_dtype =
            crate::openai::models::layers::isq_high_precision_dtype(cfg.isq_quant.as_deref());
        let down_quant_type = if down_last_dim % hp_dtype.block_size() == 0 {
            hp_dtype
        } else if down_last_dim % quant_type.block_size() == 0 {
            quant_type
        } else if down_last_dim % GgmlDType::Q8_0.block_size() == 0 {
            GgmlDType::Q8_0
        } else {
            candle::bail!(
                "ISQ MoE down_experts last dim {} incompatible with any GGUF block size",
                down_last_dim
            );
        };
        let gate_experts = QTensor::quantize(&gate_experts, gate_up_quant_type)?;
        let up_experts = QTensor::quantize(&up_experts, gate_up_quant_type)?;
        let down_experts = QTensor::quantize(&down_experts, down_quant_type)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }
}

/// FP8 Mixture of Experts layer with block-wise scales
/// MoE WNA16 loader for compressed-tensors `pack-quantized` and symmetric
/// classic GPTQ checkpoints.
///
/// The checkpoint stores each expert projection independently as
/// `weight_packed` (`[out, in / pack_factor]`) and `weight_scale`
/// (`[out, in / group_size]`).  The packed tensors stay packed on device and
/// are consumed by attention.rs' grouped WNA16 kernel.
pub struct FusedMoeWNA16 {
    gate: Linear,
    gate_up_packed: Tensor,
    gate_up_scales: Tensor,
    down_packed: Tensor,
    down_scales: Tensor,
    w_size_n: usize,
    act: Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    bits: usize,
    group_size: usize,
    gate_dtype: DType,
    legacy_gptq: bool,
}

impl FusedMoeWNA16 {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        Self::new_with_gate(
            cfg,
            vb.pp("gate"),
            vb.pp("experts"),
            &vb,
            comm,
            dtype,
            quant_cfg,
        )
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilder,
        experts_vb: VarBuilder,
        _bias_vb: &VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0");
        }
        let bits = quant_cfg.bits;
        let legacy_gptq = quant_cfg.quant_method == "gptq";
        if !matches!(bits, 4 | 8) || quant_cfg.group_size <= 0 {
            candle_core::bail!(
                "WNA16 MoE requires 4/8 bits and a positive group size, got bits={bits}, group_size={}",
                quant_cfg.group_size
            );
        }
        let group_size = quant_cfg.group_size as usize;
        let pack_factor = 32 / bits;
        if cfg.hidden_size % pack_factor != 0
            || moe_cfg.moe_intermediate_size % pack_factor != 0
            || cfg.hidden_size % group_size != 0
            || moe_cfg.moe_intermediate_size % group_size != 0
        {
            candle_core::bail!(
                "WNA16 MoE dimensions must be divisible by pack_factor/group_size: hidden_size={}, moe_intermediate_size={}, pack_factor={}, group_size={group_size}",
                cfg.hidden_size,
                moe_cfg.moe_intermediate_size,
                pack_factor
            );
        }
        if quant_cfg.sym == Some(false) {
            candle_core::bail!("asymmetric WNA16 MoE is not supported yet");
        }
        if legacy_gptq && quant_cfg.desc_act == Some(true) {
            candle_core::bail!("classic GPTQ WNA16 MoE with desc_act/g_idx is not supported yet");
        }

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = linear_no_bias(cfg.hidden_size, num_experts, gate_vb, Shard::default())?;

        let mut gate_weights = Vec::with_capacity(num_experts);
        let mut gate_scales = Vec::with_capacity(num_experts);
        let mut up_weights = Vec::with_capacity(num_experts);
        let mut up_scales = Vec::with_capacity(num_experts);
        let mut down_weights = Vec::with_capacity(num_experts);
        let mut down_scales = Vec::with_capacity(num_experts);
        for expert_id in 0..num_experts {
            let expert = experts_vb.pp(expert_id.to_string().as_str());
            let (gate_name, up_name, down_name) = resolve_expert_proj_prefix(&expert);
            let gate_vb = expert.pp(gate_name);
            let up_vb = expert.pp(up_name);
            let down_vb = expert.pp(down_name);

            let (gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale) =
                if legacy_gptq {
                    // Classic GPTQ stores qweight=[K/pack,N] and scales=[K/group,N].
                    // The grouped WNA16 kernels consume [N,K/pack] and [N,K/group].
                    for (name, vb) in [
                        ("gate_proj", &gate_vb),
                        ("up_proj", &up_vb),
                        ("down_proj", &down_vb),
                    ] {
                        if !vb.contains_tensor("qzeros") {
                            candle_core::bail!(
                                "classic GPTQ MoE projection {name} is missing qzeros"
                            );
                        }
                    }
                    let gate_weight = gate_vb
                        .get_with_hints_dtype(
                            (cfg.hidden_size / pack_factor, moe_cfg.moe_intermediate_size),
                            "qweight",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::U32,
                        )?
                        .t()?
                        .contiguous()?;
                    let gate_scale = gate_vb
                        .get_with_hints_dtype(
                            (cfg.hidden_size / group_size, moe_cfg.moe_intermediate_size),
                            "scales",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    let up_weight = up_vb
                        .get_with_hints_dtype(
                            (cfg.hidden_size / pack_factor, moe_cfg.moe_intermediate_size),
                            "qweight",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::U32,
                        )?
                        .t()?
                        .contiguous()?;
                    let up_scale = up_vb
                        .get_with_hints_dtype(
                            (cfg.hidden_size / group_size, moe_cfg.moe_intermediate_size),
                            "scales",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    let down_weight = down_vb
                        .get_with_hints_dtype(
                            (moe_cfg.moe_intermediate_size / pack_factor, cfg.hidden_size),
                            "qweight",
                            shard(0, comm.rank(), comm.world_size()),
                            DType::U32,
                        )?
                        .t()?
                        .contiguous()?;
                    let down_scale = down_vb
                        .get_with_hints_dtype(
                            (moe_cfg.moe_intermediate_size / group_size, cfg.hidden_size),
                            "scales",
                            shard(0, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    (
                        gate_weight,
                        gate_scale,
                        up_weight,
                        up_scale,
                        down_weight,
                        down_scale,
                    )
                } else {
                    for (name, vb) in [
                        ("gate_proj", &gate_vb),
                        ("up_proj", &up_vb),
                        ("down_proj", &down_vb),
                    ] {
                        if vb.contains_tensor("weight_g_idx") {
                            candle_core::bail!(
                                "compressed-tensors WNA16 MoE projection {name} has unsupported weight_g_idx"
                            );
                        }
                    }
                    let gate_weight = gate_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / pack_factor),
                        "weight_packed",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::U32,
                    )?;
                    let gate_scale = gate_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / group_size),
                        "weight_scale",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?;
                    let up_weight = up_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / pack_factor),
                        "weight_packed",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::U32,
                    )?;
                    let up_scale = up_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / group_size),
                        "weight_scale",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?;
                    let down_weight = down_vb.get_with_hints_dtype(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / pack_factor),
                        "weight_packed",
                        shard(1, comm.rank(), comm.world_size()),
                        DType::U32,
                    )?;
                    let down_scale = down_vb.get_with_hints_dtype(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / group_size),
                        "weight_scale",
                        shard(1, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?;
                    (
                        gate_weight,
                        gate_scale,
                        up_weight,
                        up_scale,
                        down_weight,
                        down_scale,
                    )
                };
            gate_weights.push(gate_weight);
            gate_scales.push(gate_scale);
            up_weights.push(up_weight);
            up_scales.push(up_scale);
            down_weights.push(down_weight);
            down_scales.push(down_scale);
        }

        let gate_up_packed = Tensor::cat(
            &[
                &Tensor::stack(&gate_weights, 0)?,
                &Tensor::stack(&up_weights, 0)?,
            ],
            1,
        )?
        .contiguous()?;
        let gate_up_scales = Tensor::cat(
            &[
                &Tensor::stack(&gate_scales, 0)?,
                &Tensor::stack(&up_scales, 0)?,
            ],
            1,
        )?
        .contiguous()?;
        let down_packed = Tensor::stack(&down_weights, 0)?.contiguous()?;
        let down_scales = Tensor::stack(&down_scales, 0)?.contiguous()?;
        let w_size_n = gate_up_packed.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_packed,
            gate_up_scales,
            down_packed,
            down_scales,
            w_size_n,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            bits,
            group_size,
            gate_dtype,
            legacy_gptq,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let gate_input = if xs.dtype() != self.gate_dtype {
            std::borrow::Cow::Owned(xs.to_dtype(self.gate_dtype)?)
        } else {
            std::borrow::Cow::Borrowed(xs)
        };
        let router_logits = self.gate.forward(&gate_input)?;
        let (mut topk_weights, topk_ids) = attention_rs::topk::topk_softmax(
            &router_logits.to_dtype(DType::F32)?,
            self.num_experts_per_tok,
        )?;
        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }

        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let gate_up = moe::moe_gemm_wna16(
            xs,
            &self.gate_up_packed,
            &self.gate_up_scales,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            self.bits,
            self.group_size,
            is_prefill,
            self.legacy_gptq,
        )?;
        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;
        let mut ys = moe::moe_gemm_wna16(
            &down_inputs,
            &self.down_packed,
            &self.down_scales,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            self.bits,
            self.group_size,
            is_prefill,
            self.legacy_gptq,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }
}

pub struct FusedMoeFp8 {
    gate: Linear,
    gate_up_experts: Tensor,
    gate_up_experts_scale: Tensor,
    w_size_n: usize,
    down_experts: Tensor,
    down_experts_scale: Tensor,
    act: Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    block_size: Vec<usize>,
    gate_dtype: DType,
}

impl FusedMoeFp8 {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let block_size = quant_cfg
            .weight_block_size
            .clone()
            .unwrap_or(vec![128, 128]);
        if block_size.len() != 2 {
            candle::bail!("FusedMoeFp8: weight_block_size must have 2 elements");
        }
        let by = block_size[0];
        let bx = block_size[1];

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = Linear::new(
            vb.pp("gate").get_with_hints_dtype(
                (num_experts, cfg.hidden_size),
                "weight",
                Shard::default(),
                gate_dtype,
            )?,
            None,
        );

        let experts_vb = vb.pp("experts");

        let (
            gate_experts,
            gate_experts_scale,
            up_experts,
            up_experts_scale,
            down_experts,
            down_experts_scale,
        ) = if has_packed_gate_up(&experts_vb) {
            let gate_up_layout = resolve_packed_gate_up_layout(cfg)?;
            let hidden_blocks = (cfg.hidden_size + bx - 1) / bx;
            let inter_blocks = (moe_cfg.moe_intermediate_size + by - 1) / by;
            let local_inter_blocks = inter_blocks / comm.world_size();
            let start_blocks = comm.rank() * local_inter_blocks;

            let (gate_weight, gate_s, up_weight, up_s) = match gate_up_layout {
                PackedGateUpLayout::HiddenPacked => {
                    let scale_n = (cfg.hidden_size + by - 1) / by;
                    let scale_k = (moe_cfg.moe_intermediate_size * 2 + bx - 1) / bx;

                    let gate_weight = experts_vb
                        .get_with_hints_dtype(
                            (
                                num_experts,
                                cfg.hidden_size,
                                moe_cfg.moe_intermediate_size * 2,
                            ),
                            "gate_up_proj",
                            shard(2, comm.rank(), comm.world_size() * 2),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let up_weight = experts_vb
                        .get_with_hints_dtype(
                            (
                                num_experts,
                                cfg.hidden_size,
                                moe_cfg.moe_intermediate_size * 2,
                            ),
                            "gate_up_proj",
                            shard(2, comm.rank() + comm.world_size(), comm.world_size() * 2),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let gate_up_scale = experts_vb.get_with_hints_dtype(
                        (num_experts, scale_n, scale_k),
                        "gate_up_proj_scale_inv",
                        Shard::default(),
                        DType::F32,
                    )?;

                    let gate_s_t = gate_up_scale.narrow(2, 0, inter_blocks)?.contiguous()?;
                    let up_s_t = gate_up_scale
                        .narrow(2, inter_blocks, inter_blocks)?
                        .contiguous()?;

                    let gate_s = gate_s_t
                        .narrow(2, start_blocks, local_inter_blocks)?
                        .t()?
                        .contiguous()?;
                    let up_s = up_s_t
                        .narrow(2, start_blocks, local_inter_blocks)?
                        .t()?
                        .contiguous()?;
                    (gate_weight, gate_s, up_weight, up_s)
                }
                PackedGateUpLayout::InterPacked => {
                    let scale_n = (moe_cfg.moe_intermediate_size * 2 + by - 1) / by;

                    let gate_weight = experts_vb
                        .get_with_hints_dtype(
                            (
                                num_experts,
                                moe_cfg.moe_intermediate_size * 2,
                                cfg.hidden_size,
                            ),
                            "gate_up_proj",
                            shard(1, comm.rank(), comm.world_size() * 2),
                            DType::U8,
                        )?
                        .contiguous()?;

                    let up_weight = experts_vb
                        .get_with_hints_dtype(
                            (
                                num_experts,
                                moe_cfg.moe_intermediate_size * 2,
                                cfg.hidden_size,
                            ),
                            "gate_up_proj",
                            shard(1, comm.rank() + comm.world_size(), comm.world_size() * 2),
                            DType::U8,
                        )?
                        .contiguous()?;

                    let gate_up_scale = experts_vb.get_with_hints_dtype(
                        (num_experts, scale_n, hidden_blocks),
                        "gate_up_proj_scale_inv",
                        Shard::default(),
                        DType::F32,
                    )?;

                    let gate_s = gate_up_scale
                        .narrow(1, start_blocks, local_inter_blocks)?
                        .contiguous()?;
                    let up_s = gate_up_scale
                        .narrow(1, inter_blocks + start_blocks, local_inter_blocks)?
                        .contiguous()?;
                    (gate_weight, gate_s, up_weight, up_s)
                }
            };

            let (down_weight, down_s) = match resolve_packed_down_layout(cfg) {
                PackedDownLayout::InterHidden => {
                    let scale_n = (cfg.hidden_size + by - 1) / by;
                    let scale_k = (moe_cfg.moe_intermediate_size + bx - 1) / bx;
                    let down_weight = experts_vb
                        .get_with_hints_dtype(
                            (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                            "down_proj",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::U8,
                        )?
                        .t()?
                        .contiguous()?;

                    let down_s = experts_vb
                        .get_with_hints_dtype(
                            (num_experts, scale_k, scale_n),
                            "down_proj_scale_inv",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .t()?
                        .contiguous()?;
                    (down_weight, down_s)
                }
                PackedDownLayout::HiddenInter => {
                    let scale_n = (cfg.hidden_size + by - 1) / by;
                    let scale_k = (moe_cfg.moe_intermediate_size + bx - 1) / bx;
                    let down_weight = experts_vb
                        .get_with_hints_dtype(
                            (num_experts, cfg.hidden_size, moe_cfg.moe_intermediate_size),
                            "down_proj",
                            shard(2, comm.rank(), comm.world_size()),
                            DType::U8,
                        )?
                        .contiguous()?;

                    let down_s = experts_vb
                        .get_with_hints_dtype(
                            (num_experts, scale_n, scale_k),
                            "down_proj_scale_inv",
                            shard(2, comm.rank(), comm.world_size()),
                            DType::F32,
                        )?
                        .contiguous()?;
                    (down_weight, down_s)
                }
            };

            (gate_weight, gate_s, up_weight, up_s, down_weight, down_s)
        } else {
            // Per-expert loading
            let mut gate_experts = Vec::with_capacity(num_experts);
            let mut gate_experts_scale = Vec::with_capacity(num_experts);
            let mut up_experts = Vec::with_capacity(num_experts);
            let mut up_experts_scale = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);
            let mut down_experts_scale = Vec::with_capacity(num_experts);

            for i in 0..num_experts {
                let expert_vb = experts_vb.pp(format!("{}", i).as_str());
                let (gate_name, up_name, down_name) = resolve_expert_proj_prefix(&expert_vb);

                let gate_weight = expert_vb.pp(gate_name).get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let sn = (moe_cfg.moe_intermediate_size + by - 1) / by;
                let sk = (cfg.hidden_size + bx - 1) / bx;
                let gate_s = match expert_vb.pp(gate_name).get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp(gate_name).get_with_hints_dtype(
                        (sn, sk),
                        "weight_scale_inv",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                let up_weight = expert_vb.pp(up_name).get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let up_s = match expert_vb.pp(up_name).get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp(up_name).get_with_hints_dtype(
                        (sn, sk),
                        "weight_scale_inv",
                        shard(0, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                let down_weight = expert_vb.pp(down_name).get_with_hints_dtype(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let down_sn = (cfg.hidden_size + by - 1) / by;
                let down_sk = (moe_cfg.moe_intermediate_size + bx - 1) / bx;
                let down_s = match expert_vb.pp(down_name).get_with_hints_dtype(
                    (down_sn, down_sk),
                    "weight_scale",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb.pp(down_name).get_with_hints_dtype(
                        (down_sn, down_sk),
                        "weight_scale_inv",
                        shard(1, comm.rank(), comm.world_size()),
                        DType::F32,
                    )?,
                };

                gate_experts.push(gate_weight);
                gate_experts_scale.push(gate_s);
                up_experts.push(up_weight);
                up_experts_scale.push(up_s);
                down_experts.push(down_weight);
                down_experts_scale.push(down_s);
            }

            (
                Tensor::stack(&gate_experts, 0)?,
                Tensor::stack(&gate_experts_scale, 0)?,
                Tensor::stack(&up_experts, 0)?,
                Tensor::stack(&up_experts_scale, 0)?,
                Tensor::stack(&down_experts, 0)?,
                Tensor::stack(&down_experts_scale, 0)?,
            )
        };

        let gate_up_experts = Tensor::cat(&[&gate_experts, &up_experts], 1)?;
        let gate_up_experts_scale = Tensor::cat(&[&gate_experts_scale, &up_experts_scale], 1)?;
        let w_size_n = gate_up_experts.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_experts,
            gate_up_experts_scale,
            w_size_n,
            down_experts,
            down_experts_scale,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            block_size: vec![by, bx],
            gate_dtype,
        })
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilder,
        experts_vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let block_size = quant_cfg
            .weight_block_size
            .clone()
            .unwrap_or(vec![128, 128]);
        if block_size.len() != 2 {
            candle::bail!("FusedMoeFp8: weight_block_size must have 2 elements");
        }
        let by = block_size[0];
        let bx = block_size[1];

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = Linear::new(
            gate_vb.get_with_hints_dtype(
                (num_experts, cfg.hidden_size),
                "weight",
                Shard::default(),
                gate_dtype,
            )?,
            None,
        );

        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut gate_experts_scale = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut up_experts_scale = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);
        let mut down_experts_scale = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            let expert_vb = experts_vb.pp(format!("{}", i).as_str());
            let (gate_name, up_name, down_name) = resolve_expert_proj_prefix(&expert_vb);

            let gate_weight = expert_vb.pp(gate_name).get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                "weight",
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let sn = (moe_cfg.moe_intermediate_size + by - 1) / by;
            let sk = (cfg.hidden_size + bx - 1) / bx;
            let gate_s = match expert_vb.pp(gate_name).get_with_hints_dtype(
                (sn, sk),
                "weight_scale",
                shard(0, comm.rank(), comm.world_size()),
                DType::F32,
            ) {
                Ok(s) => s,
                Err(_) => expert_vb.pp(gate_name).get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale_inv",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                )?,
            };

            let up_weight = expert_vb.pp(up_name).get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                "weight",
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let up_s = match expert_vb.pp(up_name).get_with_hints_dtype(
                (sn, sk),
                "weight_scale",
                shard(0, comm.rank(), comm.world_size()),
                DType::F32,
            ) {
                Ok(s) => s,
                Err(_) => expert_vb.pp(up_name).get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale_inv",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                )?,
            };

            let down_weight = expert_vb.pp(down_name).get_with_hints_dtype(
                (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                "weight",
                shard(1, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let down_sn = (cfg.hidden_size + by - 1) / by;
            let down_sk = (moe_cfg.moe_intermediate_size + bx - 1) / bx;
            let down_s = match expert_vb.pp(down_name).get_with_hints_dtype(
                (down_sn, down_sk),
                "weight_scale",
                shard(1, comm.rank(), comm.world_size()),
                DType::F32,
            ) {
                Ok(s) => s,
                Err(_) => expert_vb.pp(down_name).get_with_hints_dtype(
                    (down_sn, down_sk),
                    "weight_scale_inv",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::F32,
                )?,
            };

            gate_experts.push(gate_weight);
            gate_experts_scale.push(gate_s);
            up_experts.push(up_weight);
            up_experts_scale.push(up_s);
            down_experts.push(down_weight);
            down_experts_scale.push(down_s);
        }

        let gate_experts = Tensor::stack(&gate_experts, 0)?;
        let gate_experts_scale = Tensor::stack(&gate_experts_scale, 0)?;
        let up_experts = Tensor::stack(&up_experts, 0)?;
        let up_experts_scale = Tensor::stack(&up_experts_scale, 0)?;
        let down_experts = Tensor::stack(&down_experts, 0)?;
        let down_experts_scale = Tensor::stack(&down_experts_scale, 0)?;

        let gate_up_experts = Tensor::cat(&[&gate_experts, &up_experts], 1)?;
        let gate_up_experts_scale = Tensor::cat(&[&gate_experts_scale, &up_experts_scale], 1)?;
        let w_size_n = gate_up_experts.dim(1)? / 2;

        Ok(Self {
            gate,
            gate_up_experts,
            gate_up_experts_scale,
            w_size_n,
            down_experts,
            down_experts_scale,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            block_size: vec![by, bx],
            gate_dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let gate_input = if xs.dtype() != self.gate_dtype {
            Cow::Owned(xs.to_dtype(self.gate_dtype)?)
        } else {
            Cow::Borrowed(xs)
        };
        let router_logits = self.gate.forward(&gate_input)?.to_dtype(DType::F32)?;

        let (mut topk_weights, topk_ids) =
            attention_rs::topk::topk_softmax(&router_logits, self.num_experts_per_tok)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }
        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }

        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let original_dtype = xs.dtype();

        let (expert_ids, sorted_token_ids) = sort_expert_assignments(&topk_ids, is_prefill)?;

        let xs = if xs.dtype() == DType::F32 {
            xs.to_dtype(self.dtype)?
        } else {
            xs.clone()
        };

        let gate_up = moe_gemm_fp8(
            &xs,
            &self.gate_up_experts,
            &self.gate_up_experts_scale,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            self.block_size[0],
            self.block_size[1],
            is_prefill,
        )?;

        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;

        let mut ys = moe_gemm_fp8(
            &down_inputs,
            &self.down_experts,
            &self.down_experts_scale,
            &Some(topk_weights),
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            self.block_size[0],
            self.block_size[1],
            is_prefill,
        )?
        .reshape((num_tokens, (), hidden_dim))?
        .sum(D::Minus2)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}

pub struct FusedMoeMxfp4 {
    gate: LinearX,
    gate_up_blocks: Tensor,
    gate_up_scales: Tensor,
    down_blocks: Tensor,
    down_scales: Tensor,
    w_size_n: usize,
    act: Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeMxfp4 {
    fn mxfp4_tensor_name_packed(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_packed") {
            "weight_packed"
        } else {
            "blocks"
        }
    }

    fn mxfp4_tensor_name_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_scale") {
            "weight_scale"
        } else {
            "scales"
        }
    }

    pub fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let quant_method = cfg
            .quantization_config
            .as_ref()
            .map(|q| q.quant_method.clone());
        let gate = linear_no_bias_x(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &quant_method,
            &cfg.quantization_config,
            dtype,
            None,
        )?;

        let mut gate_blocks_vec = Vec::new();
        let mut gate_scales_vec = Vec::new();
        let mut up_blocks_vec = Vec::new();
        let mut up_scales_vec = Vec::new();
        let mut down_blocks_vec = Vec::new();
        let mut down_scales_vec = Vec::new();

        let experts_vb = vb.pp("experts");

        for i in 0..num_experts {
            let expert_vb = experts_vb.pp(i.to_string());
            let (gn, un, dn) = resolve_expert_proj_prefix(&expert_vb);

            let gate_proj_vb = expert_vb.pp(gn);
            let packed_name = Self::mxfp4_tensor_name_packed(&gate_proj_vb);
            let scale_name = Self::mxfp4_tensor_name_scale(&gate_proj_vb);

            let gate_b = gate_proj_vb.get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                packed_name,
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let gate_s = gate_proj_vb.get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size / 32),
                scale_name,
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;

            let up_proj_vb = expert_vb.pp(un);
            let packed_name = Self::mxfp4_tensor_name_packed(&up_proj_vb);
            let scale_name = Self::mxfp4_tensor_name_scale(&up_proj_vb);

            let up_b = up_proj_vb.get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                packed_name,
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let up_s = up_proj_vb.get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size / 32),
                scale_name,
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;

            let down_proj_vb = expert_vb.pp(dn);
            let packed_name = Self::mxfp4_tensor_name_packed(&down_proj_vb);
            let scale_name = Self::mxfp4_tensor_name_scale(&down_proj_vb);

            let down_b = down_proj_vb.get_with_hints_dtype(
                (cfg.hidden_size, moe_cfg.moe_intermediate_size / 2),
                packed_name,
                shard(1, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let down_s = down_proj_vb.get_with_hints_dtype(
                (cfg.hidden_size, moe_cfg.moe_intermediate_size / 32),
                scale_name,
                shard(1, comm.rank(), comm.world_size()),
                DType::U8,
            )?;

            gate_blocks_vec.push(gate_b);
            gate_scales_vec.push(gate_s);
            up_blocks_vec.push(up_b);
            up_scales_vec.push(up_s);
            down_blocks_vec.push(down_b);
            down_scales_vec.push(down_s);
        }

        let gate_blocks = Tensor::stack(&gate_blocks_vec, 0)?;
        let gate_scales = Tensor::stack(&gate_scales_vec, 0)?;
        let up_blocks = Tensor::stack(&up_blocks_vec, 0)?;
        let up_scales = Tensor::stack(&up_scales_vec, 0)?;

        let gate_up_blocks = Tensor::cat(&[&gate_blocks, &up_blocks], 1)?;
        let gate_up_scales = Tensor::cat(&[&gate_scales, &up_scales], 1)?;
        let w_size_n = gate_up_blocks.dim(1)? / 2;

        let down_blocks = Tensor::stack(&down_blocks_vec, 0)?;
        let down_scales = Tensor::stack(&down_scales_vec, 0)?;

        Ok(Self {
            gate,
            gate_up_blocks,
            gate_up_scales,
            down_blocks,
            down_scales,
            w_size_n,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
        })
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilder,
        experts_vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let gate = linear_no_bias_x(
            cfg.hidden_size,
            num_experts,
            gate_vb,
            Shard::default(),
            &None,
            &None,
            dtype,
            None,
        )?;

        let mut gate_blocks_vec = Vec::new();
        let mut gate_scales_vec = Vec::new();
        let mut up_blocks_vec = Vec::new();
        let mut up_scales_vec = Vec::new();
        let mut down_blocks_vec = Vec::new();
        let mut down_scales_vec = Vec::new();

        for i in 0..num_experts {
            let expert_vb = experts_vb.pp(i.to_string());
            let (gn, un, dn) = resolve_expert_proj_prefix(&expert_vb);

            let gate_proj_vb = expert_vb.pp(gn);
            let packed_name = Self::mxfp4_tensor_name_packed(&gate_proj_vb);
            let scale_name = Self::mxfp4_tensor_name_scale(&gate_proj_vb);

            let gate_b = gate_proj_vb.get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                packed_name,
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let gate_s = gate_proj_vb.get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size / 32),
                scale_name,
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;

            let up_proj_vb = expert_vb.pp(un);
            let packed_name = Self::mxfp4_tensor_name_packed(&up_proj_vb);
            let scale_name = Self::mxfp4_tensor_name_scale(&up_proj_vb);

            let up_b = up_proj_vb.get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                packed_name,
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let up_s = up_proj_vb.get_with_hints_dtype(
                (moe_cfg.moe_intermediate_size, cfg.hidden_size / 32),
                scale_name,
                shard(0, comm.rank(), comm.world_size()),
                DType::U8,
            )?;

            let down_proj_vb = expert_vb.pp(dn);
            let packed_name = Self::mxfp4_tensor_name_packed(&down_proj_vb);
            let scale_name = Self::mxfp4_tensor_name_scale(&down_proj_vb);

            let down_b = down_proj_vb.get_with_hints_dtype(
                (cfg.hidden_size, moe_cfg.moe_intermediate_size / 2),
                packed_name,
                shard(1, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
            let down_s = down_proj_vb.get_with_hints_dtype(
                (cfg.hidden_size, moe_cfg.moe_intermediate_size / 32),
                scale_name,
                shard(1, comm.rank(), comm.world_size()),
                DType::U8,
            )?;

            gate_blocks_vec.push(gate_b);
            gate_scales_vec.push(gate_s);
            up_blocks_vec.push(up_b);
            up_scales_vec.push(up_s);
            down_blocks_vec.push(down_b);
            down_scales_vec.push(down_s);
        }

        let gate_blocks = Tensor::stack(&gate_blocks_vec, 0)?;
        let gate_scales = Tensor::stack(&gate_scales_vec, 0)?;
        let up_blocks = Tensor::stack(&up_blocks_vec, 0)?;
        let up_scales = Tensor::stack(&up_scales_vec, 0)?;

        let gate_up_blocks = Tensor::cat(&[&gate_blocks, &up_blocks], 1)?;
        let gate_up_scales = Tensor::cat(&[&gate_scales, &up_scales], 1)?;
        let w_size_n = gate_up_blocks.dim(1)? / 2;

        let down_blocks = Tensor::stack(&down_blocks_vec, 0)?;
        let down_scales = Tensor::stack(&down_scales_vec, 0)?;

        Ok(Self {
            gate,
            gate_up_blocks,
            gate_up_scales,
            down_blocks,
            down_scales,
            w_size_n,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let router_logits = self.gate.forward(xs)?;

        let (mut topk_weights, topk_ids) = attention_rs::topk::topk_softmax(
            &router_logits.to_dtype(DType::F32)?,
            self.num_experts_per_tok,
        )?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }

        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;

        let moe_dtype = if self.dtype == DType::F32 {
            DType::BF16
        } else {
            self.dtype
        };
        let xs = if xs.dtype() != moe_dtype {
            xs.to_dtype(moe_dtype)?
        } else {
            xs.clone()
        };

        let gate_up = moe::moe_gemm_mxfp4(
            &xs,
            &self.gate_up_blocks,
            &self.gate_up_scales,
            None,
            &topk_ids,
            is_prefill,
            None,
        )?;

        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;
        let down_inputs = if down_inputs.dtype() != moe_dtype {
            down_inputs.to_dtype(moe_dtype)?
        } else {
            down_inputs
        };

        let mut ys = moe::moe_gemm_mxfp4(
            &down_inputs,
            &self.down_blocks,
            &self.down_scales,
            None,
            &topk_ids,
            is_prefill,
            Some(&topk_weights),
        )?
        .reshape((num_tokens, self.num_experts_per_tok, hidden_dim))?
        .sum(1)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }
}

struct Nvfp4Projection {
    blocks: Tensor,
    scales: Tensor,
    global_scales: Tensor,
    input_scales: Tensor,
    scales_swizzled: Option<Tensor>,
}

enum Nvfp4GateUpWeights {
    Fused(Nvfp4Projection),
    Separate {
        gate: Nvfp4Projection,
        up: Nvfp4Projection,
    },
}

fn maybe_swizzle_nvfp4_scales(scales: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    {
        let sm = attention_rs::cuda_utils::sm_version(scales.device().as_cuda_device()?)
            .unwrap_or(0) as usize;
        if sm >= 100 {
            return Ok(Some(
                attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(scales)?,
            ));
        }
    }
    Ok(None)
}

impl Nvfp4Projection {
    fn new(
        blocks: Tensor,
        scales: Tensor,
        global_scales: Vec<f32>,
        input_scales: Vec<f32>,
    ) -> Result<Self> {
        let dev = blocks.device().clone();
        let num_experts = blocks.dim(0)?;
        if global_scales.len() != num_experts || input_scales.len() != num_experts {
            candle::bail!(
                "NVFP4 MoE projection scale count mismatch: experts={}, global={}, input={}",
                num_experts,
                global_scales.len(),
                input_scales.len()
            );
        }
        let scales_swizzled = maybe_swizzle_nvfp4_scales(&scales)?;
        Ok(Self {
            blocks,
            scales,
            global_scales: Tensor::from_vec(global_scales, (num_experts,), &dev)?,
            input_scales: Tensor::from_vec(input_scales, (num_experts,), &dev)?,
            scales_swizzled,
        })
    }
}

pub struct FusedMoeNvfp4 {
    gate: LinearX,
    gate_up: Nvfp4GateUpWeights,
    down_blocks: Tensor,
    down_scales: Tensor,
    down_global_scales: Tensor,
    down_input_scales: Tensor,
    down_scales_swizzled: Option<Tensor>,
    w_size_n: usize,
    act: Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    gate_dtype: DType,
    use_sigmoid_scoring: bool,
    n_group: usize,
    topk_group: usize,
    apply_router_weight_on_input: bool,
    e_score_correction_bias: Option<Tensor>,
}

impl FusedMoeNvfp4 {
    fn load_mlx_weight(
        vb: &candle_nn::var_builder::ShardedVarBuilder,
        out_dim: usize,
        in_dim: usize,
        shard: Shard,
    ) -> Result<(Tensor, Tensor)> {
        let weight = vb.get_with_hints_dtype((out_dim, in_dim / 8), "weight", shard, DType::U32)?;
        let blocks = attention_rs::nvfp4_linear::mlx_repack_u32_to_u8(&weight)?;
        let scales = vb.get_with_hints_dtype((out_dim, in_dim / 16), "scales", shard, DType::U8)?;
        Ok((blocks, scales))
    }

    fn tensor_name_packed(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_packed") {
            "weight_packed"
        } else if vb.contains_tensor("weight") {
            "weight"
        } else {
            "blocks"
        }
    }

    fn tensor_name_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> &'static str {
        if vb.contains_tensor("weight_scale") {
            "weight_scale"
        } else {
            "scales"
        }
    }

    fn load_global_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> f32 {
        let no_shard = Shard::default();
        if vb.contains_tensor("weight_global_scale") {
            let raw = vb
                .get_with_hints_dtype((1,), "weight_global_scale", no_shard, DType::F32)
                .or_else(|_| {
                    vb.get_with_hints_dtype((), "weight_global_scale", no_shard, DType::F32)
                })
                .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                .unwrap_or(1.0);
            if raw != 0.0 {
                1.0 / raw
            } else {
                1.0
            }
        } else if vb.contains_tensor("weight_scale_2") {
            vb.get_with_hints_dtype((1,), "weight_scale_2", no_shard, DType::F32)
                .or_else(|_| vb.get_with_hints_dtype((), "weight_scale_2", no_shard, DType::F32))
                .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                .unwrap_or(1.0)
        } else {
            1.0
        }
    }

    fn load_input_scale(vb: &candle_nn::var_builder::ShardedVarBuilder) -> f32 {
        let no_shard = Shard::default();
        if vb.contains_tensor("input_scale") {
            vb.get_with_hints_dtype((1,), "input_scale", no_shard, DType::F32)
                .or_else(|_| vb.get_with_hints_dtype((), "input_scale", no_shard, DType::F32))
                .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                .unwrap_or(1.0)
        } else if vb.contains_tensor("input_global_scale") {
            let raw = vb
                .get_with_hints_dtype((1,), "input_global_scale", no_shard, DType::F32)
                .or_else(|_| {
                    vb.get_with_hints_dtype((), "input_global_scale", no_shard, DType::F32)
                })
                .and_then(|t| t.flatten_all()?.to_vec1::<f32>().map(|v| v[0]))
                .unwrap_or(1.0);
            if raw != 0.0 {
                1.0 / raw
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    fn use_sigmoid_routing(moe_cfg: &QwenMoEConfig) -> bool {
        moe_cfg
            .topk_method
            .as_deref()
            .is_some_and(|method| method == "noaux_tc")
            || moe_cfg
                .scoring_func
                .as_deref()
                .is_some_and(|scoring| scoring == "sigmoid")
    }

    fn get_packed_tensor(
        vb: &candle_nn::var_builder::ShardedVarBuilder,
        shape: (usize, usize, usize),
        names: &[&str],
        shard: Shard,
        dtype: DType,
    ) -> Result<Tensor> {
        for name in names {
            if let Ok(tensor) = vb.get_with_hints_dtype(shape, name, shard, dtype) {
                return Ok(tensor);
            }
        }
        candle::bail!("missing packed NVFP4 tensor (tried {:?})", names)
    }

    fn get_scalar(vb: &candle_nn::var_builder::ShardedVarBuilder, names: &[&str]) -> Option<f32> {
        for name in names {
            if !vb.contains_tensor(name) {
                continue;
            }
            let tensor = vb
                .get_with_hints_dtype((), name, Shard::default(), DType::F32)
                .or_else(|_| vb.get_with_hints_dtype((1,), name, Shard::default(), DType::F32));
            if let Ok(tensor) = tensor {
                if let Ok(values) = tensor.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    if let Some(value) = values.first() {
                        return Some(*value);
                    }
                }
            }
        }
        None
    }

    fn packed_global_scale(
        vb: &candle_nn::var_builder::ShardedVarBuilder,
        direct_names: &[&str],
        inverse_names: &[&str],
    ) -> f32 {
        if let Some(value) = Self::get_scalar(vb, direct_names) {
            value
        } else if let Some(value) = Self::get_scalar(vb, inverse_names) {
            if value != 0.0 {
                1.0 / value
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    fn packed_input_scale(
        vb: &candle_nn::var_builder::ShardedVarBuilder,
        direct_names: &[&str],
        inverse_names: &[&str],
    ) -> f32 {
        if let Some(value) = Self::get_scalar(vb, direct_names) {
            value
        } else if let Some(value) = Self::get_scalar(vb, inverse_names) {
            if value != 0.0 {
                1.0 / value
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    fn load_packed_experts(
        cfg: &Config,
        experts_vb: &candle_nn::var_builder::ShardedVarBuilder,
        comm: &Comm,
    ) -> Result<(
        Nvfp4GateUpWeights,
        Vec<Tensor>,
        Vec<Tensor>,
        Vec<f32>,
        Vec<f32>,
    )> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        let inter = moe_cfg.moe_intermediate_size;
        let hidden = cfg.hidden_size;
        let world = comm.world_size().max(1);
        let rank = comm.rank();
        // ModelOpt's packed NVFP4 layout is [E, K/2, 2*N]. Split the output
        // dimension across gate/up, then rebuild [E, 2*N_local, K/2] so the
        // MoE kernel can execute one fused gate_up GEMM.
        let gate_blocks = Self::get_packed_tensor(
            experts_vb,
            (num_experts, hidden / 2, 2 * inter),
            &["gate_up_proj.weight", "gate_up_proj"],
            shard(2, rank, world * 2),
            DType::U8,
        )?;
        let up_blocks = Self::get_packed_tensor(
            experts_vb,
            (num_experts, hidden / 2, 2 * inter),
            &["gate_up_proj.weight", "gate_up_proj"],
            shard(2, rank + world, world * 2),
            DType::U8,
        )?;
        let gate_scales = Self::get_packed_tensor(
            experts_vb,
            (num_experts, hidden / 16, 2 * inter),
            &["gate_up_proj.weight_scale", "gate_up_proj_weight_scale"],
            shard(2, rank, world * 2),
            DType::U8,
        )?;
        let up_scales = Self::get_packed_tensor(
            experts_vb,
            (num_experts, hidden / 16, 2 * inter),
            &["gate_up_proj.weight_scale", "gate_up_proj_weight_scale"],
            shard(2, rank + world, world * 2),
            DType::U8,
        )?;

        let gate_global = Self::packed_global_scale(
            experts_vb,
            &["gate_up_proj.weight_scale_2", "gate_up_proj_weight_scale_2"],
            &[
                "gate_up_proj.weight_global_scale",
                "gate_up_proj_weight_global_scale",
            ],
        );
        let gate_input = Self::packed_input_scale(
            experts_vb,
            &["gate_up_proj.input_scale", "gate_up_proj_input_scale"],
            &[
                "gate_up_proj.input_global_scale",
                "gate_up_proj_input_global_scale",
            ],
        );

        let mut fused_blocks = Vec::with_capacity(num_experts);
        let mut fused_scales = Vec::with_capacity(num_experts);
        for expert in 0..num_experts {
            let gate = gate_blocks.get(expert)?.t()?.contiguous()?;
            let up = up_blocks.get(expert)?.t()?.contiguous()?;
            fused_blocks.push(Tensor::cat(&[&gate, &up], 0)?);

            let gate = gate_scales.get(expert)?.t()?.contiguous()?;
            let up = up_scales.get(expert)?.t()?.contiguous()?;
            fused_scales.push(Tensor::cat(&[&gate, &up], 0)?);
        }
        let gate_up = Nvfp4GateUpWeights::Fused(Nvfp4Projection::new(
            Tensor::stack(&fused_blocks, 0)?,
            Tensor::stack(&fused_scales, 0)?,
            vec![gate_global; num_experts],
            vec![gate_input; num_experts],
        )?);

        let down_blocks = Self::get_packed_tensor(
            experts_vb,
            (num_experts, inter / 2, hidden),
            &["down_proj.weight", "down_proj"],
            shard(1, rank, world),
            DType::U8,
        )?;
        let down_scales = Self::get_packed_tensor(
            experts_vb,
            (num_experts, inter / 16, hidden),
            &["down_proj.weight_scale", "down_proj_weight_scale"],
            shard(1, rank, world),
            DType::U8,
        )?;
        let down_global = Self::packed_global_scale(
            experts_vb,
            &["down_proj.weight_scale_2", "down_proj_weight_scale_2"],
            &[
                "down_proj.weight_global_scale",
                "down_proj_weight_global_scale",
            ],
        );
        let down_input = Self::packed_input_scale(
            experts_vb,
            &["down_proj.input_scale", "down_proj_input_scale"],
            &[
                "down_proj.input_global_scale",
                "down_proj_input_global_scale",
            ],
        );

        let mut down_blocks_vec = Vec::with_capacity(num_experts);
        let mut down_scales_vec = Vec::with_capacity(num_experts);
        for expert in 0..num_experts {
            down_blocks_vec.push(down_blocks.get(expert)?.t()?.contiguous()?);
            down_scales_vec.push(down_scales.get(expert)?.t()?.contiguous()?);
        }
        Ok((
            gate_up,
            down_blocks_vec,
            down_scales_vec,
            vec![down_global; num_experts],
            vec![down_input; num_experts],
        ))
    }

    pub fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let quant_method = cfg
            .quantization_config
            .as_ref()
            .map(|q| q.quant_method.clone());
        let gate = linear_no_bias_x(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate"),
            Shard::default(),
            &quant_method,
            &cfg.quantization_config,
            gate_dtype,
            None,
        )?;

        let experts_vb = vb.pp("experts");
        let is_mlx_nvfp4 = cfg
            .quantization_config
            .as_ref()
            .is_some_and(|q| q.is_mlx_nvfp4);
        let has_packed_gate_up = experts_vb.contains_tensor("gate_up_proj")
            || experts_vb.contains_tensor("gate_up_proj.weight")
            || experts_vb.contains_tensor("gate_up_proj_weight_scale_2");

        let mut gate_blocks_vec = Vec::new();
        let mut gate_scales_vec = Vec::new();
        let mut gate_gscales_vec: Vec<f32> = Vec::new();
        let mut gate_iscales_vec: Vec<f32> = Vec::new();
        let mut up_blocks_vec = Vec::new();
        let mut up_scales_vec = Vec::new();
        let mut up_gscales_vec: Vec<f32> = Vec::new();
        let mut up_iscales_vec: Vec<f32> = Vec::new();
        let mut down_blocks_vec = Vec::new();
        let mut down_scales_vec = Vec::new();
        let mut down_gscales_vec: Vec<f32> = Vec::new();
        let mut down_iscales_vec: Vec<f32> = Vec::new();
        let mut packed_gate_up = None;

        if has_packed_gate_up {
            let (gate_up, blocks, scales, global, input) =
                Self::load_packed_experts(cfg, &experts_vb, &comm)?;
            packed_gate_up = Some(gate_up);
            down_blocks_vec = blocks;
            down_scales_vec = scales;
            down_gscales_vec = global;
            down_iscales_vec = input;
        } else {
            for i in 0..num_experts {
                let expert_vb = experts_vb.pp(i.to_string());
                let (gn, un, dn) = resolve_expert_proj_prefix(&expert_vb);

                let gate_proj_vb = expert_vb.pp(gn);
                let packed_name = Self::tensor_name_packed(&gate_proj_vb);
                let scale_name = Self::tensor_name_scale(&gate_proj_vb);
                let sh0 = shard(0, comm.rank(), comm.world_size());

                if is_mlx_nvfp4 {
                    let (blocks, scales) = Self::load_mlx_weight(
                        &gate_proj_vb,
                        moe_cfg.moe_intermediate_size,
                        cfg.hidden_size,
                        sh0,
                    )?;
                    gate_blocks_vec.push(blocks);
                    gate_scales_vec.push(scales);
                    gate_gscales_vec.push(1.0);
                    gate_iscales_vec.push(1.0);
                } else {
                    gate_blocks_vec.push(gate_proj_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                        packed_name,
                        sh0,
                        DType::U8,
                    )?);
                    gate_scales_vec.push(gate_proj_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 16),
                        scale_name,
                        sh0,
                        DType::U8,
                    )?);
                    gate_gscales_vec.push(Self::load_global_scale(&gate_proj_vb));
                    gate_iscales_vec.push(Self::load_input_scale(&gate_proj_vb));
                }

                let up_proj_vb = expert_vb.pp(un);
                let packed_name = Self::tensor_name_packed(&up_proj_vb);
                let scale_name = Self::tensor_name_scale(&up_proj_vb);

                if is_mlx_nvfp4 {
                    let (blocks, scales) = Self::load_mlx_weight(
                        &up_proj_vb,
                        moe_cfg.moe_intermediate_size,
                        cfg.hidden_size,
                        sh0,
                    )?;
                    up_blocks_vec.push(blocks);
                    up_scales_vec.push(scales);
                    up_gscales_vec.push(1.0);
                    up_iscales_vec.push(1.0);
                } else {
                    up_blocks_vec.push(up_proj_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                        packed_name,
                        sh0,
                        DType::U8,
                    )?);
                    up_scales_vec.push(up_proj_vb.get_with_hints_dtype(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size / 16),
                        scale_name,
                        sh0,
                        DType::U8,
                    )?);
                    up_gscales_vec.push(Self::load_global_scale(&up_proj_vb));
                    up_iscales_vec.push(Self::load_input_scale(&up_proj_vb));
                }

                let down_proj_vb = expert_vb.pp(dn);
                let packed_name = Self::tensor_name_packed(&down_proj_vb);
                let scale_name = Self::tensor_name_scale(&down_proj_vb);
                let sh1 = shard(1, comm.rank(), comm.world_size());

                if is_mlx_nvfp4 {
                    let (blocks, scales) = Self::load_mlx_weight(
                        &down_proj_vb,
                        cfg.hidden_size,
                        moe_cfg.moe_intermediate_size,
                        sh1,
                    )?;
                    down_blocks_vec.push(blocks);
                    down_scales_vec.push(scales);
                    down_gscales_vec.push(1.0);
                    down_iscales_vec.push(1.0);
                } else {
                    down_blocks_vec.push(down_proj_vb.get_with_hints_dtype(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / 2),
                        packed_name,
                        sh1,
                        DType::U8,
                    )?);
                    down_scales_vec.push(down_proj_vb.get_with_hints_dtype(
                        (cfg.hidden_size, moe_cfg.moe_intermediate_size / 16),
                        scale_name,
                        sh1,
                        DType::U8,
                    )?);
                    down_gscales_vec.push(Self::load_global_scale(&down_proj_vb));
                    down_iscales_vec.push(Self::load_input_scale(&down_proj_vb));
                }
            }
        }

        let (gate_up, w_size_n) = if let Some(gate_up) = packed_gate_up {
            let w_size_n = match &gate_up {
                Nvfp4GateUpWeights::Fused(projection) => projection.blocks.dim(1)? / 2,
                Nvfp4GateUpWeights::Separate { .. } => unreachable!(),
            };
            (gate_up, w_size_n)
        } else {
            let gate_blocks = Tensor::stack(&gate_blocks_vec, 0)?;
            let gate_scales = Tensor::stack(&gate_scales_vec, 0)?;
            let up_blocks = Tensor::stack(&up_blocks_vec, 0)?;
            let up_scales = Tensor::stack(&up_scales_vec, 0)?;
            let gate =
                Nvfp4Projection::new(gate_blocks, gate_scales, gate_gscales_vec, gate_iscales_vec)?;
            let up = Nvfp4Projection::new(up_blocks, up_scales, up_gscales_vec, up_iscales_vec)?;
            let w_size_n = gate.blocks.dim(1)?;
            if up.blocks.dim(1)? != w_size_n {
                candle::bail!(
                    "NVFP4 MoE gate/up output dimensions differ: gate={}, up={}",
                    w_size_n,
                    up.blocks.dim(1)?
                );
            }
            (Nvfp4GateUpWeights::Separate { gate, up }, w_size_n)
        };

        let dev = match &gate_up {
            Nvfp4GateUpWeights::Fused(projection) => projection.blocks.device().clone(),
            Nvfp4GateUpWeights::Separate { gate, .. } => gate.blocks.device().clone(),
        };

        let down_blocks = Tensor::stack(&down_blocks_vec, 0)?;
        let down_scales = Tensor::stack(&down_scales_vec, 0)?;
        let down_global_scales = Tensor::from_vec(down_gscales_vec, (num_experts,), &dev)?;
        let down_input_scales = Tensor::from_vec(down_iscales_vec, (num_experts,), &dev)?;

        let down_scales_swizzled = maybe_swizzle_nvfp4_scales(&down_scales)?;

        let e_score_correction_bias = vb
            .pp("gate")
            .get_with_hints_dtype(
                num_experts,
                "e_score_correction_bias",
                Shard::default(),
                DType::F32,
            )
            .ok()
            .or_else(|| {
                vb.get_with_hints_dtype(
                    num_experts,
                    "e_score_correction_bias",
                    Shard::default(),
                    DType::F32,
                )
                .ok()
            });

        Ok(Self {
            gate,
            gate_up,
            down_blocks,
            down_scales,
            down_global_scales,
            down_input_scales,
            down_scales_swizzled,
            w_size_n,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            gate_dtype,
            use_sigmoid_scoring: Self::use_sigmoid_routing(&moe_cfg),
            n_group: moe_cfg.n_group.unwrap_or(1),
            topk_group: moe_cfg.topk_group.unwrap_or(1),
            apply_router_weight_on_input: false,
            e_score_correction_bias,
        })
    }

    pub fn new_with_gate(
        cfg: &Config,
        gate_vb: VarBuilder,
        experts_vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        let moe_cfg = qwen_moe_cfg(cfg)?;
        let num_experts = moe_cfg.num_experts.unwrap_or(0);
        if num_experts == 0 {
            candle::bail!("num_experts must be > 0")
        }

        let gate_dtype = if cfg.higher_precision_required() {
            DType::F32
        } else {
            dtype
        };
        let gate = linear_no_bias_x(
            cfg.hidden_size,
            num_experts,
            gate_vb,
            Shard::default(),
            &None,
            &None,
            gate_dtype,
            None,
        )?;

        let has_packed_gate_up = experts_vb.contains_tensor("gate_up_proj")
            || experts_vb.contains_tensor("gate_up_proj.weight")
            || experts_vb.contains_tensor("gate_up_proj_weight_scale_2");
        let mut gate_blocks_vec = Vec::new();
        let mut gate_scales_vec = Vec::new();
        let mut gate_gscales_vec: Vec<f32> = Vec::new();
        let mut gate_iscales_vec: Vec<f32> = Vec::new();
        let mut up_blocks_vec = Vec::new();
        let mut up_scales_vec = Vec::new();
        let mut up_gscales_vec: Vec<f32> = Vec::new();
        let mut up_iscales_vec: Vec<f32> = Vec::new();
        let mut down_blocks_vec = Vec::new();
        let mut down_scales_vec = Vec::new();
        let mut down_gscales_vec: Vec<f32> = Vec::new();
        let mut down_iscales_vec: Vec<f32> = Vec::new();
        let mut packed_gate_up = None;

        if has_packed_gate_up {
            let (gate_up, blocks, scales, global, input) =
                Self::load_packed_experts(cfg, &experts_vb, &comm)?;
            packed_gate_up = Some(gate_up);
            down_blocks_vec = blocks;
            down_scales_vec = scales;
            down_gscales_vec = global;
            down_iscales_vec = input;
        } else {
            for i in 0..num_experts {
                let expert_vb = experts_vb.pp(i.to_string());
                let (gn, un, dn) = resolve_expert_proj_prefix(&expert_vb);

                let gate_proj_vb = expert_vb.pp(gn);
                let packed_name = Self::tensor_name_packed(&gate_proj_vb);
                let scale_name = Self::tensor_name_scale(&gate_proj_vb);
                let sh0 = shard(0, comm.rank(), comm.world_size());

                gate_blocks_vec.push(gate_proj_vb.get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                    packed_name,
                    sh0,
                    DType::U8,
                )?);
                gate_scales_vec.push(gate_proj_vb.get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size / 16),
                    scale_name,
                    sh0,
                    DType::U8,
                )?);
                gate_gscales_vec.push(Self::load_global_scale(&gate_proj_vb));
                gate_iscales_vec.push(Self::load_input_scale(&gate_proj_vb));

                let up_proj_vb = expert_vb.pp(un);
                let packed_name = Self::tensor_name_packed(&up_proj_vb);
                let scale_name = Self::tensor_name_scale(&up_proj_vb);

                up_blocks_vec.push(up_proj_vb.get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size / 2),
                    packed_name,
                    sh0,
                    DType::U8,
                )?);
                up_scales_vec.push(up_proj_vb.get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size / 16),
                    scale_name,
                    sh0,
                    DType::U8,
                )?);
                up_gscales_vec.push(Self::load_global_scale(&up_proj_vb));
                up_iscales_vec.push(Self::load_input_scale(&up_proj_vb));

                let down_proj_vb = expert_vb.pp(dn);
                let packed_name = Self::tensor_name_packed(&down_proj_vb);
                let scale_name = Self::tensor_name_scale(&down_proj_vb);
                let sh1 = shard(1, comm.rank(), comm.world_size());

                down_blocks_vec.push(down_proj_vb.get_with_hints_dtype(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size / 2),
                    packed_name,
                    sh1,
                    DType::U8,
                )?);
                down_scales_vec.push(down_proj_vb.get_with_hints_dtype(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size / 16),
                    scale_name,
                    sh1,
                    DType::U8,
                )?);
                down_gscales_vec.push(Self::load_global_scale(&down_proj_vb));
                down_iscales_vec.push(Self::load_input_scale(&down_proj_vb));
            }
        }

        let (gate_up, w_size_n) = if let Some(gate_up) = packed_gate_up {
            let w_size_n = match &gate_up {
                Nvfp4GateUpWeights::Fused(projection) => projection.blocks.dim(1)? / 2,
                Nvfp4GateUpWeights::Separate { .. } => unreachable!(),
            };
            (gate_up, w_size_n)
        } else {
            let gate_blocks = Tensor::stack(&gate_blocks_vec, 0)?;
            let gate_scales = Tensor::stack(&gate_scales_vec, 0)?;
            let up_blocks = Tensor::stack(&up_blocks_vec, 0)?;
            let up_scales = Tensor::stack(&up_scales_vec, 0)?;
            let gate =
                Nvfp4Projection::new(gate_blocks, gate_scales, gate_gscales_vec, gate_iscales_vec)?;
            let up = Nvfp4Projection::new(up_blocks, up_scales, up_gscales_vec, up_iscales_vec)?;
            let w_size_n = gate.blocks.dim(1)?;
            if up.blocks.dim(1)? != w_size_n {
                candle::bail!(
                    "NVFP4 MoE gate/up output dimensions differ: gate={}, up={}",
                    w_size_n,
                    up.blocks.dim(1)?
                );
            }
            (Nvfp4GateUpWeights::Separate { gate, up }, w_size_n)
        };

        let dev = match &gate_up {
            Nvfp4GateUpWeights::Fused(projection) => projection.blocks.device().clone(),
            Nvfp4GateUpWeights::Separate { gate, .. } => gate.blocks.device().clone(),
        };

        let down_blocks = Tensor::stack(&down_blocks_vec, 0)?;
        let down_scales = Tensor::stack(&down_scales_vec, 0)?;
        let down_global_scales = Tensor::from_vec(down_gscales_vec, (num_experts,), &dev)?;
        let down_input_scales = Tensor::from_vec(down_iscales_vec, (num_experts,), &dev)?;

        let down_scales_swizzled = maybe_swizzle_nvfp4_scales(&down_scales)?;

        Ok(Self {
            gate,
            gate_up,
            down_blocks,
            down_scales,
            down_global_scales,
            down_input_scales,
            down_scales_swizzled,
            w_size_n,
            act: get_hidden_act(cfg),
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            gate_dtype,
            use_sigmoid_scoring: Self::use_sigmoid_routing(&moe_cfg),
            n_group: moe_cfg.n_group.unwrap_or(1),
            topk_group: moe_cfg.topk_group.unwrap_or(1),
            apply_router_weight_on_input: false,
            e_score_correction_bias: None,
        })
    }

    pub fn set_sigmoid_routing(&mut self) {
        self.use_sigmoid_scoring = true;
    }

    pub fn set_apply_router_weight_on_input(&mut self, v: bool) {
        self.apply_router_weight_on_input = v;
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let gate_input = if xs.dtype() != self.gate_dtype {
            Cow::Owned(xs.to_dtype(self.gate_dtype)?)
        } else {
            Cow::Borrowed(xs)
        };
        let router_logits = self.gate.forward(&gate_input)?;

        let (mut topk_weights, topk_ids): (Tensor, Tensor) = if self.use_sigmoid_scoring {
            let scores = candle_nn::ops::sigmoid(&router_logits.to_dtype(DType::F32)?)?;
            let scores_for_choice = if let Some(bias) = &self.e_score_correction_bias {
                scores.broadcast_add(&bias.to_dtype(DType::F32)?)?
            } else {
                scores.clone()
            };

            let topk_indices = if self.n_group > 1 {
                let num_tokens = scores_for_choice.dim(0)?;
                let num_experts = scores_for_choice.dim(1)?;
                if num_experts % self.n_group != 0 {
                    candle::bail!(
                        "MoE routing requires num_experts ({num_experts}) divisible by n_group ({})",
                        self.n_group
                    );
                }
                if self.topk_group > self.n_group {
                    candle::bail!(
                        "MoE routing requires topk_group ({}) <= n_group ({})",
                        self.topk_group,
                        self.n_group
                    );
                }
                let experts_per_group = num_experts / self.n_group;
                if experts_per_group * self.topk_group < self.num_experts_per_tok {
                    candle::bail!(
                        "MoE routing selected-group capacity ({}) is smaller than num_experts_per_tok ({})",
                        experts_per_group * self.topk_group,
                        self.num_experts_per_tok
                    );
                }
                let grouped =
                    scores_for_choice.reshape((num_tokens, self.n_group, experts_per_group))?;
                let top2_idx = select_topk_indices(&grouped, experts_per_group.min(2), is_prefill)?;
                let top2_vals = grouped.gather(&top2_idx, D::Minus1)?;
                let group_scores = top2_vals.sum(D::Minus1)?;
                let group_idx = select_topk_indices(&group_scores, self.topk_group, is_prefill)?;
                let group_mask = group_scores.zeros_like()?.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(DType::F32)?,
                    1,
                )?;
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .broadcast_as((num_tokens, self.n_group, experts_per_group))?
                    .reshape((num_tokens, num_experts))?;
                let masked = scores_for_choice.broadcast_mul(&score_mask)?;
                select_topk_indices(&masked, self.num_experts_per_tok, is_prefill)?
            } else {
                select_topk_indices(&scores_for_choice, self.num_experts_per_tok, is_prefill)?
            };

            let topk_weights = scores.gather(&topk_indices, D::Minus1)?;
            (topk_weights, topk_indices.to_dtype(DType::U32)?)
        } else {
            let mut logits = router_logits.to_dtype(DType::F32)?;
            if let Some(bias) = &self.e_score_correction_bias {
                logits = logits.broadcast_add(&bias.to_dtype(DType::F32)?)?;
            }
            attention_rs::topk::topk_softmax(&logits, self.num_experts_per_tok)?
        };

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }

        self.forward_with_routing(xs, topk_weights, topk_ids, is_prefill)
    }

    pub fn forward_with_routing(
        &self,
        xs: &Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;

        let xs = if xs.dtype() == DType::F32 {
            xs.to_dtype(self.dtype)?
        } else {
            xs.clone()
        };

        let xs = if self.apply_router_weight_on_input {
            let w = topk_weights.to_dtype(xs.dtype())?;
            xs.broadcast_mul(&w)?
        } else {
            xs
        };

        let pre_sorted = presorted_expert_assignments(&topk_ids, is_prefill)?;
        let pre_sorted_refs = pre_sorted.as_ref().map(|(a, b)| (a, b));

        let gate_up = match &self.gate_up {
            Nvfp4GateUpWeights::Fused(projection) => moe::moe_gemm_nvfp4(
                &xs,
                &projection.blocks,
                &projection.scales,
                &projection.global_scales,
                Some(&projection.input_scales),
                None,
                &topk_ids,
                pre_sorted_refs,
                is_prefill,
                None,
                projection.scales_swizzled.as_ref(),
            )?,
            Nvfp4GateUpWeights::Separate { gate, up } => {
                let gate_output = moe::moe_gemm_nvfp4(
                    &xs,
                    &gate.blocks,
                    &gate.scales,
                    &gate.global_scales,
                    Some(&gate.input_scales),
                    None,
                    &topk_ids,
                    pre_sorted_refs,
                    is_prefill,
                    None,
                    gate.scales_swizzled.as_ref(),
                )?;
                let up_output = moe::moe_gemm_nvfp4(
                    &xs,
                    &up.blocks,
                    &up.scales,
                    &up.global_scales,
                    Some(&up.input_scales),
                    None,
                    &topk_ids,
                    pre_sorted_refs,
                    is_prefill,
                    None,
                    up.scales_swizzled.as_ref(),
                )?;
                Tensor::cat(&[&gate_output, &up_output], 2)?
            }
        };

        let down_inputs = gated_activation(&gate_up, self.w_size_n, &self.act)?;

        let down_topk_weights = if self.apply_router_weight_on_input {
            None
        } else {
            Some(&topk_weights)
        };

        let mut ys = moe::moe_gemm_nvfp4(
            &down_inputs,
            &self.down_blocks,
            &self.down_scales,
            &self.down_global_scales,
            Some(&self.down_input_scales),
            None,
            &topk_ids,
            pre_sorted_refs,
            is_prefill,
            down_topk_weights,
            self.down_scales_swizzled.as_ref(),
        )?
        .reshape((num_tokens, self.num_experts_per_tok, hidden_dim))?
        .sum(1)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys.to_dtype(self.dtype)?)
    }
}
