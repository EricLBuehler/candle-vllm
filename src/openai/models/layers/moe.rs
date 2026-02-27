use crate::candle::quantized::QTensor;
use crate::openai::distributed::{shard, AllReduce, Comm, VarBuilder};
use crate::openai::models::linear::{linear_no_bias, Linear};
use crate::openai::models::{Config, MoEConfig, QuantConfig, QwenMoEConfig};
use attention_rs::moe;
use attention_rs::moe::moe_gemm_fp8;
use candle::{DType, Module, Result, Tensor, D};
use candle_core as candle;
use candle_core::quantized::GgmlDType;
use candle_nn::var_builder::Shard;
use std::rc::Rc;

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

fn arch_name(cfg: &Config) -> &str {
    cfg.architectures
        .as_ref()
        .and_then(|a| a.first())
        .map(|s| s.as_str())
        .unwrap_or("")
}

fn resolve_packed_gate_up_layout(cfg: &Config) -> Result<PackedGateUpLayout> {
    let arch = arch_name(cfg);

    // Qwen3.5 MoE / Qwen3-Next store gate_up as [experts, 2*intermediate, hidden].
    if matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
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

    // Qwen3.5 MoE / Qwen3-Next store down_proj as [experts, hidden, intermediate].
    if matches!(
        arch,
        "Qwen3_5MoeForCausalLM"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForCausalLM"
            | "Qwen3NextForConditionalGeneration"
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

fn get_packed_weight_3d_dtype(
    experts_vb: &VarBuilder,
    tensor_name: &str,
    shape: (usize, usize, usize),
    sh: Shard,
    dtype: DType,
) -> Result<Tensor> {
    experts_vb
        .pp(tensor_name)
        .get_with_hints_dtype(shape, "weight", sh, dtype)
        .or_else(|_| experts_vb.get_with_hints_dtype(shape, tensor_name, sh, dtype))
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
        let gate = expert_vb.pp("gate_proj").get_with_hints(
            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
            "weight",
            shard(0, comm.rank(), comm.world_size()),
        )?;
        let up = expert_vb.pp("up_proj").get_with_hints(
            (moe_cfg.moe_intermediate_size, cfg.hidden_size),
            "weight",
            shard(0, comm.rank(), comm.world_size()),
        )?;
        let down = expert_vb.pp("down_proj").get_with_hints(
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

#[allow(dead_code)]
pub struct FusedMoe {
    gate: Linear,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    act: candle_nn::Activation,
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
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_w,
            up_w,
            down_w,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
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

        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        let gate = moe::moe_gemm(
            &xs,
            &self.gate_w,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?;

        let up = moe::moe_gemm(
            &xs,
            &self.up_w,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            is_prefill,
        )?;

        let down_inputs = (up * gate.apply(&self.act)?)?;

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

        let mut quant_type = match cfg.quant.as_ref().unwrap().as_str() {
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
                    let gate = expert_vb.pp("gate_proj").get_with_hints(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                        Shard::default(),
                    )?;
                    let up = expert_vb.pp("up_proj").get_with_hints(
                        (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                        "weight",
                        Shard::default(),
                    )?;
                    let down = expert_vb.pp("down_proj").get_with_hints(
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

        let gate_experts = QTensor::quantize(&gate_experts, quant_type)?;
        let up_experts = QTensor::quantize(&up_experts, quant_type)?;
        let down_experts = QTensor::quantize(&down_experts, GgmlDType::Q8_0)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
            dtype,
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
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
        if let Some(routed_scaling_factor) = self.routed_scaling_factor {
            topk_weights = (topk_weights * routed_scaling_factor)?;
        }
        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

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
        if ys.dtype() != self.dtype {
            ys = ys.to_dtype(self.dtype)?;
        }
        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        ys.to_dtype(original_dtype)
    }
}

/// FP8 Mixture of Experts layer with block-wise scales
pub struct FusedMoeFp8 {
    gate: Linear,
    gate_experts: Tensor,
    gate_experts_scale: Tensor,
    up_experts: Tensor,
    up_experts_scale: Tensor,
    down_experts: Tensor,
    down_experts_scale: Tensor,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    routed_scaling_factor: Option<f64>,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
    block_size: Vec<usize>,
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

        let gate_ws = vb.pp("gate").get_with_hints_dtype(
            (num_experts, cfg.hidden_size),
            "weight",
            Shard::default(),
            dtype,
        )?;
        let gate = Linear::new(gate_ws, None);

        let experts_vb = vb.pp("experts");

        let (
            gate_experts,
            gate_experts_scale,
            up_experts,
            up_experts_scale,
            down_experts,
            down_experts_scale,
        ) = if has_packed_gate_up(&experts_vb) {
            // Packed (Qwen3.5/Qwen3-Next FP8 checkpoints).
            let gate_weight = get_packed_weight_3d_dtype(
                &experts_vb,
                "gate_up_proj",
                (
                    num_experts,
                    cfg.hidden_size,
                    moe_cfg.moe_intermediate_size * 2,
                ),
                shard(2, comm.rank(), comm.world_size() * 2),
                DType::U8,
            )?
            .t()?
            .contiguous()?;

            let up_weight = get_packed_weight_3d_dtype(
                &experts_vb,
                "gate_up_proj",
                (
                    num_experts,
                    cfg.hidden_size,
                    moe_cfg.moe_intermediate_size * 2,
                ),
                shard(2, comm.rank() + comm.world_size(), comm.world_size() * 2),
                DType::U8,
            )?
            .t()?
            .contiguous()?;

            let scale_n = (cfg.hidden_size + by - 1) / by;
            let scale_k = (moe_cfg.moe_intermediate_size * 2 + bx - 1) / bx;

            let gate_up_scale = experts_vb
                .get_with_hints_dtype(
                    (num_experts, scale_n, scale_k),
                    "gate_up_proj_scale_inv",
                    Shard::default(),
                    DType::F32,
                )
                .or_else(|_| {
                    experts_vb.pp("gate_up_proj").get_with_hints_dtype(
                        (num_experts, scale_n, scale_k),
                        "weight_scale_inv",
                        Shard::default(),
                        DType::F32,
                    )
                })?;

            let inter_blocks = moe_cfg.moe_intermediate_size / bx;
            let local_inter_blocks = inter_blocks / comm.world_size();
            let start_blocks = comm.rank() * local_inter_blocks;

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

            let down_weight = get_packed_weight_3d_dtype(
                &experts_vb,
                "down_proj",
                (num_experts, moe_cfg.moe_intermediate_size, cfg.hidden_size),
                shard(1, comm.rank(), comm.world_size()),
                DType::U8,
            )?
            .t()?
            .contiguous()?;

            let down_s = experts_vb
                .get_with_hints_dtype(
                    (num_experts, scale_k / 2, scale_n),
                    "down_proj_scale_inv",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::F32,
                )
                .or_else(|_| {
                    experts_vb.pp("down_proj").get_with_hints_dtype(
                        (num_experts, scale_k / 2, scale_n),
                        "weight_scale_inv",
                        shard(1, comm.rank(), comm.world_size()),
                        DType::F32,
                    )
                })?
                .t()?
                .contiguous()?;

            (gate_weight, gate_s, up_weight, up_s, down_weight, down_s)
        } else {
            // Per-expert layout.
            let mut gate_experts = Vec::with_capacity(num_experts);
            let mut gate_experts_scale = Vec::with_capacity(num_experts);
            let mut up_experts = Vec::with_capacity(num_experts);
            let mut up_experts_scale = Vec::with_capacity(num_experts);
            let mut down_experts = Vec::with_capacity(num_experts);
            let mut down_experts_scale = Vec::with_capacity(num_experts);

            for i in 0..num_experts {
                let expert_vb = experts_vb.pp(format!("{}", i).as_str());

                let gate_weight = expert_vb.pp("gate_proj").get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let sn = (moe_cfg.moe_intermediate_size + by - 1) / by;
                let sk = (cfg.hidden_size + bx - 1) / bx;
                let gate_s = match expert_vb.pp("gate_proj").get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb
                        .pp("gate_proj")
                        .get_with_hints_dtype(
                            (sn, sk),
                            "weight_scale_inv",
                            shard(0, comm.rank(), comm.world_size()),
                            DType::F32,
                        )
                        .map_err(|_| {
                            candle::Error::Msg(
                                format!(
                                    "FusedMoeFp8: Missing weight_scale/inv for expert {} gate_proj",
                                    i
                                )
                                .into(),
                            )
                        })?,
                };

                let up_weight = expert_vb.pp("up_proj").get_with_hints_dtype(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let up_s = match expert_vb.pp("up_proj").get_with_hints_dtype(
                    (sn, sk),
                    "weight_scale",
                    shard(0, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb
                        .pp("up_proj")
                        .get_with_hints_dtype(
                            (sn, sk),
                            "weight_scale_inv",
                            shard(0, comm.rank(), comm.world_size()),
                            DType::F32,
                        )
                        .map_err(|_| {
                            candle::Error::Msg(
                                format!(
                                    "FusedMoeFp8: Missing weight_scale/inv for expert {} up_proj",
                                    i
                                )
                                .into(),
                            )
                        })?,
                };

                let down_weight = expert_vb.pp("down_proj").get_with_hints_dtype(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::U8,
                )?;
                let down_sn = (cfg.hidden_size + by - 1) / by;
                let down_sk = (moe_cfg.moe_intermediate_size + bx - 1) / bx;
                let down_s = match expert_vb.pp("down_proj").get_with_hints_dtype(
                    (down_sn, down_sk),
                    "weight_scale",
                    shard(1, comm.rank(), comm.world_size()),
                    DType::F32,
                ) {
                    Ok(s) => s,
                    Err(_) => expert_vb
                        .pp("down_proj")
                        .get_with_hints_dtype(
                            (down_sn, down_sk),
                            "weight_scale_inv",
                            shard(1, comm.rank(), comm.world_size()),
                            DType::F32,
                        )
                        .map_err(|_| {
                            candle::Error::Msg(
                                format!(
                                    "FusedMoeFp8: Missing weight_scale/inv for expert {} down_proj",
                                    i
                                )
                                .into(),
                            )
                        })?,
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

        Ok(Self {
            gate,
            gate_experts,
            gate_experts_scale,
            up_experts,
            up_experts_scale,
            down_experts,
            down_experts_scale,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            routed_scaling_factor: moe_cfg.routed_scaling_factor,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm.clone()),
            world_size: comm.world_size(),
            dtype,
            block_size: vec![by, bx],
        })
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
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

        let (expert_ids, sorted_token_ids) = if is_prefill {
            #[cfg(feature = "cuda")]
            {
                use attention_rs::sort::ArgSortOp;
                topk_ids.flatten_all()?.sort(true)?
            }
            #[cfg(not(feature = "cuda"))]
            topk_ids.flatten_all()?.sort_last_dim(true)?
        } else {
            topk_ids.flatten_all()?.sort_last_dim(true)?
        };

        let xs = if xs.dtype() == DType::F32 {
            xs.to_dtype(DType::BF16)?
        } else {
            xs.clone()
        };

        let gate = moe_gemm_fp8(
            &xs,
            &self.gate_experts,
            &self.gate_experts_scale,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            self.block_size[0],
            self.block_size[1],
            is_prefill,
        )?;

        let up = moe_gemm_fp8(
            &xs,
            &self.up_experts,
            &self.up_experts_scale,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.num_experts_per_tok,
            self.block_size[0],
            self.block_size[1],
            is_prefill,
        )?;

        let down_inputs = (up * gate.apply(&self.act)?)?;

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
        Ok(ys.to_dtype(self.dtype)?)
    }
}
