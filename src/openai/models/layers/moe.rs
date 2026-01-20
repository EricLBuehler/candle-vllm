use crate::candle::quantized::QTensor;
use crate::openai::distributed::shard;
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, VarBuilder};
use crate::openai::models::linear::linear_no_bias;
use crate::openai::models::linear::Linear;
use crate::openai::models::{Config, MoEConfig};
use attention_rs::moe;
use candle::{DType, Module, Result, Tensor, D};
use candle_core as candle;
use candle_core::quantized::GgmlDType;
use candle_nn::var_builder::Shard;
use std::rc::Rc;

#[allow(dead_code)]
pub struct FusedMoe {
    gate: Linear,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoe {
    pub fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = if let Some(MoEConfig::QwenMoE(moe_cfg)) = &cfg.moe_config {
            moe_cfg.clone()
        } else {
            candle::bail!("Expected QwenMoEConfig")
        };
        let num_experts = moe_cfg.num_experts.unwrap();

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

        let experts_vb = vb.pp("experts");
        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        //pack experts
        for i in 0..num_experts {
            let experts_vb = experts_vb.pp(format!("{}", i).as_str());
            let (gate_expert, up_expert, down_expert) = {
                // n x k format
                let gate_expert = experts_vb.pp("gate_proj").get_with_hints(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                )?;
                let up_expert = experts_vb.pp("up_proj").get_with_hints(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                )?;
                let down_expert = experts_vb.pp("down_proj").get_with_hints(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    shard(1, comm.rank(), comm.world_size()),
                )?;

                (gate_expert, up_expert, down_expert)
            };

            gate_experts.push(gate_expert);
            up_experts.push(up_expert);
            down_experts.push(down_expert);
        }

        let gate_w = Tensor::stack(&gate_experts, 0)?;
        let up_w = Tensor::stack(&up_experts, 0)?;
        let down_w = Tensor::stack(&down_experts, 0)?;
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_w,
            up_w,
            down_w,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
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

        //out (M, top_k, N)
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

        //(M * top_k, N // 2)
        let down_inputs = (up * gate.apply(&self.act)?)?;

        //view(M, top_k, K) -> sum -> (M, K)
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
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
    dtype: DType,
}

impl FusedMoeISQ {
    pub fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>, dtype: DType) -> Result<Self> {
        let moe_cfg = if let Some(MoEConfig::QwenMoE(moe_cfg)) = &cfg.moe_config {
            moe_cfg.clone()
        } else {
            candle::bail!("Expected QwenMoEConfig")
        };

        let num_experts = moe_cfg.num_experts.unwrap();

        let quant_type = match cfg.quant.as_ref().unwrap().as_str() {
            "q4_0" => GgmlDType::Q4_0,
            "q4_1" => GgmlDType::Q4_1,
            "q5_0" => GgmlDType::Q5_0,
            "q5_1" => GgmlDType::Q5_1,
            "q8_0" => GgmlDType::Q8_0,
            "q2k" => GgmlDType::Q2K,
            "q3k" => GgmlDType::Q3K,
            "q4k" => GgmlDType::Q4K,
            "q5k" => GgmlDType::Q5K,
            "q6k" => GgmlDType::Q6K,
            _ => panic!("Unsupported GGML data type!"),
        };

        let block_size = quant_type.block_size();

        let ws = vb.pp("gate").get_with_hints_dtype(
            (num_experts, cfg.hidden_size),
            "weight",
            Shard::default(),
            DType::F32,
        )?;
        let gate = Linear::new(ws, None);

        let experts_vb = vb.pp("experts");
        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);

        let moe_intermediate_chunk =
            if moe_cfg.moe_intermediate_size / comm.world_size() % block_size != 0 {
                ((moe_cfg.moe_intermediate_size / comm.world_size() + block_size - 1) / block_size)
                    * block_size
            } else {
                moe_cfg.moe_intermediate_size / comm.world_size()
            };

        //pack experts
        for i in 0..num_experts {
            let experts_vb = experts_vb.pp(format!("{}", i).as_str());
            let (gate_expert, up_expert, down_expert) = if moe_cfg.moe_intermediate_size
                / comm.world_size()
                % block_size
                != 0
            {
                let gate_expert = experts_vb.pp("gate_proj").get_with_hints(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    Shard::default(),
                )?;
                let up_expert = experts_vb.pp("up_proj").get_with_hints(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    Shard::default(),
                )?;
                let down_expert = experts_vb.pp("down_proj").get_with_hints(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    Shard::default(),
                )?;

                let (gate_expert, up_expert, down_expert) = if comm.rank() * moe_intermediate_chunk
                    + moe_intermediate_chunk
                    < moe_cfg.moe_intermediate_size
                {
                    (
                        gate_expert.narrow(
                            0,
                            comm.rank() * moe_intermediate_chunk,
                            moe_intermediate_chunk,
                        )?,
                        up_expert.narrow(
                            0,
                            comm.rank() * moe_intermediate_chunk,
                            moe_intermediate_chunk,
                        )?,
                        down_expert.narrow(
                            1,
                            comm.rank() * moe_intermediate_chunk,
                            moe_intermediate_chunk,
                        )?,
                    )
                } else {
                    let last_remain_size =
                        moe_cfg.moe_intermediate_size - comm.rank() * moe_intermediate_chunk;
                    assert!(last_remain_size > 0 && last_remain_size % block_size == 0,
                        "Unable to split moe_intermediate_size {} into {} ranks under block_size of {}! \n \
                        \t*****Tips: you may try these gglm types: `q8_0` (recommend), `q4_0`, `q4_1`, `q5_0`, `q5_1` (with smaller block_size 32)",
                        moe_cfg.moe_intermediate_size,
                        comm.world_size(),
                        block_size
                    );
                    let gate_expert = gate_expert.narrow(
                        0,
                        comm.rank() * moe_intermediate_chunk,
                        last_remain_size,
                    )?;
                    let up_expert = up_expert.narrow(
                        0,
                        comm.rank() * moe_intermediate_chunk,
                        last_remain_size,
                    )?;
                    let down_expert = down_expert.narrow(
                        1,
                        comm.rank() * moe_intermediate_chunk,
                        last_remain_size,
                    )?;
                    (gate_expert, up_expert, down_expert)
                };
                (gate_expert, up_expert, down_expert)
            } else {
                let gate_expert = experts_vb.pp("gate_proj").get_with_hints(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                )?;
                let up_expert = experts_vb.pp("up_proj").get_with_hints(
                    (moe_cfg.moe_intermediate_size, cfg.hidden_size),
                    "weight",
                    shard(0, comm.rank(), comm.world_size()),
                )?;
                let down_expert = experts_vb.pp("down_proj").get_with_hints(
                    (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                    "weight",
                    shard(1, comm.rank(), comm.world_size()),
                )?;
                (gate_expert, up_expert, down_expert)
            };

            gate_experts.push(gate_expert);
            up_experts.push(up_expert);
            down_experts.push(down_expert);
        }
        let gate_experts = Tensor::stack(&gate_experts, 0)?;
        let up_experts = Tensor::stack(&up_experts, 0)?;
        let down_experts = Tensor::stack(&down_experts, 0)?;
        // in-situ quantization for using fused moe kernel
        let gate_experts = QTensor::quantize(&gate_experts, quant_type).unwrap();
        let up_experts = QTensor::quantize(&up_experts, quant_type).unwrap();

        //down_experts requires higher precision
        let down_experts = QTensor::quantize(&down_experts, GgmlDType::Q8_0).unwrap();
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_experts,
            up_experts,
            down_experts,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
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
        quant_cfg: &crate::openai::models::QuantConfig,
    ) -> Result<Self> {
        let moe_cfg = if let Some(MoEConfig::QwenMoE(moe_cfg)) = &cfg.moe_config {
            moe_cfg.clone()
        } else {
            candle::bail!("Expected QwenMoEConfig")
        };
        let num_experts = moe_cfg.num_experts.unwrap();

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

        // Per-expert loading
        let mut gate_experts = Vec::with_capacity(num_experts);
        let mut gate_experts_scale = Vec::with_capacity(num_experts);
        let mut up_experts = Vec::with_capacity(num_experts);
        let mut up_experts_scale = Vec::with_capacity(num_experts);
        let mut down_experts = Vec::with_capacity(num_experts);
        let mut down_experts_scale = Vec::with_capacity(num_experts);

        for i in 0..num_experts {
            let expert_vb = experts_vb.pp(format!("{}", i).as_str());

            // Gate proj
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
                                "FusedMoeFp8: Missing weight_scale for expert {} gate_proj",
                                i
                            )
                            .into(),
                        )
                    })?,
            };

            // Up proj
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
                            format!("FusedMoeFp8: Missing weight_scale for expert {} up_proj", i)
                                .into(),
                        )
                    })?,
            };

            // Down proj
            let down_sn = (cfg.hidden_size + by - 1) / by;
            let down_sk = (moe_cfg.moe_intermediate_size + bx - 1) / bx;
            let down_weight = expert_vb.pp("down_proj").get_with_hints_dtype(
                (cfg.hidden_size, moe_cfg.moe_intermediate_size),
                "weight",
                shard(1, comm.rank(), comm.world_size()),
                DType::U8,
            )?;
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
                                "FusedMoeFp8: Missing weight_scale for expert {} down_proj",
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

        Ok(Self {
            gate,
            gate_experts: Tensor::stack(&gate_experts, 0)?,
            gate_experts_scale: Tensor::stack(&gate_experts_scale, 0)?,
            up_experts: Tensor::stack(&up_experts, 0)?,
            up_experts_scale: Tensor::stack(&up_experts_scale, 0)?,
            down_experts: Tensor::stack(&down_experts, 0)?,
            down_experts_scale: Tensor::stack(&down_experts_scale, 0)?,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
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

        let gate = moe::moe_gemm_fp8(
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

        let up = moe::moe_gemm_fp8(
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

        let mut ys = moe::moe_gemm_fp8(
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
