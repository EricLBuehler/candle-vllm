use super::{
    attention::Attention, rotary_emb::ScalingRotaryEmbedding, Config, InputMetadata, MoEConfig,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
use crate::candle::quantized::QTensor;
use crate::openai::distributed::shard;
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{
    embedding, rms_norm, Comm, ReplicatedLinear, TensorParallelColumnLinear,
    TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::linear::LinearX as Linear;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::openai::models::QwenMoEConfig;
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_core as candle;
use candle_core::quantized::GgmlDType;
use candle_core::quantized::QMatMul;
use candle_nn::var_builder::Shard;
use candle_nn::RmsNorm;
use parking_lot::RwLock;
use std::iter::zip;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

impl Qwen3MoE {
    pub fn load_config(filename: &PathBuf, isq: Option<String>) -> Result<Config> {
        let mut config = Config::load_config(filename.clone())?;
        config.head_dim = Some(
            config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads),
        );
        config.num_key_value_heads = Some(
            config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
        );
        config.max_seq_len = config.max_position_embeddings.unwrap_or(config.max_seq_len);
        config.attention_bias = Some(config.attention_bias.unwrap_or(true));
        if config.quantization_config.is_some() {
            config.quant = Some(
                config
                    .quantization_config
                    .as_ref()
                    .unwrap()
                    .quant_method
                    .clone(),
            );
        } else if isq.is_some() {
            config.quant = Some(isq.unwrap().to_string());
        }

        match std::fs::read(filename) {
            Ok(f) => {
                let cfg: QwenMoEConfig =
                    serde_json::from_slice(&f).map_err(candle_core::Error::wrap)?;
                config.moe_config = Some(MoEConfig::QwenMoE(cfg));
            }
            Err(e) => panic!("Unable to load MoE config from file {:?}!", e),
        }
        Ok(config)
    }
}

struct Mlp {
    gate_proj: TensorParallelColumnLinear,
    up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, intermediate_size: usize, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        // let intermediate_sz = cfg.intermediate_size;

        let gate_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            intermediate_size,
            false,
            vb.pp("gate_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        let up_proj = TensorParallelColumnLinear::load_with_hints(
            hidden_sz,
            intermediate_size,
            false,
            vb.pp("up_proj"),
            comm.clone(),
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        let down_proj = TensorParallelRowLinear::load_with_hints(
            intermediate_size,
            hidden_sz,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.act_fn.forward(&self.gate_proj.forward(xs)?)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(&lhs * &rhs)?)
    }
}

struct Moe {
    gate: Linear,
    experts: Vec<Mlp>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl Moe {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let moe_cfg = if let Some(MoEConfig::QwenMoE(moe_cfg)) = &cfg.moe_config {
            moe_cfg.clone()
        } else {
            candle::bail!("Expected QwenMoEConfig")
        };
        let num_experts = moe_cfg.num_experts.unwrap();
        let ws = vb.pp("gate").get_with_hints_dtype(
            (num_experts, cfg.hidden_size),
            "weight",
            Shard::default(),
            DType::F32,
        )?;
        let gate = Linear::new(ws, None, &None, &None);

        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(Mlp::new(
                cfg,
                moe_cfg.moe_intermediate_size,
                experts_vb.pp(format!("{}", i).as_str()).clone(),
                comm.clone(),
            )?);
        }

        Ok(Self {
            gate,
            experts,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let experts_per_tok = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let routing_weights = routing_weights.gather(&experts_per_tok, D::Minus1)?;

        let routing_weights = routing_weights.to_vec2::<f32>()?;
        let experts_per_tok = experts_per_tok.to_vec2::<u32>()?;
        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_experts = vec![vec![]; self.experts.len()];
        for (row_idx, (rw, expert_idxs)) in routing_weights
            .iter()
            .zip(experts_per_tok.iter())
            .enumerate()
        {
            let sum_rw = rw.iter().sum::<f32>();
            for (&rw, &expert_idx) in rw.iter().zip(expert_idxs.iter()) {
                top_x[expert_idx as usize].push(row_idx as u32);
                let rw = if self.norm_topk_prob { rw / sum_rw } else { rw };
                selected_experts[expert_idx as usize].push(rw)
            }
        }

        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }
            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_experts =
                Tensor::new(selected_experts[expert_idx].as_slice(), xs.device())?
                    .reshape(((), 1))?
                    .to_dtype(xs.dtype())?;
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
            let current_hidden_states = expert_layer
                .forward(&current_state.unsqueeze(0)?)?
                .squeeze(0)?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&selected_experts)?;
            ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
        }
        ys = ys.reshape((num_tokens, hidden_dim))?;
        Ok(ys)
    }
}

struct FusedMoe {
    gate: Linear,
    gate_up_experts: QMatMul,
    expert_size_n: usize,
    down_experts: QMatMul,
    act: candle_nn::Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
    all_reduce: AllReduce,
    world_size: usize,
}

impl FusedMoe {
    fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
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
        let gate = Linear::new(ws, None, &None, &None);

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

        // pack gate_proj and up_proj
        let gate_up_experts = Tensor::cat(&[gate_experts, up_experts], candle_core::D::Minus2)?;
        let expert_size_n = gate_up_experts.dim(1)?;

        // in-situ quantization for using fused moe kernel
        let qtensor = QTensor::quantize(&gate_up_experts, quant_type).unwrap();
        let gate_up_experts = QMatMul::QTensor(Arc::new(qtensor));

        //down_experts requires higher precision
        let qtensor = QTensor::quantize(&down_experts, GgmlDType::Q8_0).unwrap();
        let down_experts = QMatMul::QTensor(Arc::new(qtensor));
        let world_size = comm.world_size();

        Ok(Self {
            gate,
            gate_up_experts,
            expert_size_n,
            down_experts,
            act: candle_nn::Activation::Silu,
            norm_topk_prob: moe_cfg.norm_topk_prob,
            num_experts_per_tok: moe_cfg.num_experts_per_tok,
            all_reduce: AllReduce::new(comm),
            world_size,
        })
    }

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
            let gate_up = self.gate_up_experts.indexed_moe_forward(&xs, &indices)?;
            let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.expert_size_n / 2)?;
            let up = gate_up.narrow(
                candle_core::D::Minus1,
                self.expert_size_n / 2,
                self.expert_size_n / 2,
            )?;
            self.down_experts
                .indexed_moe_forward(&(up * gate.apply(&self.act)?)?, &indices)?
        };
        let mut ys = ys
            .broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((num_tokens, hidden_dim))?
            .to_dtype(original_dtype)?;

        if self.world_size > 1 {
            ys = self.all_reduce.apply(&ys)?;
        }
        Ok(ys)
    }
}

enum MoeOrMlp {
    Moe(Moe),
    FusedMoe(FusedMoe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::Moe(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    shared_gate: Option<Linear>,
    shared_expert: Option<Mlp>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        comm: Rc<Comm>,
        dtype: DType,
        layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            comm.clone(),
            cfg.sliding_window,
        )?;

        let moe_cfg = if let Some(MoEConfig::QwenMoE(moe_cfg)) = &cfg.moe_config {
            moe_cfg.clone()
        } else {
            candle::bail!("Expected QwenMoEConfig")
        };

        let mlp = if !moe_cfg
            .mlp_only_layers
            .as_ref()
            .unwrap_or(&Vec::<usize>::new())
            .contains(&layer_idx)
            && (moe_cfg.num_experts.unwrap_or(0) > 0
                && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap_or(1) == 0)
        {
            if cfg.quant.is_some() {
                MoeOrMlp::FusedMoe(FusedMoe::new(cfg, vb.pp("mlp").clone(), comm.clone())?)
            } else {
                MoeOrMlp::Moe(Moe::new(cfg, vb.pp("mlp").clone(), comm.clone())?)
            }
        } else {
            let mlp = Mlp::new(
                cfg,
                cfg.intermediate_size,
                vb.pp("mlp").clone(),
                comm.clone(),
            )?;

            MoeOrMlp::Mlp(mlp)
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        //shared experts weights in Qwen2 MoE models
        let (shared_gate, shared_expert) =
            if let Some(intermediate_size) = moe_cfg.shared_expert_intermediate_size {
                if intermediate_size > 0 {
                    let ws = vb.pp("mlp.shared_expert_gate").get_with_hints_dtype(
                        (1, cfg.hidden_size),
                        "weight",
                        Default::default(),
                        dtype,
                    )?;

                    let shared_gate = Linear::new(ws, None, &None, &None);

                    let mlp = Mlp::new(
                        cfg,
                        intermediate_size,
                        vb.pp("mlp.shared_expert").clone(),
                        comm.clone(),
                    )?;
                    (Some(shared_gate), Some(mlp))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

        Ok(Self {
            self_attn,
            mlp,
            shared_gate,
            shared_expert,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        //shared experts for Qwen2 MoE models
        let shared_output = match (&self.shared_gate, &self.shared_expert) {
            (Some(shared_gate), Some(shared_expert)) => {
                let gate = candle_nn::ops::sigmoid(&shared_gate.forward(&xs)?)?;
                let shared_output = shared_expert.forward(&xs)?;
                Some(gate.broadcast_mul(&shared_output)?)
            }
            _ => None,
        };
        let mlp_output = self.mlp.forward(&xs)?;
        if let Some(shared_output) = shared_output {
            residual + (mlp_output + shared_output)?
        } else {
            residual + mlp_output
        }
    }
}

pub struct Qwen3MoE {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: ReplicatedLinear,
    device: Device,
    dtype: DType,
    cfg: Config,
}

impl Qwen3MoE {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        comm: Rc<Comm>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, cfg, device, true)?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let reporter = progress_reporter.clone();
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                comm.clone(),
                dtype,
                layer_idx,
            )?;
            layers.push(layer);
            reporter.write().set_progress(layer_idx + 1);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = ReplicatedLinear::load_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            if cfg.tie_word_embeddings {
                vb_m.pp("embed_tokens")
            } else {
                vb.pp("lm_head")
            },
            &None,
            &None,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
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
        let mut xs = self.embed_tokens.forward(input_ids)?;

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?
            }
        } else {
            for layer in self.layers.iter() {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?
            }
        }

        if !seqlens.is_empty() {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)?.to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
