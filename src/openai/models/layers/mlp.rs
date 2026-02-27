use crate::openai::distributed::{
    shard, Comm, TensorParallelColumnLinear, TensorParallelRowLinear, VarBuilder,
};
use crate::openai::models::Config;
use candle::{Module, Result, Tensor};
use candle_core as candle;
pub use std::rc::Rc;
pub struct Mlp {
    gate_proj: TensorParallelColumnLinear,
    up_proj: TensorParallelColumnLinear,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    pub fn new(cfg: &Config, vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        // Some checkpoints (notably FP8 exports) may store packed gate/up as gate_up_proj.
        let has_gate_up_merged =
            vb.contains_tensor("gate_up_proj.weight") || vb.contains_tensor("gate_up_proj");
        let has_split_gate =
            vb.contains_tensor("gate_proj.weight") || vb.contains_tensor("gate_proj");
        let has_split_up = vb.contains_tensor("up_proj.weight") || vb.contains_tensor("up_proj");
        let use_gate_up_merged = has_gate_up_merged && !(has_split_gate && has_split_up);

        let (gate_proj, up_proj) = if use_gate_up_merged {
            let gate_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_sz,
                intermediate_sz * 2,
                false,
                vb.pp("gate_up_proj"),
                shard(0, comm.rank(), comm.world_size() * 2),
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?;
            let up_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_sz,
                intermediate_sz * 2,
                false,
                vb.pp("gate_up_proj"),
                shard(0, comm.world_size() + comm.rank(), comm.world_size() * 2),
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?;
            (gate_proj, up_proj)
        } else {
            let gate_proj = TensorParallelColumnLinear::load_with_hints(
                hidden_sz,
                intermediate_sz,
                false,
                vb.pp("gate_proj"),
                comm.clone(),
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?;
            let up_proj = TensorParallelColumnLinear::load_with_hints(
                hidden_sz,
                intermediate_sz,
                false,
                vb.pp("up_proj"),
                comm.clone(),
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?;
            (gate_proj, up_proj)
        };
        let down_proj = TensorParallelRowLinear::load_with_hints(
            intermediate_sz,
            hidden_sz,
            false,
            vb.pp("down_proj"),
            comm,
            &cfg.isq_quant,
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
