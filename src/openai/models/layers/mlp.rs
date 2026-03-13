use crate::openai::distributed::{
    shard, Comm, MergedParallelColumnLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
    VarBuilder,
};
use crate::openai::models::Config;
use candle::{DType, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::var_builder::Shard;
pub use std::rc::Rc;

enum GateUpProjection {
    Separate {
        gate_proj: TensorParallelColumnLinear,
        up_proj: TensorParallelColumnLinear,
    },
    Packed(MergedParallelColumnLinear),
}

pub struct Mlp {
    gate_up_proj: GateUpProjection,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn normalize_sharded_2d(
        t: Tensor,
        shard: Shard,
        global_dim0: usize,
        global_dim1: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        if shard.dim > 1 {
            candle_core::bail!("unexpected shard dim {} for {}", shard.dim, name);
        }
        let (d0, d1) = t.dims2()?;
        if shard.dim == 0 {
            let local = global_dim0 / shard.world_size;
            if d0 == local {
                return Ok(t);
            }
            if d0 == global_dim0 {
                return t.narrow(0, shard.rank * local, local)?.contiguous();
            }
            candle_core::bail!(
                "unexpected {} shape ({}, {}), shard dim 0 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim0
            );
        }

        let local = global_dim1 / shard.world_size;
        if d1 == local {
            return Ok(t);
        }
        if d1 == global_dim1 {
            return t.narrow(1, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "unexpected {} shape ({}, {}), shard dim 1 expects local {} or global {}",
            name,
            d0,
            d1,
            local,
            global_dim1
        );
    }

    fn try_load_sharded_fp8_weight_scale(
        vb: &VarBuilder,
        out_dim: usize,
        in_dim: usize,
        shard: Shard,
        block_size: &[usize],
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !vb.contains_tensor("weight_scale") && !vb.contains_tensor("weight_scale_inv") {
            return Ok(None);
        }

        let by = block_size[0];
        let bx = block_size[1];
        let scale_dim0 = out_dim.div_ceil(by);
        let scale_dim1 = in_dim.div_ceil(bx);

        let weight = match vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8) {
            Ok(weight) => weight,
            Err(_) => return Ok(None),
        };
        let weight = Self::normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;
        let weight_scale = match vb.get_with_hints_dtype(
            (scale_dim0, scale_dim1),
            "weight_scale",
            shard,
            DType::F32,
        ) {
            Ok(scale) => scale,
            Err(_) => match vb.get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale_inv",
                shard,
                DType::F32,
            ) {
                Ok(scale) => scale,
                Err(_) => return Ok(None),
            },
        };
        let weight_scale = Self::normalize_sharded_2d(
            weight_scale,
            shard,
            scale_dim0,
            scale_dim1,
            "weight_scale",
        )?;
        Ok(Some((weight, weight_scale)))
    }

    fn try_load_packed_gate_up(
        vb: &VarBuilder,
        comm: Rc<Comm>,
        hidden_sz: usize,
        intermediate_sz: usize,
        quant_cfg: &Option<crate::openai::models::QuantConfig>,
        quant: &Option<String>,
        gate_up_merged: bool,
        dtype: DType,
    ) -> Result<Option<GateUpProjection>> {
        if quant.is_some() {
            return Ok(None);
        }

        let is_fp8_quant = quant_cfg
            .as_ref()
            .is_some_and(|cfg| cfg.quant_method == "fp8");
        if let Some(cfg) = quant_cfg {
            if cfg.quant_method != "fp8" {
                return Ok(None);
            }
        }

        let gate_shard = if gate_up_merged {
            shard(0, comm.rank(), comm.world_size() * 2)
        } else {
            shard(0, comm.rank(), comm.world_size())
        };
        let up_shard = if gate_up_merged {
            shard(0, comm.world_size() + comm.rank(), comm.world_size() * 2)
        } else {
            shard(0, comm.rank(), comm.world_size())
        };

        if gate_up_merged {
            let gate_up_vb = vb.pp("gate_up_proj");
            if is_fp8_quant {
                let Some(block_size) = quant_cfg
                    .as_ref()
                    .and_then(|cfg| cfg.weight_block_size.clone())
                else {
                    candle_core::bail!(
                        "LnFp8: weight_block_size must be configured for packed gate_up"
                    );
                };
                if block_size.len() != 2 {
                    candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
                }
                let by = block_size[0];
                let total_out = intermediate_sz * 2;
                let Some((gate_weight, gate_scale)) = Self::try_load_sharded_fp8_weight_scale(
                    &gate_up_vb,
                    total_out,
                    hidden_sz,
                    gate_shard,
                    &block_size,
                )?
                else {
                    return Ok(None);
                };
                let Some((up_weight, up_scale)) = Self::try_load_sharded_fp8_weight_scale(
                    &gate_up_vb,
                    total_out,
                    hidden_sz,
                    up_shard,
                    &block_size,
                )?
                else {
                    return Ok(None);
                };
                let local_gate = gate_weight.dim(0)?;
                let local_up = up_weight.dim(0)?;
                let gate_start = gate_shard.rank * local_gate;
                let up_start = up_shard.rank * local_up;
                if gate_start % by != 0 || up_start % by != 0 {
                    return Ok(None);
                }
                let packed_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
                let packed_scale = Tensor::cat(&[&gate_scale, &up_scale], 0)?;
                #[cfg(feature = "cuda")]
                let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                    .unwrap_or(0) as usize;
                #[cfg(not(feature = "cuda"))]
                let sm_version = 0;
                let merged = MergedParallelColumnLinear::from_packed_local_fp8(
                    packed_weight,
                    packed_scale,
                    None,
                    block_size,
                    sm_version,
                    vec![local_gate, local_up],
                );
                return Ok(Some(GateUpProjection::Packed(merged)));
            }

            if quant_cfg.is_some() {
                return Ok(None);
            }
            let total_out = intermediate_sz * 2;
            let gate_weight = gate_up_vb.get_with_hints_dtype(
                (total_out, hidden_sz),
                "weight",
                gate_shard,
                dtype,
            )?;
            let up_weight = gate_up_vb.get_with_hints_dtype(
                (total_out, hidden_sz),
                "weight",
                up_shard,
                dtype,
            )?;
            let gate_weight = Self::normalize_sharded_2d(
                gate_weight,
                gate_shard,
                total_out,
                hidden_sz,
                "gate_up weight",
            )?;
            let up_weight = Self::normalize_sharded_2d(
                up_weight,
                up_shard,
                total_out,
                hidden_sz,
                "gate_up weight",
            )?;
            let packed_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
            let merged = MergedParallelColumnLinear::from_packed_local(
                packed_weight,
                None,
                vec![gate_weight.dim(0)?, up_weight.dim(0)?],
            );
            return Ok(Some(GateUpProjection::Packed(merged)));
        }

        let gate_vb = vb.pp("gate_proj");
        let up_vb = vb.pp("up_proj");
        if is_fp8_quant {
            let Some(block_size) = quant_cfg
                .as_ref()
                .and_then(|cfg| cfg.weight_block_size.clone())
            else {
                candle_core::bail!(
                    "LnFp8: weight_block_size must be configured for packed gate/up"
                );
            };
            if block_size.len() != 2 {
                candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
            }
            let by = block_size[0];
            let Some((gate_weight, gate_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &gate_vb,
                intermediate_sz,
                hidden_sz,
                gate_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let Some((up_weight, up_scale)) = Self::try_load_sharded_fp8_weight_scale(
                &up_vb,
                intermediate_sz,
                hidden_sz,
                up_shard,
                &block_size,
            )?
            else {
                return Ok(None);
            };
            let local_gate = gate_weight.dim(0)?;
            let local_up = up_weight.dim(0)?;
            let gate_start = gate_shard.rank * local_gate;
            let up_start = up_shard.rank * local_up;
            if gate_start % by != 0 || up_start % by != 0 {
                return Ok(None);
            }
            let packed_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
            let packed_scale = Tensor::cat(&[&gate_scale, &up_scale], 0)?;
            #[cfg(feature = "cuda")]
            let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                .unwrap_or(0) as usize;
            #[cfg(not(feature = "cuda"))]
            let sm_version = 0;
            let merged = MergedParallelColumnLinear::from_packed_local_fp8(
                packed_weight,
                packed_scale,
                None,
                block_size,
                sm_version,
                vec![local_gate, local_up],
            );
            return Ok(Some(GateUpProjection::Packed(merged)));
        }

        if quant_cfg.is_some() {
            return Ok(None);
        }

        let gate_weight = gate_vb.get_with_hints_dtype(
            (intermediate_sz, hidden_sz),
            "weight",
            gate_shard,
            dtype,
        )?;
        let up_weight =
            up_vb.get_with_hints_dtype((intermediate_sz, hidden_sz), "weight", up_shard, dtype)?;
        let gate_weight = Self::normalize_sharded_2d(
            gate_weight,
            gate_shard,
            intermediate_sz,
            hidden_sz,
            "gate weight",
        )?;
        let up_weight = Self::normalize_sharded_2d(
            up_weight,
            up_shard,
            intermediate_sz,
            hidden_sz,
            "up weight",
        )?;
        let packed_weight = Tensor::cat(&[&gate_weight, &up_weight], 0)?;
        let merged = MergedParallelColumnLinear::from_packed_local(
            packed_weight,
            None,
            vec![gate_weight.dim(0)?, up_weight.dim(0)?],
        );
        Ok(Some(GateUpProjection::Packed(merged)))
    }

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

        let gate_up_proj = if let Some(packed) = Self::try_load_packed_gate_up(
            &vb,
            comm.clone(),
            hidden_sz,
            intermediate_sz,
            &cfg.quantization_config,
            &cfg.isq_quant,
            use_gate_up_merged,
            vb.dtype(),
        )? {
            packed
        } else {
            let gate_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_sz,
                if use_gate_up_merged {
                    intermediate_sz * 2
                } else {
                    intermediate_sz
                },
                false,
                vb.pp(if use_gate_up_merged {
                    "gate_up_proj"
                } else {
                    "gate_proj"
                }),
                if use_gate_up_merged {
                    shard(0, comm.rank(), comm.world_size() * 2)
                } else {
                    shard(0, comm.rank(), comm.world_size())
                },
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?;
            let up_proj = TensorParallelColumnLinear::load_with_shard(
                hidden_sz,
                if use_gate_up_merged {
                    intermediate_sz * 2
                } else {
                    intermediate_sz
                },
                false,
                vb.pp(if use_gate_up_merged {
                    "gate_up_proj"
                } else {
                    "up_proj"
                }),
                if use_gate_up_merged {
                    shard(0, comm.world_size() + comm.rank(), comm.world_size() * 2)
                } else {
                    shard(0, comm.rank(), comm.world_size())
                },
                &cfg.isq_quant,
                &cfg.quantization_config,
            )?;
            GateUpProjection::Separate { gate_proj, up_proj }
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
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (gate, up) = match &self.gate_up_proj {
            GateUpProjection::Separate { gate_proj, up_proj } => {
                (gate_proj.forward(xs)?, up_proj.forward(xs)?)
            }
            GateUpProjection::Packed(gate_up_proj) => {
                let gate_up = gate_up_proj.forward(xs)?;
                if gate_up.len() != 2 {
                    candle_core::bail!(
                        "Expected 2 outputs from packed gate/up projection, got {}",
                        gate_up.len()
                    );
                }
                (gate_up[0].clone(), gate_up[1].clone())
            }
        };
        self.down_proj.forward(&(self.act_fn.forward(&gate)? * up)?)
    }
}
