use crate::openai::distributed::{
    shard, Comm, MergedParallelColumnLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
    VarBuilder,
};
use crate::openai::models::linear::is_channel_scale_shape;
use crate::openai::models::Config;
use candle::{DType, Module, Result, Tensor};
use candle_core as candle;
use candle_nn::var_builder::Shard;
pub use std::rc::Rc;

/// NVFP4 gate/up merged into a single fused GEMM. Gate and up may have
/// different weight global scales; `row_scales` carries the per-row (half)
/// scale so the kernel dequantizes each half correctly without re-quant.
struct Nvfp4MergedWeights {
    blocks: Tensor,
    scales: Tensor,
    gate_gscale: f32,
    input_scale: f32,
    row_scales: Tensor,
    weight_scale_swizzled: Option<Tensor>,
    n: usize,
}

enum GateUpProjection {
    Separate {
        gate_proj: TensorParallelColumnLinear,
        up_proj: TensorParallelColumnLinear,
    },
    Packed(MergedParallelColumnLinear),
    Nvfp4Merged(Nvfp4MergedWeights),
}

pub struct Mlp {
    gate_up_proj: GateUpProjection,
    down_proj: TensorParallelRowLinear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    /// Merge two independently-loaded NVFP4 gate/up projections into a single
    /// fused GEMM (sglang-style). Byte-concatenates the FP4 blocks and E4M3
    /// block scales and carries the per-half weight global scales in
    /// `row_scales` so the kernel applies the correct scale to each half. No
    /// re-quantization, so the original NVFP4 weights are preserved exactly.
    fn merge_nvfp4_gate_up(
        gate_proj: TensorParallelColumnLinear,
        up_proj: TensorParallelColumnLinear,
        gate_up_merged: bool,
    ) -> GateUpProjection {
        if gate_up_merged {
            return GateUpProjection::Separate { gate_proj, up_proj };
        }
        // Debug aid: force the separate gate/up path to build a precision
        // baseline against the fused NVFP4 merge. No effect unless set.
        if std::env::var("XINFER_DISABLE_NVFP4_GATEUP_MERGE").is_ok() {
            return GateUpProjection::Separate { gate_proj, up_proj };
        }
        match (gate_proj.as_nvfp4(), up_proj.as_nvfp4()) {
            (Some(g), Some(u)) => {
                let n = g.blocks.dim(0).unwrap_or(0);
                let same_shape = match (g.blocks.dims(), u.blocks.dims()) {
                    (gd, ud) => gd.len() == 2 && gd == ud,
                };
                if n == 0 || !same_shape {
                    return GateUpProjection::Separate { gate_proj, up_proj };
                }
                let blocks = match Tensor::cat(&[&g.blocks, &u.blocks], 0) {
                    Ok(b) => b,
                    Err(_) => return GateUpProjection::Separate { gate_proj, up_proj },
                };
                let scales = match Tensor::cat(&[&g.scales, &u.scales], 0) {
                    Ok(s) => s,
                    Err(_) => return GateUpProjection::Separate { gate_proj, up_proj },
                };
                let dev = blocks.device().clone();
                let (row_scales, swizzled) = match (
                    Tensor::full(g.global_scale, (n,), &dev),
                    Tensor::full(u.global_scale, (n,), &dev),
                ) {
                    (Ok(gr), Ok(ur)) => match Tensor::cat(&[&gr, &ur], 0) {
                        Ok(rs) => {
                            let swizzled = {
                                #[cfg(feature = "cuda")]
                                {
                                    let sm = match scales.device().as_cuda_device() {
                                        Ok(dev) => attention_rs::cuda_utils::sm_version(dev)
                                            .unwrap_or(0)
                                            as usize,
                                        Err(_) => 0,
                                    };
                                    if sm >= 100 {
                                        attention_rs::nvfp4_linear::swizzle_nvfp4_weight_scales(
                                            &scales,
                                        )
                                        .ok()
                                    } else {
                                        None
                                    }
                                }
                                #[cfg(not(feature = "cuda"))]
                                None
                            };
                            (rs, swizzled)
                        }
                        Err(_) => return GateUpProjection::Separate { gate_proj, up_proj },
                    },
                    _ => return GateUpProjection::Separate { gate_proj, up_proj },
                };
                GateUpProjection::Nvfp4Merged(Nvfp4MergedWeights {
                    blocks,
                    scales,
                    gate_gscale: g.global_scale,
                    input_scale: g.input_scale.max(u.input_scale),
                    row_scales,
                    weight_scale_swizzled: swizzled,
                    n,
                })
            }
            _ => GateUpProjection::Separate { gate_proj, up_proj },
        }
    }

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
    ) -> Result<Option<(Tensor, Tensor, Vec<usize>)>> {
        if !vb.contains_tensor("weight_scale")
            && !vb.contains_tensor("weight_scale_inv")
            && !vb.contains_tensor("scale")
        {
            return Ok(None);
        }

        let by = block_size[0];
        let bx = block_size[1];
        let scale_dim0 = out_dim.div_ceil(by);
        let scale_dim1 = in_dim.div_ceil(bx);

        let weight = match vb
            .get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::F8E4M3)
            .or_else(|_| vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8))
        {
            Ok(weight) => weight,
            Err(_) => return Ok(None),
        };
        let weight = Self::normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;
        // Channel-wise compressed-tensors FP8 stores [out_dim, 1].
        let channel_shard = if shard.dim == 0 {
            shard
        } else {
            Shard::default()
        };
        let channel_scale = ["weight_scale", "weight_scale_inv", "scale"]
            .into_iter()
            .find_map(|name| {
                let scale = vb
                    .get_with_hints_dtype((out_dim, 1), name, channel_shard, DType::F32)
                    .ok()?;
                is_channel_scale_shape(scale.dims(), out_dim, channel_shard).then_some(scale)
            });
        let (weight_scale, effective_block_size, scale_shard, global_dims) =
            if let Some(scale) = channel_scale {
                (scale, vec![1, weight.dim(1)?], channel_shard, (out_dim, 1))
            } else {
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
                        Err(_) => match vb.get_with_hints_dtype(
                            (scale_dim0, scale_dim1),
                            "scale",
                            shard,
                            DType::F32,
                        ) {
                            Ok(scale) => scale,
                            Err(_) => return Ok(None),
                        },
                    },
                };
                (
                    weight_scale,
                    block_size.to_vec(),
                    shard,
                    (scale_dim0, scale_dim1),
                )
            };
        let weight_scale = Self::normalize_sharded_2d(
            weight_scale,
            scale_shard,
            global_dims.0,
            global_dims.1,
            "weight_scale",
        )?;
        Ok(Some((weight, weight_scale, effective_block_size)))
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
                let total_out = intermediate_sz * 2;
                let Some((gate_weight, gate_scale, gate_block_size)) =
                    Self::try_load_sharded_fp8_weight_scale(
                        &gate_up_vb,
                        total_out,
                        hidden_sz,
                        gate_shard,
                        &block_size,
                    )?
                else {
                    return Ok(None);
                };
                let Some((up_weight, up_scale, up_block_size)) =
                    Self::try_load_sharded_fp8_weight_scale(
                        &gate_up_vb,
                        total_out,
                        hidden_sz,
                        up_shard,
                        &block_size,
                    )?
                else {
                    return Ok(None);
                };
                if gate_block_size != up_block_size {
                    return Ok(None);
                }
                let by = gate_block_size[0];
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
                    gate_block_size,
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
            let Some((gate_weight, gate_scale, gate_block_size)) =
                Self::try_load_sharded_fp8_weight_scale(
                    &gate_vb,
                    intermediate_sz,
                    hidden_sz,
                    gate_shard,
                    &block_size,
                )?
            else {
                return Ok(None);
            };
            let Some((up_weight, up_scale, up_block_size)) =
                Self::try_load_sharded_fp8_weight_scale(
                    &up_vb,
                    intermediate_sz,
                    hidden_sz,
                    up_shard,
                    &block_size,
                )?
            else {
                return Ok(None);
            };
            if gate_block_size != up_block_size {
                return Ok(None);
            }
            let by = gate_block_size[0];
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
                gate_block_size,
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
            Self::merge_nvfp4_gate_up(gate_proj, up_proj, use_gate_up_merged)
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
            GateUpProjection::Nvfp4Merged(m) => {
                let orig_dims = xs.dims().to_vec();
                let x_2d = if orig_dims.len() > 2 {
                    let features = orig_dims[orig_dims.len() - 1];
                    let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                    xs.reshape((batch_size, features))?
                } else {
                    xs.clone()
                };
                let gate_up = attention_rs::nvfp4_linear::nvfp4_matmul(
                    &x_2d,
                    &m.blocks,
                    &m.scales,
                    m.gate_gscale,
                    m.input_scale,
                    None,
                    crate::openai::models::linear::linear_is_prefill(),
                    m.weight_scale_swizzled.as_ref(),
                    Some(&m.row_scales),
                )?;
                let leading: Vec<usize> = if orig_dims.len() > 2 {
                    orig_dims[..orig_dims.len() - 1].to_vec()
                } else {
                    vec![x_2d.dim(0)?]
                };
                let mut gate_shape = leading.clone();
                gate_shape.push(m.n);
                let gate = gate_up.narrow(1, 0, m.n)?.reshape(gate_shape.clone())?;
                let up = gate_up.narrow(1, m.n, m.n)?.reshape(gate_shape)?;
                (gate, up)
            }
        };
        self.down_proj.forward(&(self.act_fn.forward(&gate)? * up)?)
    }
}
