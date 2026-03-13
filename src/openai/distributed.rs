use crate::openai::models::linear::{linear_no_bias_x as linear, Linear, LinearX, LnFp8};
#[cfg(feature = "nccl")]
pub use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, Id};
use candle_core::{CpuStorage, Layout, Module, Result, Shape, Tensor};
use candle_core::{CustomOp1, DType};
use candle_nn::var_builder::Shard;
pub use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, LayerNorm, RmsNorm};
#[cfg(not(feature = "nccl"))]
pub struct Comm {}
#[cfg(not(feature = "nccl"))]
impl Comm {
    //dummy Comm
    pub fn default() -> Self {
        Self {}
    }
    pub fn dim(&self) -> usize {
        0
    }

    pub fn rank(&self) -> usize {
        0
    }
    pub fn world_size(&self) -> usize {
        1
    }
}

#[cfg(not(feature = "nccl"))]
pub struct Id {}

pub use std::rc::Rc;

pub struct ReplicatedLinear {
    linear: LinearX,
    bias: Option<Tensor>,
}

pub struct TensorParallelColumnLinear {
    linear: LinearX,
    bias: Option<Tensor>,
}

impl TensorParallelColumnLinear {
    pub fn new(linear: LinearX) -> Self {
        Self { linear, bias: None }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut xs = self.linear.forward(x)?;
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(bias)?;
        }
        Ok(xs)
    }
}

pub struct MergedParallelColumnLinear {
    pub linears: Vec<TensorParallelColumnLinear>,
    pub biases: Vec<Option<Tensor>>,
    pub output_splits: Option<Vec<usize>>,
}

impl MergedParallelColumnLinear {
    /// Load FP8 gate_up_proj with proper sharding for tensor parallelism.
    /// This function loads the packed gate_up weight and splits it into gate and up projections.
    pub fn load_fp8_gate_up_merged(
        in_dim: usize,
        out_dim: usize, // Should be intermediate_size * 2
        vb: VarBuilder,
        comm: Rc<Comm>,
        quant_cfg: &QuantConfig,
        _dtype: DType,
    ) -> Result<Self> {
        let block_size = quant_cfg
            .weight_block_size
            .clone()
            .unwrap_or(vec![128, 128]);
        if block_size.len() != 2 {
            candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
        }
        let by = block_size[0];
        let bx = block_size[1];
        if by == 0 || bx == 0 {
            candle_core::bail!("LnFp8: invalid zero in weight_block_size");
        }

        // Load full weight and scale (no sharding initially)
        let weight =
            vb.get_with_hints_dtype((out_dim, in_dim), "weight", Shard::default(), DType::U8)?;
        let scale_dim0 = (out_dim + by - 1) / by;
        let scale_dim1 = (in_dim + bx - 1) / bx;
        let weight_scale = match vb.get_with_hints_dtype(
            (scale_dim0, scale_dim1),
            "weight_scale",
            Shard::default(),
            DType::F32,
        ) {
            Ok(s) => s,
            Err(_) => vb
                .get_with_hints_dtype(
                    (scale_dim0, scale_dim1),
                    "weight_scale_inv",
                    Shard::default(),
                    DType::F32,
                )
                .map_err(|_| {
                    candle_core::Error::Msg(
                        "LnFp8: Missing weight_scale or weight_scale_inv".into(),
                    )
                })?,
        };

        #[cfg(feature = "cuda")]
        let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
            .unwrap_or(0) as usize;

        #[cfg(not(feature = "cuda"))]
        let sm_version = 0;

        let world_size = comm.world_size();
        let rank = comm.rank();

        // Split gate (first half) and up (second half)
        let intermediate_size = out_dim / 2;

        // Gate projection: rows [0, intermediate_size)
        let gate_weight_start = 0;
        let gate_weight_end = intermediate_size;
        let gate_weight = weight.narrow(0, gate_weight_start, intermediate_size)?;

        // Calculate scale rows for gate
        let gate_scale_start = gate_weight_start / by;
        let gate_scale_end = (gate_weight_end + by - 1) / by;
        let gate_scale_rows = gate_scale_end - gate_scale_start;
        let gate_scale = weight_scale.narrow(0, gate_scale_start, gate_scale_rows)?;

        // Up projection: rows [intermediate_size, out_dim)
        let up_weight_start = intermediate_size;
        let up_weight = weight.narrow(0, up_weight_start, intermediate_size)?;

        // Calculate scale rows for up
        let up_scale_start = up_weight_start / by;
        let up_scale_end = (up_weight_start + intermediate_size + by - 1) / by;
        let up_scale_rows = up_scale_end - up_scale_start;
        let up_scale = weight_scale.narrow(0, up_scale_start, up_scale_rows)?;

        // Now shard gate and up along dimension 0
        // Each rank gets a slice of the intermediate_size dimension
        if intermediate_size % world_size != 0 {
            candle_core::bail!(
                "FP8 gate_up intermediate_size {} must be divisible by world_size {}",
                intermediate_size,
                world_size
            );
        }
        let local_intermediate = intermediate_size / world_size;

        // Gate shard
        let gate_local_start = rank * local_intermediate;
        let gate_weight_shard = gate_weight
            .narrow(0, gate_local_start, local_intermediate)?
            .contiguous()?;
        let gate_scale_local_start = gate_local_start / by;
        let gate_scale_shard = gate_scale
            .narrow(
                0,
                gate_scale_local_start,
                (local_intermediate + by - 1) / by,
            )?
            .contiguous()?;

        // Up shard - starts at intermediate_size + rank * local_intermediate
        let up_local_start = up_weight_start + rank * local_intermediate;
        let up_weight_shard = up_weight
            .narrow(0, up_local_start - up_weight_start, local_intermediate)?
            .contiguous()?;
        let up_scale_local_start = up_local_start / by;
        let up_scale_shard = up_scale
            .narrow(0, up_scale_local_start, (local_intermediate + by - 1) / by)?
            .contiguous()?;

        #[cfg(feature = "cutlass")]
        let gate_scale_cutlass = if sm_version >= 100 {
            Some(gate_scale_shard.t()?)
        } else if sm_version >= 90 {
            Some(gate_scale_shard.t()?.contiguous()?)
        } else {
            None
        };

        #[cfg(feature = "cutlass")]
        let up_scale_cutlass = if sm_version >= 100 {
            Some(up_scale_shard.t()?)
        } else if sm_version >= 90 {
            Some(up_scale_shard.t()?.contiguous()?)
        } else {
            None
        };

        #[cfg(not(feature = "cutlass"))]
        let gate_scale_cutlass = None;
        #[cfg(not(feature = "cutlass"))]
        let up_scale_cutlass = None;

        let gate_linear = LinearX::LnFp8(LnFp8 {
            weight: gate_weight_shard,
            weight_scale: gate_scale_shard,
            weight_scale_cutlass: gate_scale_cutlass,
            bias: None,
            weight_block_size: block_size.clone(),
            sm_version,
        });
        let gate_tp = TensorParallelColumnLinear {
            linear: gate_linear,
            bias: None,
        };

        let up_linear = LinearX::LnFp8(LnFp8 {
            weight: up_weight_shard,
            weight_scale: up_scale_shard,
            weight_scale_cutlass: up_scale_cutlass,
            bias: None,
            weight_block_size: block_size.clone(),
            sm_version,
        });
        let up_tp = TensorParallelColumnLinear {
            linear: up_linear,
            bias: None,
        };

        Ok(Self {
            linears: vec![gate_tp, up_tp],
            biases: vec![None, None],
            output_splits: Some(vec![local_intermediate, local_intermediate]),
        })
    }

    pub fn new(linears: Vec<TensorParallelColumnLinear>) -> Self {
        Self {
            linears,
            biases: Vec::new(),
            output_splits: None,
        }
    }

    pub fn from_packed_local(
        packed_weight: Tensor,
        packed_bias: Option<Tensor>,
        output_splits: Vec<usize>,
    ) -> Self {
        let linear = LinearX::Linear(Linear::new(packed_weight, None));
        Self {
            linears: vec![TensorParallelColumnLinear { linear, bias: None }],
            biases: vec![packed_bias],
            output_splits: Some(output_splits),
        }
    }

    pub fn from_packed_local_fp8(
        packed_weight: Tensor,
        packed_scale: Tensor,
        packed_bias: Option<Tensor>,
        block_size: Vec<usize>,
        sm_version: usize,
        output_splits: Vec<usize>,
    ) -> Self {
        #[cfg(feature = "cutlass")]
        let packed_scale_cutlass = if sm_version >= 100 {
            Some(packed_scale.t().unwrap())
        } else if sm_version >= 90 {
            Some(packed_scale.t().unwrap().contiguous().unwrap())
        } else {
            None
        };

        #[cfg(not(feature = "cutlass"))]
        let packed_scale_cutlass = None;

        let linear = LinearX::LnFp8(LnFp8 {
            weight: packed_weight,
            weight_scale: packed_scale,
            weight_scale_cutlass: packed_scale_cutlass,
            bias: None,
            weight_block_size: block_size,
            sm_version,
        });
        Self {
            linears: vec![TensorParallelColumnLinear { linear, bias: None }],
            biases: vec![packed_bias],
            output_splits: Some(output_splits),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        if let Some(output_splits) = &self.output_splits {
            if self.linears.len() != 1 {
                candle_core::bail!(
                    "MergedParallelColumnLinear expected exactly 1 linear for split outputs, got {}",
                    self.linears.len()
                );
            }
            let mut ys = self.linears[0].forward(x)?;
            if let Some(Some(bias)) = self.biases.first() {
                ys = ys.broadcast_add(bias)?;
            }
            let split_dim = ys.dims().len().saturating_sub(1);
            let total_dim = ys.dim(split_dim)?;
            let expected_dim: usize = output_splits.iter().sum();
            if total_dim != expected_dim {
                candle_core::bail!(
                    "MergedParallelColumnLinear split mismatch: output dim {} vs expected {}",
                    total_dim,
                    expected_dim
                );
            }
            let mut outputs = Vec::with_capacity(output_splits.len());
            let mut start = 0usize;
            for split_size in output_splits {
                outputs.push(ys.narrow(split_dim, start, *split_size)?.contiguous()?);
                start += *split_size;
            }
            return Ok(outputs);
        }

        let mut xss = Vec::<Tensor>::new();
        for i in 0..self.linears.len() {
            let mut xs = self.linears[i].forward(x)?;
            if self.biases.len() > 0 && i < self.biases.len() {
                if let Some(bias) = &self.biases[i] {
                    xs = xs.broadcast_add(bias)?;
                }
            }
            xss.push(xs);
        }
        Ok(xss)
    }
}

pub struct TensorParallelRowLinear {
    linear: LinearX,
    #[cfg(feature = "nccl")]
    all_reduce: AllReduce,
    bias: Option<Tensor>,
}

#[allow(dead_code)]
pub struct AllReduce {
    comm: Rc<Comm>,
}

unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl AllReduce {
    pub fn new(comm: Rc<Comm>) -> Self {
        Self { comm: comm.clone() }
    }
    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        // use candle_core::cuda::cudarc::driver::result;
        // unsafe { result::ctx::set_current(*self.comm.comm.device().cu_primary_ctx()) }.unwrap();
        // self.comm.barrier.wait()?;
        xs.apply_op1_no_bwd(self)
    }
}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is never used on cpu")
    }

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::nccl::safe::ReduceOp;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::DType;
        use half::{bf16, f16};

        if !l.is_contiguous() {
            candle_core::bail!("Inputs for all_reduce must be contiguous!");
        }
        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let start_offset = l.start_offset();

        let dst = match s.dtype() {
            DType::BF16 => {
                let full_slice = s.as_cuda_slice::<bf16>()?;
                // Slice to only the valid elements (handles narrow/view tensors)
                let src_slice = full_slice.slice(start_offset..start_offset + elem_count);
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(&src_slice, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let full_slice = s.as_cuda_slice::<f16>()?;
                // Slice to only the valid elements (handles narrow/view tensors)
                let src_slice = full_slice.slice(start_offset..start_offset + elem_count);
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(&src_slice, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

impl TensorParallelRowLinear {
    #[allow(unused_variables)]
    pub fn new(linear: LinearX, comm: Rc<Comm>) -> Self {
        #[cfg(feature = "nccl")]
        let all_reduce = AllReduce { comm };
        Self {
            linear,
            #[cfg(feature = "nccl")]
            all_reduce,
            bias: None,
        }
    }

    #[allow(unused_variables)]
    pub fn new_with_bias(linear: LinearX, bias: Option<Tensor>, comm: Rc<Comm>) -> Self {
        #[cfg(feature = "nccl")]
        let all_reduce = AllReduce { comm };
        Self {
            linear,
            #[cfg(feature = "nccl")]
            all_reduce,
            bias,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xs = self.linear.forward(x)?;
        #[cfg(feature = "nccl")]
        let xs = xs.apply_op1_no_bwd(&self.all_reduce)?;

        if let Some(bias) = &self.bias {
            let xs = xs.broadcast_add(bias)?;
            Ok(xs)
        } else {
            Ok(xs)
        }
    }
}

pub fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}
use crate::openai::models::QuantConfig;
impl TensorParallelColumnLinear {
    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilder,
        comm: Rc<Comm>,
        quant: &Option<String>,
        quant_config: &Option<QuantConfig>,
    ) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let dtype = vb.dtype();
        let bs = if bias {
            let full_bias = vb.get((out_dim,), "bias");
            if full_bias.is_ok() {
                if comm.world_size() > 1 {
                    //match bias to its corresponding partial weight
                    let out_dim_partition = out_dim / comm.world_size();
                    let full_bias = full_bias
                        .unwrap()
                        .narrow(0, comm.rank() * out_dim_partition, out_dim_partition)?
                        .contiguous()?;
                    Some(full_bias)
                } else {
                    Some(vb.get((out_dim,), "bias")?)
                }
            } else {
                None
            }
        } else {
            None
        };
        let linear = linear(
            in_dim,
            out_dim,
            vb,
            shard(0, rank, size),
            quant,
            quant_config,
            dtype,
            None,
        )?;
        Ok(Self { linear, bias: bs })
    }

    pub fn load_with_shard(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilder,
        shard: Shard,
        quant: &Option<String>,
        quant_config: &Option<QuantConfig>,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let bs = if bias {
            let full_bias = vb.get((out_dim,), "bias");
            if full_bias.is_ok() {
                if shard.world_size > 1 {
                    let out_dim_partition = out_dim / shard.world_size;
                    let full_bias = full_bias
                        .unwrap()
                        .narrow(0, shard.rank * out_dim_partition, out_dim_partition)?
                        .contiguous()?;
                    Some(full_bias)
                } else {
                    Some(vb.get((out_dim,), "bias")?)
                }
            } else {
                None
            }
        } else {
            None
        };
        let linear = linear(in_dim, out_dim, vb, shard, quant, quant_config, dtype, None)?;
        Ok(Self { linear, bias: bs })
    }
}

impl MergedParallelColumnLinear {
    pub fn load_merged_with_hints(
        in_dim: usize,
        out_dim: usize,
        chunk: usize,
        vb: VarBuilder,
        comm: Rc<Comm>,
        quant: &Option<String>,
        quant_config: &Option<QuantConfig>,
    ) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let dtype = vb.dtype();
        if quant_config.is_some() {
            candle_core::bail!("Merged quantized weight is not supported at the moment!");
        }
        let mut vec_linear = Vec::<TensorParallelColumnLinear>::new();
        for chunk_idx in 0..chunk {
            let linear = linear(
                in_dim,
                out_dim,
                vb.clone(),
                shard(0, rank, size),
                quant,
                quant_config,
                dtype,
                Some((chunk_idx, chunk)),
            )?;

            let ln = TensorParallelColumnLinear { linear, bias: None };
            vec_linear.push(ln);
        }
        Ok(Self {
            linears: vec_linear,
            biases: vec![None; chunk],
            output_splits: None,
        })
    }

    pub fn load_merged_chunks(
        in_dim: usize,
        out_dim: usize,
        chunk_dim: usize,
        chunks: Vec<usize>,
        vb: VarBuilder,
        comm: Rc<Comm>,
        quant_cfg: &Option<QuantConfig>,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let is_fp8_quant = if let Some(cfg) = quant_cfg {
            cfg.quant_method == "fp8"
        } else {
            false
        };

        if quant_cfg.is_some() && !is_fp8_quant {
            candle_core::bail!(
                "Merged quantized weight is not supported at the moment, using ISQ instead!"
            );
        }
        let mut vec_linear = Vec::<TensorParallelColumnLinear>::new();
        let mut output_splits: Option<Vec<usize>> = None;
        use crate::openai::models::linear::{LinearX, LnFp8, QLinear};
        if is_fp8_quant {
            if chunk_dim != 0 {
                candle_core::bail!(
                    "FP8 merged chunk loading currently supports chunk_dim=0 only, got {}",
                    chunk_dim
                );
            }
            let quant_cfg = quant_cfg.as_ref().ok_or_else(|| {
                candle_core::Error::Msg(
                    "FP8 merged chunk loading requires quantization config".to_string(),
                )
            })?;

            let block_size = quant_cfg
                .weight_block_size
                .clone()
                .unwrap_or(vec![128, 128]);
            if block_size.len() != 2 {
                candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
            }
            let by = block_size[0];
            let bx = block_size[1];
            if by == 0 || bx == 0 {
                candle_core::bail!("LnFp8: invalid zero in weight_block_size");
            }

            let weight =
                vb.get_with_hints_dtype((out_dim, in_dim), "weight", Shard::default(), DType::U8)?;
            let scale_dim0 = (out_dim + by - 1) / by;
            let scale_dim1 = (in_dim + bx - 1) / bx;
            let weight_scale = match vb.get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale",
                Shard::default(),
                DType::F32,
            ) {
                Ok(s) => s,
                Err(_) => vb
                    .get_with_hints_dtype(
                        (scale_dim0, scale_dim1),
                        "weight_scale_inv",
                        Shard::default(),
                        DType::F32,
                    )
                    .map_err(|_| {
                        candle_core::Error::Msg(
                            "LnFp8: Missing weight_scale or weight_scale_inv".into(),
                        )
                    })?,
            };

            #[cfg(feature = "cuda")]
            let sm_version = attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?)
                .unwrap_or(0) as usize;

            #[cfg(not(feature = "cuda"))]
            let sm_version = 0;

            let mut chunk_start = 0;
            let mut local_weight_chunks = Vec::<Tensor>::with_capacity(chunks.len());
            let mut local_scale_chunks = Vec::<Tensor>::with_capacity(chunks.len());
            let mut local_output_splits = Vec::<usize>::with_capacity(chunks.len());
            for chunk_idx in 0..chunks.len() {
                let chunk_size = chunks[chunk_idx];
                let ws = weight.narrow(0, chunk_start, chunk_size)?;
                if ws.dim(0)? % comm.world_size() != 0 {
                    candle_core::bail!(
                        "FP8 merged chunk {} dim {} is not divisible by shard world_size {}",
                        chunk_idx,
                        ws.dim(0)?,
                        comm.world_size()
                    );
                }
                let local_out = ws.dim(0)? / comm.world_size();
                if local_out == 0 {
                    candle_core::bail!("FP8 merged chunk {} produced empty shard", chunk_idx);
                }
                let local_out_start = chunk_start + comm.rank() * local_out;
                if local_out_start % by != 0 {
                    candle_core::bail!(
                        "FP8 merged chunk {} local start {} is not aligned to block_size_y {}",
                        chunk_idx,
                        local_out_start,
                        by
                    );
                }
                let ws_chunk = ws
                    .narrow(0, comm.rank() * local_out, local_out)?
                    .contiguous()?;
                local_output_splits.push(local_out);

                let scale_row_start = local_out_start / by;
                let scale_rows = (local_out + by - 1) / by;
                if scale_row_start + scale_rows > scale_dim0 {
                    candle_core::bail!(
                        "FP8 merged chunk {} scale slice out of bounds: start={}, rows={}, total={}",
                        chunk_idx,
                        scale_row_start,
                        scale_rows,
                        scale_dim0
                    );
                }
                let ws_scale = weight_scale
                    .narrow(0, scale_row_start, scale_rows)?
                    .contiguous()?;
                local_weight_chunks.push(ws_chunk);
                local_scale_chunks.push(ws_scale);
                chunk_start += chunk_size;
            }

            let merged_weight = if local_weight_chunks.len() == 1 {
                local_weight_chunks.remove(0)
            } else {
                let weight_refs = local_weight_chunks.iter().collect::<Vec<_>>();
                Tensor::cat(&weight_refs, 0)?
            };

            let merged_scale = if local_scale_chunks.len() == 1 {
                local_scale_chunks.remove(0)
            } else {
                let scale_refs = local_scale_chunks.iter().collect::<Vec<_>>();
                Tensor::cat(&scale_refs, 0)?
            };

            #[cfg(feature = "cutlass")]
            let merged_scale_cutlass = if sm_version >= 100 {
                Some(merged_scale.t()?)
            } else if sm_version >= 90 {
                Some(merged_scale.t()?.contiguous()?)
            } else {
                None
            };

            #[cfg(not(feature = "cutlass"))]
            let merged_scale_cutlass = None;

            let linear = LinearX::LnFp8(LnFp8 {
                weight: merged_weight,
                weight_scale: merged_scale,
                weight_scale_cutlass: merged_scale_cutlass,
                bias: None,
                weight_block_size: block_size.clone(),
                sm_version,
            });
            let ln = TensorParallelColumnLinear { linear, bias: None };
            vec_linear.push(ln);
            output_splits = Some(local_output_splits);
        } else {
            let weight = vb.get((out_dim, in_dim), "weight")?;
            let weight = if weight.dtype() != dtype {
                weight.to_dtype(dtype)?
            } else {
                weight
            };
            let mut chunk_start = 0;
            for chunk_idx in 0..chunks.len() {
                let chunk_size = chunks[chunk_idx];
                let ws = weight.narrow(chunk_dim, chunk_start, chunk_size)?;
                let c_chunk_size = ws.dim(0)? / comm.world_size();
                let ws_chunk = ws
                    .narrow(0, comm.rank() * c_chunk_size, c_chunk_size)?
                    .contiguous()?;
                chunk_start += chunk_size;

                let ln = crate::openai::models::linear::Linear::new(ws_chunk, None);
                let linear = if let Some(quantized_type) = quant {
                    let quantized_type = if chunk_idx == chunks.len() - 1 {
                        "q8_0".to_string()
                    } else {
                        quantized_type.clone()
                    };
                    LinearX::QLinear(QLinear::from_linear_x(ln, quantized_type, quant_cfg))
                } else {
                    LinearX::Linear(ln)
                };
                let ln = TensorParallelColumnLinear { linear, bias: None };
                vec_linear.push(ln);
            }
        }

        let linear_count = vec_linear.len();
        Ok(Self {
            linears: vec_linear,
            biases: vec![None; linear_count],
            output_splits,
        })
    }
}

impl TensorParallelRowLinear {
    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilder,
        comm: Rc<Comm>,
        quant: &Option<String>,
        quant_config: &Option<QuantConfig>,
    ) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let dtype = vb.dtype();
        let bs = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };
        let linear = linear(
            in_dim,
            out_dim,
            vb,
            shard(1, rank, size),
            quant,
            quant_config,
            dtype,
            None,
        )?;
        Ok(Self::new_with_bias(linear, bs, comm))
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get_with_hints(size, "weight", shard(0, 0, 1))?;
    Ok(RmsNorm::new(weight, eps))
}

pub fn rms_norm_with_dtype(size: usize, eps: f64, vb: VarBuilder, dtype: DType) -> Result<RmsNorm> {
    let weight = vb.get_with_hints_dtype(size, "weight", shard(0, 0, 1), dtype)?;
    Ok(RmsNorm::new(weight, eps))
}

pub fn rms_norm_x(
    size: usize,
    eps: f64,
    vb: VarBuilder,
    dtype: DType,
    add_unit_offset: bool,
) -> Result<RmsNorm> {
    let weight = vb.get_with_hints_dtype(size, "weight", shard(0, 0, 1), dtype)?;
    let weight = if add_unit_offset {
        (weight + 1.0f64)?
    } else {
        weight
    };
    Ok(RmsNorm::new(weight, eps))
}

pub fn layer_norm(size: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get_with_hints(size, "weight", Shard::default())?;
    if affine {
        Ok(LayerNorm::new(weight, vb.get(size, "bias")?, eps))
    } else {
        Ok(LayerNorm::new_no_bias(weight, eps))
    }
}

pub fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

impl ReplicatedLinear {
    pub fn from(linear: LinearX, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self { linear, bias })
    }

    pub fn from_weight_bias(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let linear = LinearX::Linear(Linear::new(weight, None));
        Ok(Self { linear, bias })
    }

    pub fn load_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        quant: &Option<String>,
        quant_config: &Option<QuantConfig>,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let linear = linear(
            in_dim,
            out_dim,
            vb,
            shard(0, 0, 1),
            quant,
            quant_config,
            dtype,
            None,
        )?;
        Ok(Self { linear, bias: None })
    }

    pub fn load_b(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilder,
        quant: &Option<String>,
        quant_config: &Option<QuantConfig>,
    ) -> Result<Self> {
        if !bias {
            ReplicatedLinear::load_no_bias(in_dim, out_dim, vb, quant, quant_config)
        } else {
            let dtype = vb.dtype();
            let bs = vb.get((out_dim,), "bias")?;
            let linear = linear(
                in_dim,
                out_dim,
                vb,
                shard(0, 0, 1),
                quant,
                quant_config,
                dtype,
                None,
            )?;
            Ok(Self {
                linear,
                bias: Some(bs),
            })
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut xs = self.linear.forward(x)?;
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(bias)?;
        }
        Ok(xs)
    }

    pub fn offload(&mut self) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        panic!("tensor offload not available on this device!");
        #[cfg(feature = "cuda")]
        self.linear.offload()
    }

    pub fn reload(&mut self) -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        panic!("tensor offload not available on this device!");
        #[cfg(feature = "cuda")]
        self.linear.reload()
    }
}
