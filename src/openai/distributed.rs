use crate::openai::models::linear::{linear_no_bias_x as linear, Linear, LinearX, LnFp8};
#[cfg(feature = "nccl")]
pub use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, Id};
use candle_core::quantized::GgmlDType;
use candle_core::quantized::QTensor;
use candle_core::{CpuStorage, Device, Layout, Module, Result, Shape, Tensor};
use candle_core::{CustomOp1, DType};
use candle_nn::var_builder::Shard;
pub use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, LayerNorm, RmsNorm};
use std::sync::Arc;
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

fn can_quantize_to(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8_0
    )
}

fn closest_quantizable_dtype(orig: GgmlDType) -> GgmlDType {
    if can_quantize_to(orig) {
        return orig;
    }
    let bpw = orig.type_size() as f64 * 8.0 / orig.block_size() as f64;
    let candidates = [
        (GgmlDType::Q2K, 2.625),
        (GgmlDType::Q3K, 3.4375),
        (GgmlDType::Q4K, 4.5),
        (GgmlDType::Q5K, 5.5),
        (GgmlDType::Q6K, 6.5625),
        (GgmlDType::Q8_0, 8.5),
    ];
    candidates
        .iter()
        .min_by(|a, b| (a.1 - bpw).abs().partial_cmp(&(b.1 - bpw).abs()).unwrap())
        .map(|c| c.0)
        .unwrap_or(GgmlDType::Q4K)
}

pub fn gguf_shard_qtensor(
    qt: Arc<QTensor>,
    dim: usize,
    rank: usize,
    world_size: usize,
    device: &Device,
) -> Result<Arc<QTensor>> {
    if world_size <= 1 {
        return Ok(qt);
    }
    let orig_dtype = qt.dtype();
    let t = qt.dequantize_f16(device)?;
    let dim_size = t.dim(dim)?;
    if dim_size % world_size != 0 {
        candle_core::bail!(
            "gguf_shard_qtensor dim {} size {} not divisible by world_size {}",
            dim,
            dim_size,
            world_size
        );
    }
    let chunk_size = dim_size / world_size;
    let local = t.narrow(dim, rank * chunk_size, chunk_size)?.contiguous()?;
    drop(t);
    let target = closest_quantizable_dtype(orig_dtype);
    let last_dim = local.dim(candle_core::D::Minus1)?;
    let wdtype = if last_dim % target.block_size() != 0 {
        GgmlDType::Q8_0
    } else {
        target
    };
    Ok(Arc::new(QTensor::quantize_owned(local, wdtype)?))
}

/// Shard MoE expert weight tensors with vllm.rs-aligned dtype selection.
/// gate/up experts use `closest_quantizable_dtype(orig)`, down experts use a
/// higher-precision dtype when block-aligned.  Block-aligned chunk padding is
/// applied so the sharded intermediate dimension is always a multiple of the
/// target block size.
pub fn gguf_shard_moe_experts(
    gate_qt: Arc<QTensor>,
    up_qt: Arc<QTensor>,
    down_qt: Arc<QTensor>,
    rank: usize,
    world_size: usize,
    device: &Device,
) -> Result<(Arc<QTensor>, Arc<QTensor>, Arc<QTensor>)> {
    if world_size <= 1 {
        return Ok((gate_qt, up_qt, down_qt));
    }

    let orig_gate_dtype = gate_qt.dtype();
    let orig_down_dtype = down_qt.dtype();

    let gate_f16 = gate_qt.dequantize_f16(device)?;
    let up_f16 = up_qt.dequantize_f16(device)?;
    let down_f16 = down_qt.dequantize_f16(device)?;

    let moe_intermediate_size = gate_f16.dim(1)?;

    let target_gate = closest_quantizable_dtype(orig_gate_dtype);
    let ggml_dtype = if (moe_intermediate_size / world_size) % target_gate.block_size() == 0 {
        target_gate
    } else if (moe_intermediate_size / world_size) % GgmlDType::Q4K.block_size() == 0 {
        GgmlDType::Q4K
    } else {
        GgmlDType::Q8_0
    };

    let target_down = closest_quantizable_dtype(orig_down_dtype);
    let high_precision_dtype =
        if (moe_intermediate_size / world_size) % target_down.block_size() == 0 {
            target_down
        } else if (moe_intermediate_size / world_size) % GgmlDType::Q6K.block_size() == 0 {
            GgmlDType::Q6K
        } else {
            ggml_dtype
        };

    let block_size = ggml_dtype.block_size();
    let moe_intermediate_chunk = if (moe_intermediate_size / world_size) % block_size != 0 {
        ((moe_intermediate_size / world_size + block_size - 1) / block_size) * block_size
    } else {
        moe_intermediate_size / world_size
    };

    let cur_chunk_size =
        if rank * moe_intermediate_chunk + moe_intermediate_chunk <= moe_intermediate_size {
            moe_intermediate_chunk
        } else {
            moe_intermediate_size - rank * moe_intermediate_chunk
        };

    let (ggml_dtype, high_precision_dtype, cur_chunk_size) =
        if cur_chunk_size == 0 || cur_chunk_size % block_size != 0 {
            let fb = GgmlDType::Q8_0;
            let fb_bs = fb.block_size();
            let fb_chunk = moe_intermediate_size / world_size;
            let fb_cur = if rank * fb_chunk + fb_chunk <= moe_intermediate_size {
                fb_chunk
            } else {
                moe_intermediate_size - rank * fb_chunk
            };
            assert!(
                fb_cur > 0 && fb_cur % fb_bs == 0,
                "Unable to split moe_intermediate_size {} into {} ranks under block_size {}",
                moe_intermediate_size,
                world_size,
                fb_bs
            );
            (fb, fb, fb_cur)
        } else {
            (ggml_dtype, high_precision_dtype, cur_chunk_size)
        };

    let gate_local = gate_f16
        .narrow(1, rank * moe_intermediate_chunk, cur_chunk_size)?
        .contiguous()?;
    let up_local = up_f16
        .narrow(1, rank * moe_intermediate_chunk, cur_chunk_size)?
        .contiguous()?;
    let down_local = down_f16
        .narrow(2, rank * moe_intermediate_chunk, cur_chunk_size)?
        .contiguous()?;

    drop(gate_f16);
    drop(up_f16);
    drop(down_f16);

    Ok((
        Arc::new(QTensor::quantize(&gate_local, ggml_dtype)?),
        Arc::new(QTensor::quantize(&up_local, ggml_dtype)?),
        Arc::new(QTensor::quantize(&down_local, high_precision_dtype)?),
    ))
}

pub struct ReplicatedLinear {
    linear: LinearX,
    bias: Option<Tensor>,
}

pub struct TensorParallelColumnLinear {
    linear: LinearX,
    bias: Option<Tensor>,
}

pub fn tensor_parallel_chunk(
    x: &Tensor,
    dim: usize,
    rank: usize,
    world_size: usize,
    name: &str,
) -> Result<Tensor> {
    if world_size <= 1 {
        return Ok(x.clone());
    }
    let dim_size = x.dim(dim)?;
    if dim_size % world_size != 0 {
        candle_core::bail!(
            "tensor-parallel chunk for {} dim {} size {} is not divisible by world_size {}",
            name,
            dim,
            dim_size,
            world_size
        );
    }
    let chunk_size = dim_size / world_size;
    x.narrow(dim, rank * chunk_size, chunk_size)?.contiguous()
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
        let weight = vb
            .get_with_hints_dtype((out_dim, in_dim), "weight", Shard::default(), DType::F8E4M3)
            .or_else(|_| {
                vb.get_with_hints_dtype((out_dim, in_dim), "weight", Shard::default(), DType::U8)
            })?;
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

#[allow(dead_code)]
pub struct TensorParallelRowLinear {
    linear: LinearX,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    bias: Option<Tensor>,
    dtype: DType,
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
        use candle_core::cuda_backend::cudarc::driver::DeviceSlice;
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
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_reduce BF16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(&src_slice, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let full_slice = s.as_cuda_slice::<f16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_reduce F16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(&src_slice, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F32 => {
                let full_slice = s.as_cuda_slice::<f32>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_reduce F32 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
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
    pub fn new(linear: LinearX, comm: Rc<Comm>, dtype: DType) -> Self {
        #[cfg(feature = "nccl")]
        let all_reduce = if comm.world_size() > 1 {
            Some(AllReduce { comm })
        } else {
            None
        };
        Self {
            linear,
            #[cfg(feature = "nccl")]
            all_reduce,
            bias: None,
            dtype,
        }
    }

    #[allow(unused_variables)]
    pub fn new_with_bias(
        linear: LinearX,
        bias: Option<Tensor>,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Self {
        #[cfg(feature = "nccl")]
        let all_reduce = if comm.world_size() > 1 {
            Some(AllReduce { comm })
        } else {
            None
        };
        Self {
            linear,
            #[cfg(feature = "nccl")]
            all_reduce,
            bias,
            dtype,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut xs = self.linear.forward(x)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            xs = xs.apply_op1_no_bwd(all_reduce)?;
        }
        if let Some(bias) = &self.bias {
            if bias.dtype() == xs.dtype() {
                xs = xs.broadcast_add(bias)?;
            } else {
                xs = xs.broadcast_add(&bias.to_dtype(xs.dtype())?)?;
            }
        }
        Ok(xs)
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
        let is_fp4_quant = if let Some(cfg) = quant_cfg {
            cfg.quant_method == "mxfp4" || cfg.quant_method == "nvfp4"
        } else {
            false
        };

        if quant_cfg.is_some() && !is_fp8_quant && !is_fp4_quant {
            candle_core::bail!(
                "Merged quantized weight is not supported at the moment, using ISQ instead!"
            );
        }

        if is_fp4_quant {
            let linear = crate::openai::models::linear::linear_no_bias_x(
                in_dim,
                out_dim,
                vb,
                shard(0, comm.rank(), comm.world_size()),
                &None,
                quant_cfg,
                dtype,
                None,
            )?;
            return Ok(Self {
                linears: vec![TensorParallelColumnLinear::new(linear)],
                biases: vec![],
                output_splits: None,
            });
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

            let weight = vb
                .get_with_hints_dtype((out_dim, in_dim), "weight", Shard::default(), DType::F8E4M3)
                .or_else(|_| {
                    vb.get_with_hints_dtype(
                        (out_dim, in_dim),
                        "weight",
                        Shard::default(),
                        DType::U8,
                    )
                })?;
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
                        crate::openai::models::layers::isq_high_precision_quant(quantized_type)
                            .to_string()
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
        Ok(Self::new_with_bias(linear, bs, comm, dtype))
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

pub fn rms_norm_sharded(
    size: usize,
    eps: f64,
    vb: VarBuilder,
    dtype: DType,
    add_unit_offset: bool,
    tp_shard: candle_nn::var_builder::Shard,
) -> Result<RmsNorm> {
    let weight = vb.get_with_hints_dtype(size, "weight", tp_shard, dtype)?;
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
    if vb.contains_tensor("scales") {
        // MLX NVFP4 embeddings use the same U32 packing as linear weights,
        // but embedding lookup needs a dequantized table rather than a GEMM.
        let no_shard = Shard::default();
        let weight = vb.get_with_hints_dtype(
            (vocab_size, hidden_size / 8),
            "weight",
            no_shard,
            DType::U32,
        )?;
        let scales = vb.get_with_hints_dtype(
            (vocab_size, hidden_size / 16),
            "scales",
            no_shard,
            DType::U8,
        )?;
        let dtype = match vb.dtype() {
            DType::F16 | DType::BF16 => vb.dtype(),
            _ => DType::BF16,
        };
        let embeddings = attention_rs::nvfp4_linear::mlx_dequant_embedding(
            &weight,
            &scales,
            vocab_size,
            hidden_size,
            dtype,
        )?;
        return Ok(Embedding::new(embeddings, hidden_size));
    }
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

#[allow(dead_code)]
pub struct AllGather {
    comm: Rc<Comm>,
    world_size: usize,
}

unsafe impl Sync for AllGather {}
unsafe impl Send for AllGather {}

impl AllGather {
    pub fn new(comm: Rc<Comm>) -> Self {
        let world_size = comm.world_size();
        Self { comm, world_size }
    }

    pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply_op1_no_bwd(self)
    }
}

impl CustomOp1 for AllGather {
    fn name(&self) -> &'static str {
        "allgather"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllGather is never used on cpu")
    }

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DeviceSlice;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::DType;
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let start_offset = l.start_offset();
        let total_elems = elem_count * self.world_size;

        let dst = match s.dtype() {
            DType::BF16 => {
                let full_slice = s.as_cuda_slice::<bf16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_gather BF16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<bf16>(total_elems) }.w()?;
                self.comm
                    .all_gather(&src_slice, &mut dst)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let full_slice = s.as_cuda_slice::<f16>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_gather F16 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<f16>(total_elems) }.w()?;
                self.comm
                    .all_gather(&src_slice, &mut dst)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F32 => {
                let full_slice = s.as_cuda_slice::<f32>()?;
                let full_len = full_slice.len();
                let end_offset = start_offset.saturating_add(elem_count);
                if end_offset > full_len {
                    candle_core::bail!(
                        "all_gather F32 slice out of bounds: start={}, elem_count={}, len={}",
                        start_offset,
                        elem_count,
                        full_len
                    );
                }
                let src_slice = full_slice.slice(start_offset..end_offset);
                let mut dst = unsafe { dev.alloc::<f32>(total_elems) }.w()?;
                self.comm
                    .all_gather(&src_slice, &mut dst)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype for all_gather: {dtype:?}"),
        };

        let dims = l.shape().dims();
        let mut out_dims = dims.to_vec();
        out_dims[0] *= self.world_size;
        Ok((dst, Shape::from_dims(&out_dims)))
    }
}

const VOCAB_PADDING_SIZE: usize = 64;

fn pad_vocab_size(vocab_size: usize, world_size: usize) -> usize {
    let padded = ((vocab_size + VOCAB_PADDING_SIZE - 1) / VOCAB_PADDING_SIZE) * VOCAB_PADDING_SIZE;
    let per_rank = ((padded + world_size - 1) / world_size) * world_size;
    ((per_rank + VOCAB_PADDING_SIZE - 1) / VOCAB_PADDING_SIZE) * VOCAB_PADDING_SIZE
}

#[allow(dead_code)]
pub struct VocabParallelLinear {
    linear: LinearX,
    #[cfg(feature = "nccl")]
    all_gather: Option<AllGather>,
    org_vocab_size: usize,
    dtype: DType,
}

impl VocabParallelLinear {
    #[allow(unused_variables)]
    pub fn load_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        comm: Rc<Comm>,
        quant: &Option<String>,
        quant_config: &Option<QuantConfig>,
        dtype: DType,
    ) -> Result<Self> {
        let world_size = comm.world_size();
        if world_size <= 1 {
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
            return Ok(Self {
                linear,
                #[cfg(feature = "nccl")]
                all_gather: None,
                org_vocab_size: out_dim,
                dtype,
            });
        }

        let padded_vocab = pad_vocab_size(out_dim, world_size);
        let linear = linear(
            in_dim,
            padded_vocab,
            vb,
            shard(0, comm.rank(), world_size),
            quant,
            quant_config,
            dtype,
            None,
        )?;

        #[cfg(feature = "nccl")]
        let all_gather = Some(AllGather::new(comm));

        Ok(Self {
            linear,
            #[cfg(feature = "nccl")]
            all_gather,
            org_vocab_size: out_dim,
            dtype,
        })
    }

    #[allow(unused_variables)]
    pub fn from_weight_bias(
        weight: Tensor,
        bias: Option<Tensor>,
        comm: Rc<Comm>,
        org_vocab_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        let world_size = comm.world_size();
        if world_size <= 1 {
            let linear = LinearX::Linear(Linear::new(weight, bias));
            return Ok(Self {
                linear,
                #[cfg(feature = "nccl")]
                all_gather: None,
                org_vocab_size,
                dtype,
            });
        }

        let vocab_dim = weight.dim(0)?;
        let padded_vocab = pad_vocab_size(org_vocab_size, world_size);
        let local_vocab = padded_vocab / world_size;
        let rank = comm.rank();

        let weight = if vocab_dim < padded_vocab {
            let hidden = weight.dim(1)?;
            let pad_rows = padded_vocab - vocab_dim;
            let padding = Tensor::zeros((pad_rows, hidden), weight.dtype(), weight.device())?;
            Tensor::cat(&[&weight, &padding], 0)?
        } else {
            weight
        };

        let local_weight = weight
            .narrow(0, rank * local_vocab, local_vocab)?
            .contiguous()?;

        let linear = LinearX::Linear(Linear::new(local_weight, bias));

        #[cfg(feature = "nccl")]
        let all_gather = Some(AllGather::new(comm));

        Ok(Self {
            linear,
            #[cfg(feature = "nccl")]
            all_gather,
            org_vocab_size,
            dtype,
        })
    }

    #[allow(unused_variables)]
    pub fn load_from_gguf(
        vb: &crate::openai::models::layers::quantized_var_builder::VarBuilder,
        tensor_name: &str,
        vocab_size: usize,
        comm: Rc<Comm>,
        dtype: DType,
    ) -> Result<Self> {
        use crate::openai::models::linear::QLinear;
        let world_size = comm.world_size();
        if world_size <= 1 {
            let ws = vb.get_no_shape(tensor_name)?;
            let qlinear = QLinear::from_arc_qtensor(ws, dtype);
            return Ok(Self {
                linear: LinearX::QLinear(qlinear),
                #[cfg(feature = "nccl")]
                all_gather: None,
                org_vocab_size: vocab_size,
                dtype,
            });
        }

        let padded_vocab = pad_vocab_size(vocab_size, world_size);
        let rank = comm.rank();
        let local_vocab = padded_vocab / world_size;

        let ws = vb.get_sharded_no_shape(tensor_name, 0, rank, world_size)?;
        let actual_local = ws.shape().dims()[0];
        let ws = if actual_local < local_vocab {
            let dequant = ws.dequantize_f16(vb.device())?;
            let hidden = dequant.dim(1)?;
            let pad_rows = local_vocab - actual_local;
            let padding = Tensor::zeros((pad_rows, hidden), DType::F16, vb.device())?;
            let padded = Tensor::cat(&[&dequant, &padding], 0)?;
            let wdtype = if hidden % ws.dtype().block_size() != 0 {
                GgmlDType::Q8_0
            } else {
                ws.dtype()
            };
            Arc::new(QTensor::quantize_owned(padded, wdtype)?)
        } else {
            ws
        };
        let qlinear = QLinear::from_arc_qtensor(ws, dtype);
        let linear = LinearX::QLinear(qlinear);

        #[cfg(feature = "nccl")]
        let all_gather = Some(AllGather::new(comm));

        Ok(Self {
            linear,
            #[cfg(feature = "nccl")]
            all_gather,
            org_vocab_size: vocab_size,
            dtype,
        })
    }

    #[allow(unused_variables)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.linear.forward(x)?;

        #[cfg(feature = "nccl")]
        if let Some(all_gather) = &self.all_gather {
            let gathered = all_gather.apply(&logits)?;

            let ws = all_gather.world_size;
            let local_vocab = logits.dim(logits.dims().len() - 1)?;
            let batch = logits.dims()[..logits.dims().len() - 1]
                .iter()
                .product::<usize>();

            let gathered = gathered.reshape((ws, batch, local_vocab))?;
            let gathered = gathered.transpose(0, 1)?.contiguous()?;
            let full_vocab = ws * local_vocab;
            let logits = gathered.reshape((batch, full_vocab))?;

            let orig_dims = &x.dims()[..x.dims().len() - 1];
            let logits = if orig_dims.len() > 1 {
                let mut shape = orig_dims.to_vec();
                shape.push(full_vocab);
                logits.reshape(shape)?
            } else {
                logits
            };

            if full_vocab > self.org_vocab_size {
                let last_dim = logits.dims().len() - 1;
                return logits.narrow(last_dim, 0, self.org_vocab_size);
            }
            return Ok(logits);
        }

        Ok(logits)
    }
}
