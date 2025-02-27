use candle_core::CustomOp1;
use candle_core::{CpuStorage, Layout, Module, Result, Shape, Tensor};
use candle_nn::var_builder::Shard;
pub use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, LayerNorm, Linear, RmsNorm};
#[cfg(feature = "nccl")]
pub use cudarc::nccl::safe::Comm;
#[cfg(not(feature = "nccl"))]
pub struct Comm {}
#[cfg(not(feature = "nccl"))]
impl Comm {
    //dummy Comm
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        1
    }
}

pub use std::rc::Rc;

pub struct ReplicatedLinear {
    linear: Linear,
}

pub struct TensorParallelColumnLinear {
    linear: Linear,
}

impl TensorParallelColumnLinear {
    pub fn new(linear: Linear) -> Self {
        Self { linear }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

pub struct TensorParallelRowLinear {
    linear: Linear,
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
        use candle_core::cuda_backend::cudarc::driver::DeviceSlice;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::DType;
        use cudarc::nccl::safe::ReduceOp;
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_cuda_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

impl TensorParallelRowLinear {
    pub fn new(linear: Linear, comm: Rc<Comm>) -> Self {
        let all_reduce = AllReduce { comm };
        Self {
            linear,
            all_reduce,
            bias: None,
        }
    }

    pub fn new_with_bias(linear: Linear, bias: Option<Tensor>, comm: Rc<Comm>) -> Self {
        let all_reduce = AllReduce { comm };
        Self {
            linear,
            all_reduce,
            bias,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut xs = self.linear.forward(x)?.apply_op1_no_bwd(&self.all_reduce)?;
        if let Some(bias) = &self.bias {
            xs = xs.broadcast_add(bias)?;
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

impl TensorParallelColumnLinear {
    pub fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(0, rank, size))?;
        Ok(Self::new(Linear::new(weight, None)))
    }

    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard(0, rank, size))?;
        Ok(Self::new(Linear::new(weight, None)))
    }

    pub fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weights: Vec<_> = prefixes
            .iter()
            .map(|p| vb.pp(p).get_with_hints((), "weight", shard(0, rank, size)))
            .collect::<Result<Vec<_>>>()?;
        let weight = Tensor::cat(&weights, 0)?.contiguous()?;
        Ok(Self::new(Linear::new(weight, None)))
    }
}

impl TensorParallelRowLinear {
    pub fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(1, rank, size))?;
        Ok(Self::new(Linear::new(weight, None), comm))
    }

    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilder,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard(1, rank, size))?;
        let bs = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };
        Ok(Self::new_with_bias(Linear::new(weight, None), bs, comm))
    }
}

pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get_with_hints(size, "weight", shard(0, 0, 1))?;
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

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", shard(0, 0, 1))?;
    let bs = vb.get((out_dim,), "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

impl ReplicatedLinear {
    pub fn from(linear: Linear) -> Result<Self> {
        Ok(Self { linear })
    }

    pub fn load_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear_no_bias(in_dim, out_dim, vb)?,
        })
    }

    pub fn load_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        if !bias {
            ReplicatedLinear::load_no_bias(in_dim, out_dim, vb)
        } else {
            Ok(Self {
                linear: linear_b(in_dim, out_dim, bias, vb)?,
            })
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(&x)
    }
}
