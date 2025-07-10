use crate::openai::models::linear::{linear_no_bias_x as linear, LinearX as Linear};
use candle_core::CustomOp1;
use candle_core::{CpuStorage, Layout, Module, Result, Shape, Tensor};
use candle_nn::var_builder::Shard;
pub use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, LayerNorm, RmsNorm};
#[cfg(feature = "nccl")]
pub use cudarc::nccl::safe::{Comm, Id};
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
    linear: Linear,
    bias: Option<Tensor>,
}

pub struct TensorParallelColumnLinear {
    linear: Linear,
    bias: Option<Tensor>,
}

impl TensorParallelColumnLinear {
    pub fn new(linear: Linear) -> Self {
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
    linears: Vec<TensorParallelColumnLinear>,
    biases: Vec<Option<Tensor>>,
}

impl MergedParallelColumnLinear {
    pub fn new(linears: Vec<TensorParallelColumnLinear>) -> Self {
        Self {
            linears,
            biases: Vec::new(),
        }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut xss = Vec::<Tensor>::new();
        for i in 0..self.linears.len() {
            let mut xs = self.linears[i].forward(x)?;
            if let Some(bias) = &self.biases[i] {
                xs = xs.broadcast_add(bias)?;
            }
            xss.push(xs);
        }
        Ok(xss)
    }
}

pub struct TensorParallelRowLinear {
    linear: Linear,
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
    #[allow(unused_variables)]
    pub fn new(linear: Linear, comm: Rc<Comm>) -> Self {
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
    pub fn new_with_bias(linear: Linear, bias: Option<Tensor>, comm: Rc<Comm>) -> Self {
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

    // pub fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Rc<Comm>) -> Result<Self> {
    //     let rank = comm.rank();
    //     let size = comm.world_size();
    //     let weights: Vec<_> = prefixes
    //         .iter()
    //         .map(|p| vb.pp(p).get_with_hints((), "weight", shard(0, rank, size)))
    //         .collect::<Result<Vec<_>>>()?;
    //     let weight = Tensor::cat(&weights, 0)?.contiguous()?;
    //     Ok(Self::new(Linear::new(weight, None)))
    // }
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
        if quant.is_some() || quant_config.is_some() {
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
    pub fn from(linear: Linear, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self { linear, bias })
    }

    pub fn from_weight_bias(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let linear = Linear::new(weight, None, &None, &None);
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
