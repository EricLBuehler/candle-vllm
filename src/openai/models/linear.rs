//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu};
//! use candle_nn::{Linear, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use super::QuantConfig;
use crate::backend::gptq::{gptq_matmul, marlin_weight_repack};
use crate::candle::Module;
use crate::candle::{
    quantized::{gguf_file, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use crate::openai::distributed::shard;
use candle_core::quantized;
pub use candle_nn::var_builder::Shard;
pub use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;

use std::sync::Arc;
use tracing::warn;
#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    scales: Option<Tensor>,
    qzeros: Option<Tensor>,
    g_idx: Option<Tensor>,
    workspace: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight,
            bias,
            scales: None,
            qzeros: None,
            g_idx: None,
            workspace: None,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn scales(&self) -> Option<&Tensor> {
        self.scales.as_ref()
    }

    pub fn qzeros(&self) -> Option<&Tensor> {
        self.qzeros.as_ref()
    }

    pub fn g_idx(&self) -> Option<&Tensor> {
        self.g_idx.as_ref()
    }

    pub fn workspace(&self) -> Option<&Tensor> {
        self.workspace.as_ref()
    }

    #[cfg(feature = "cuda")]
    pub fn reload(&mut self) -> Result<()> {
        self.weight = self.weight.reload()?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn offload(&mut self) -> Result<()> {
        self.weight = self.weight.offload()?;
        Ok(())
    }
}

//Revised to improve performance for batched matmul
//Remember use this linear layer throughout all of the models
impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match *x.dims() {
            [b1, seq_len, _, _] => {
                if seq_len > 1 {
                    self.weight.broadcast_left((b1, seq_len))?.t()?
                } else {
                    self.weight.t()?
                }
            }
            [bsize, seq_len, _] => {
                if seq_len > 1 {
                    self.weight.broadcast_left(bsize)?.t()?
                } else {
                    self.weight.t()?
                }
            }
            _ => self.weight.t()?,
        };
        let x = match *x.dims() {
            [bsize, seq_len, dim1, dim2] => {
                if seq_len > 1 {
                    x.matmul(&w)?
                } else {
                    let wdim = w.dims()[w.dims().len() - 1];
                    x.reshape((bsize * seq_len, dim1, dim2))?
                        .matmul(&w)?
                        .reshape((bsize, seq_len, dim1, wdim))?
                }
            }
            [bsize, seq_len, dim] => {
                if seq_len > 1 {
                    x.matmul(&w)?
                } else {
                    let wdim = w.dims()[w.dims().len() - 1];
                    x.reshape((bsize * seq_len, dim))?
                        .matmul(&w)?
                        .reshape((bsize, seq_len, wdim))?
                }
            }
            _ => x.matmul(&w)?,
        };

        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shard: Shard,
) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
    Ok(Linear::new(weight, None))
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder, shard: Shard) -> Result<Linear> {
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
    let bs = vb.get((out_dim,), "bias")?;
    Ok(Linear::new(ws, Some(bs)))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    shard: Shard,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb, shard)
    } else {
        linear_no_bias(in_dim, out_dim, vb, shard)
    }
}

pub fn qlinear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shards: Shard,
    quant_config: &Option<QuantConfig>,
    bias: bool,
    dtype: DType,
) -> Result<Linear> {
    match quant_config {
        Some(cfg) => {
            let marlin_compatible = if (cfg.quant_method != "gptq" && cfg.quant_method != "awq")
                || (cfg.bits != 4 && cfg.bits != 8)
            {
                false
            } else {
                true
            };
            let marlin_format = cfg.checkpoint_format.is_some()
                && cfg.checkpoint_format.as_ref().unwrap() == "marlin";

            let ws = vb.get_with_hints_dtype(
                if cfg.quant_method == "gptq" {
                    //quantized gptq (k/pack_factor, n) format
                    (
                        in_dim / (32 / cfg.bits) / if marlin_format { 2 } else { 1 },
                        out_dim * if marlin_format { 2 } else { 1 },
                    )
                } else {
                    //quantized awq (k, n/pack_factor) format
                    (
                        in_dim * if marlin_format { 2 } else { 1 },
                        out_dim / (32 / cfg.bits) / if marlin_format { 2 } else { 1 },
                    )
                },
                if marlin_format { "B" } else { "qweight" },
                shards,
                DType::U32,
            )?;

            let scale_and_zero_size = in_dim / (cfg.group_size as usize);
            let scales = vb
                .get_with_hints_dtype(
                    (scale_and_zero_size, out_dim),
                    if marlin_format { "s" } else { "scales" },
                    shards,
                    DType::F16,
                )?
                .to_dtype(dtype)?;

            let in_dim_partition = if shards.dim == 0 {
                in_dim / shards.world_size
            } else {
                in_dim
            };

            let out_dim_partition = if shards.dim == 1 {
                out_dim / shards.world_size
            } else {
                out_dim
            };

            let bs = if bias {
                let bs = vb
                    .get_with_hints_dtype(
                        (out_dim,),
                        "bias",
                        shard(0, shards.rank, shards.world_size),
                        DType::F16,
                    )?
                    .to_dtype(dtype)?;
                Some(bs)
            } else {
                None
            };

            if marlin_format {
                let workspace = Tensor::zeros(out_dim_partition, DType::U32, vb.device())?;
                //marlin weight file
                Ok(Linear {
                    weight: ws,
                    bias: bs,
                    scales: Some(scales),
                    qzeros: None,
                    g_idx: None,
                    workspace: Some(workspace),
                })
            } else {
                let qzeros = vb.get_with_hints_dtype(
                    (scale_and_zero_size, out_dim / (32 / cfg.bits)),
                    "qzeros",
                    shards,
                    DType::U32,
                )?;
                let g_idx = if cfg.quant_method == "gptq" {
                    let mut g_idx = vb.get_with_hints_dtype(
                        (in_dim,),
                        "g_idx",
                        Default::default(),
                        DType::U32,
                    )?;
                    g_idx = if shards.world_size > 1 {
                        let dim_size = g_idx.dims()[0];
                        let start = shards.rank * (dim_size / shards.world_size);
                        g_idx
                            .narrow(0, start, dim_size / shards.world_size)?
                            .contiguous()?
                    } else {
                        g_idx
                    };
                    Some(g_idx)
                } else {
                    None
                };

                if (cfg.sym.is_some() && !cfg.sym.unwrap())
                    || cfg.bits != 4
                    || !matches!(cfg.group_size, 64 | 128 | -1)
                    || (cfg.desc_act.is_some()
                        && cfg.desc_act.unwrap()
                        && cfg.quant_method == "gptq")
                {
                    //only model with 4-bit and desc_act==false can be repacked to marlin format
                    if cfg.quant_method == "marlin" {
                        warn!("The current GPTQ model does no compatible with marlin format because one of the following conditions: !cfg.sym || cfg.bits != 4 || (cfg.group_size != 128 && cfg.group_size != -1) || (cfg.desc_act == true)");
                    }
                    //conventional gptq format
                    Ok(Linear {
                        weight: ws,
                        bias: bs,
                        scales: Some(scales),
                        qzeros: Some(qzeros),
                        g_idx,
                        workspace: None,
                    })
                } else {
                    //repack gptq format to marlin
                    fn get_scale_perms() -> (Vec<u32>, Vec<u32>) {
                        let mut scale_perm: Vec<u32> = Vec::new();
                        for i in 0..8 {
                            scale_perm.extend((0..8).map(|j| i + 8 * j));
                        }
                        let mut scale_perm_single: Vec<u32> = Vec::new();
                        for i in 0..4 {
                            scale_perm_single
                                .extend([0, 1, 8, 9, 16, 17, 24, 25].iter().map(|&j| 2 * i + j));
                        }
                        (scale_perm, scale_perm_single)
                    }

                    fn marlin_permute_scales(
                        s: &Tensor,
                        size_k: usize,
                        size_n: usize,
                        group_size: i32,
                        _num_bits: u32,
                    ) -> Result<Tensor> {
                        let (scale_perm, scale_perm_single) = get_scale_perms();
                        let s = if (group_size as usize) < size_k && group_size != -1 {
                            let s = s.reshape(((), scale_perm.len()))?;
                            let scale_perm_tensor =
                                Tensor::from_slice(&scale_perm, scale_perm.len(), s.device())?;
                            s.index_select(&scale_perm_tensor, 1)?
                        } else {
                            let s = s.reshape(((), scale_perm_single.len()))?;
                            let scale_perm_single_tensor = Tensor::from_slice(
                                &scale_perm_single,
                                scale_perm_single.len(),
                                s.device(),
                            )?;
                            s.index_select(&scale_perm_single_tensor, 1)?
                        };

                        let s = s.reshape(((), size_n))?.contiguous()?;
                        Ok(s)
                    }

                    let ws = if marlin_compatible {
                        marlin_weight_repack(&ws, cfg.bits as i32, cfg.quant_method != "gptq")?
                    } else {
                        ws
                    }; //repack to marlin format

                    let scales = if marlin_compatible {
                        marlin_permute_scales(
                            &scales,
                            in_dim_partition,
                            out_dim_partition,
                            cfg.group_size,
                            cfg.bits as u32,
                        )?
                    } else {
                        scales
                    };

                    let workspace = Tensor::zeros(out_dim_partition, DType::U32, vb.device())?;
                    Ok(Linear {
                        weight: ws,
                        bias: bs,
                        scales: Some(scales),
                        qzeros: Some(qzeros),
                        g_idx,
                        workspace: Some(workspace),
                    })
                }
            }
        }
        None => linear_b(in_dim, out_dim, bias, vb, shards),
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
    scales: Option<Tensor>,
    qzeros: Option<Tensor>,
    g_idx: Option<Tensor>,
    workspace: Option<Tensor>,
    group_size: i32,
    bits: i32,
    dtype: DType,
    is_awq: bool,
    transposed_weight: bool,
}

impl QLinear {
    pub fn new<R: std::io::Read + std::io::Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let w = ct.tensor(r, &format!("{name}.weight"), device)?;
        let b = ct.tensor(r, &format!("{name}.bias"), device)?;
        let inner = QMatMul::from_qtensor(w)?;
        let bias = b.dequantize(device)?;
        Ok(Self {
            inner,
            bias: Some(bias),
            scales: None,
            qzeros: None,
            g_idx: None,
            workspace: None,
            group_size: 0,
            bits: 0,
            dtype: DType::F32,
            is_awq: false,
            transposed_weight: false,
        })
    }

    pub fn from_linear(linear: Linear, group_size: i32, bits: i32, is_awq: bool) -> Self {
        Self {
            inner: QMatMul::Tensor(linear.weight().clone()),
            bias: linear.bias().cloned(),
            scales: linear.scales().cloned(),
            qzeros: linear.qzeros().cloned(),
            g_idx: linear.g_idx().cloned(),
            workspace: linear.workspace().cloned(),
            group_size,
            bits,
            dtype: linear.weight().dtype(),
            is_awq,
            transposed_weight: false,
        }
    }

    pub fn from_parts(w: Tensor, b: Option<Tensor>) -> Self {
        let dtype = w.dtype();
        Self {
            inner: QMatMul::Tensor(w),
            bias: b,
            scales: None,
            qzeros: None,
            g_idx: None,
            workspace: None,
            group_size: 0,
            bits: 0,
            dtype,
            is_awq: false,
            transposed_weight: false,
        }
    }

    pub fn from_qparts(w: QTensor, b: Option<Tensor>) -> Self {
        if let Some(ref b) = b {
            assert_eq!(b.dtype(), DType::F32);
        }
        Self {
            inner: QMatMul::QTensor(Arc::new(w)),
            bias: b,
            scales: None,
            qzeros: None,
            g_idx: None,
            workspace: None,
            group_size: 0,
            bits: 0,
            dtype: DType::F32,
            is_awq: false,
            transposed_weight: false,
        }
    }

    pub fn from_qparts_x(w: QTensor, b: Option<Tensor>, dtype: DType, transposed: bool) -> Self {
        let bx = match b {
            Some(b_) => {
                if b_.dtype() != DType::F32 {
                    Some(b_.to_dtype(DType::F32).unwrap())
                } else {
                    Some(b_)
                }
            }
            _ => None,
        };

        Self {
            inner: QMatMul::QTensor(Arc::new(w)),
            bias: bx,
            scales: None,
            qzeros: None,
            g_idx: None,
            workspace: None,
            group_size: 0,
            bits: 0,
            dtype,
            is_awq: false,
            transposed_weight: transposed,
        }
    }

    pub fn from_linear_x(
        linear: Linear,
        quant: String,
        quant_config: &Option<QuantConfig>,
    ) -> Self {
        match quant_config {
            Some(cfg) => {
                assert!(
                    cfg.quant_method == "gptq"
                        || cfg.quant_method == "awq"
                        || cfg.quant_method == "marlin"
                        || quant == "marlin"
                );
                QLinear::from_linear(
                    linear,
                    cfg.group_size,
                    cfg.bits as i32,
                    cfg.quant_method == "awq",
                )
            }
            None => {
                use quantized::GgmlDType;
                let ggml_dtype = match quant.as_str() {
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
                let weight = linear.weight();
                let qbias = linear.bias().cloned();
                let dtype = weight.dtype();
                let dims = weight.dims();
                let (w, transposed) = if dims[dims.len() - 1] % ggml_dtype.block_size() != 0 {
                    (weight.t().unwrap().contiguous().unwrap(), true)
                } else {
                    (weight.to_owned(), false)
                };

                let qtensor = QTensor::quantize(&w, ggml_dtype).unwrap();
                QLinear::from_qparts_x(qtensor, qbias, dtype, transposed)
            }
        }
    }

    pub fn from_old_and_qmatmul(inner: QMatMul, old: &Self) -> Self {
        Self {
            inner,
            bias: old.bias.clone(),
            scales: None,
            qzeros: None,
            g_idx: None,
            workspace: None,
            group_size: 0,
            bits: 0,
            dtype: old.dtype,
            is_awq: false,
            transposed_weight: false,
        }
    }

    pub fn inner(&mut self) -> &mut QMatMul {
        &mut self.inner
    }

    pub fn inner_ref(&self) -> &QMatMul {
        &self.inner
    }

    pub fn is_quant(&self) -> bool {
        matches!(self.inner, QMatMul::QTensor(_))
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }

    #[cfg(feature = "cuda")]
    pub fn offload(&mut self) -> Result<()> {
        let w = match &self.inner {
            QMatMul::Tensor(qw) => qw.offload()?,
            _ => {
                unreachable!()
            }
        };
        self.inner = QMatMul::Tensor(w);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn reload(&mut self) -> Result<()> {
        let w = match &self.inner {
            QMatMul::Tensor(qw) => qw.reload()?,
            _ => {
                unreachable!()
            }
        };
        self.inner = QMatMul::Tensor(w);
        Ok(())
    }
}

impl QLinear {
    pub fn forward_no_dequant(&self, x: &Tensor) -> Result<Tensor> {
        let xs = match *x.dims() {
            [bsize, seq_len, dim1, dim2] => {
                if seq_len > 1 {
                    x.to_dtype(DType::F32)?
                } else {
                    x.reshape((bsize, dim1, dim2))?.to_dtype(DType::F32)?
                }
            }
            [bsize, seq_len, dim] => {
                if seq_len > 1 {
                    x.to_dtype(DType::F32)?
                } else {
                    x.reshape((bsize, dim))?.to_dtype(DType::F32)?
                }
            }
            _ => x.to_dtype(DType::F32)?,
        };
        let xs = match *x.dims() {
            [bsize, seq_len, dim1, _] => {
                if seq_len > 1 {
                    QMatMul::forward(&self.inner, &xs)?
                } else {
                    QMatMul::forward(&self.inner, &xs)?.reshape((bsize, seq_len, dim1, ()))?
                }
            }
            [bsize, seq_len, _] => {
                if seq_len > 1 {
                    QMatMul::forward(&self.inner, &xs)?
                } else {
                    QMatMul::forward(&self.inner, &xs)?.reshape((bsize, seq_len, ()))?
                }
            }
            _ => QMatMul::forward(&self.inner, &xs)?,
        };

        if let Some(bias) = &self.bias {
            xs.broadcast_add(bias)?.to_dtype(self.dtype)
        } else {
            xs.to_dtype(self.dtype)
        }
    }

    pub fn forward_via_dequant(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let w = self.inner.dequantize_f16()?.to_dtype(in_dtype)?;
        let x = match *x.dims() {
            [bsize, seq_len, dim1, dim2] => {
                if seq_len > 1 {
                    let w = w.broadcast_left((bsize, seq_len))?;
                    x.matmul(&w)?
                } else {
                    let wdim = w.dims()[w.dims().len() - 1];
                    x.reshape((bsize * seq_len, dim1, dim2))?
                        .matmul(&w)?
                        .reshape((bsize, seq_len, dim1, wdim))?
                }
            }
            [bsize, seq_len, dim] => {
                if seq_len > 1 {
                    let w = w.broadcast_left(bsize)?;
                    x.matmul(&w)?
                } else {
                    let wdim = w.dims()[w.dims().len() - 1];
                    x.reshape((bsize * seq_len, dim))?
                        .matmul(&w)?
                        .reshape((bsize, seq_len, wdim))?
                }
            }
            _ => x.matmul(&w)?,
        };
        // let x = x.to_dtype(DType::F16)?;
        if let Some(bias) = &self.bias {
            x.broadcast_add(bias)
        } else {
            Ok(x)
        }
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match (
            &self.inner,
            &self.scales,
            &self.qzeros,
            &self.g_idx,
            &self.workspace,
        ) {
            (QMatMul::Tensor(qw), Some(scale), qzeros, g_idx, workspace) => {
                //gptq (only f16/bf16 inputs for marlin format)
                let x = match *x.dims() {
                    [bsize, seq_len, dim1, dim2] => {
                        let x = x.reshape((bsize * seq_len, dim1, dim2))?;
                        let o = gptq_matmul(
                            &x,
                            qw,
                            scale,
                            qzeros,
                            g_idx,
                            workspace,
                            self.bits,
                            self.group_size,
                            self.is_awq,
                        )?;
                        o.reshape((bsize, seq_len, dim1, ()))?
                    }
                    [_, _, _] => gptq_matmul(
                        x,
                        qw,
                        scale,
                        qzeros,
                        g_idx,
                        workspace,
                        self.bits,
                        self.group_size,
                        self.is_awq,
                    )?,
                    [seq_len, dim] => {
                        let x = x.reshape((1, seq_len, dim))?;
                        let o = gptq_matmul(
                            &x,
                            qw,
                            scale,
                            qzeros,
                            g_idx,
                            workspace,
                            self.bits,
                            self.group_size,
                            self.is_awq,
                        )?;
                        o.reshape((seq_len, ()))?
                    }
                    _ => panic!("Invalid input format!"),
                };

                if let Some(bias) = &self.bias {
                    x.broadcast_add(bias)
                } else {
                    Ok(x)
                }
            }
            _ => {
                if self.transposed_weight {
                    self.forward_via_dequant(x)
                } else {
                    self.forward_no_dequant(x)
                }
            }
        }
    }
}

/// FP8 Linear layer with block-wise scales
#[derive(Debug, Clone)]
pub struct LnFp8 {
    pub weight: Tensor,
    pub weight_scale: Tensor,
    pub bias: Option<Tensor>,
    pub weight_block_size: Vec<usize>,
    pub sm_version: usize,
}

impl LnFp8 {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        shard: Shard,
        quant_cfg: &QuantConfig,
    ) -> Result<Self> {
        let block_size = quant_cfg
            .weight_block_size
            .clone()
            .unwrap_or(vec![128, 128]);
        if block_size.len() != 2 {
            candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
        }

        let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8)?;

        let by = block_size[0];
        let bx = block_size[1];

        let scale_dim0 = (out_dim + by - 1) / by;
        let scale_dim1 = (in_dim + bx - 1) / bx;

        let weight_scale = match vb.get_with_hints_dtype(
            (scale_dim0, scale_dim1),
            "weight_scale",
            shard,
            DType::F32,
        ) {
            Ok(s) => s,
            Err(_) => vb
                .get_with_hints_dtype(
                    (scale_dim0, scale_dim1),
                    "weight_scale_inv",
                    shard,
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

        #[cfg(feature = "cutlass")]
        let weight_scale = if sm_version >= 100 {
            weight_scale.t()?
        } else if sm_version >= 90 {
            weight_scale.t()?.contiguous()?
        } else {
            weight_scale
        };

        // Load bias if present
        let bias = vb.get((out_dim,), "bias");
        let bias = if bias.is_ok() {
            let bs = bias.unwrap();
            let bs = if shard.world_size > 1 {
                let dim_size = bs.dim(0)?;
                let start = shard.rank * (dim_size / shard.world_size);
                bs.narrow(0, start, dim_size / shard.world_size)?
                    .contiguous()?
            } else {
                bs
            };
            Some(bs)
        } else {
            None
        };

        Ok(Self {
            weight,
            weight_scale,
            bias,
            weight_block_size: block_size,
            sm_version,
        })
    }
}

fn load_ln_fp8_with_hints(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shard: Shard,
    quant_cfg: &QuantConfig,
    load_bias: bool,
) -> Result<LnFp8> {
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
            candle_core::bail!("LnFp8: unsupported shard dim {} for {}", shard.dim, name);
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
                "LnFp8: unexpected {} shape ({}, {}), shard dim 0 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim0
            )
        } else {
            let local = global_dim1 / shard.world_size;
            if d1 == local {
                return Ok(t);
            }
            if d1 == global_dim1 {
                return t.narrow(1, shard.rank * local, local)?.contiguous();
            }
            candle_core::bail!(
                "LnFp8: unexpected {} shape ({}, {}), shard dim 1 expects local {} or global {}",
                name,
                d0,
                d1,
                local,
                global_dim1
            )
        }
    }

    fn normalize_sharded_1d(
        t: Tensor,
        shard: Shard,
        global_dim: usize,
        name: &str,
    ) -> Result<Tensor> {
        if shard.world_size <= 1 {
            return Ok(t);
        }
        let d0 = t.dim(0)?;
        let local = global_dim / shard.world_size;
        if d0 == local {
            return Ok(t);
        }
        if d0 == global_dim {
            return t.narrow(0, shard.rank * local, local)?.contiguous();
        }
        candle_core::bail!(
            "LnFp8: unexpected {} shape ({}), expects local {} or global {}",
            name,
            d0,
            local,
            global_dim
        )
    }

    let block_size = quant_cfg
        .weight_block_size
        .clone()
        .unwrap_or(vec![128, 128]);
    if block_size.len() != 2 {
        candle_core::bail!("LnFp8: weight_block_size must have 2 elements");
    }

    let by = block_size[0];
    let bx = block_size[1];
    let scale_dim0 = (out_dim + by - 1) / by;
    let scale_dim1 = (in_dim + bx - 1) / bx;

    let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", shard, DType::U8)?;
    let weight = normalize_sharded_2d(weight, shard, out_dim, in_dim, "weight")?;
    let weight_scale = match vb.get_with_hints_dtype(
        (scale_dim0, scale_dim1),
        "weight_scale",
        shard,
        DType::F32,
    ) {
        Ok(s) => s,
        Err(_) => vb
            .get_with_hints_dtype(
                (scale_dim0, scale_dim1),
                "weight_scale_inv",
                shard,
                DType::F32,
            )
            .map_err(|_| {
                candle_core::Error::Msg("LnFp8: Missing weight_scale or weight_scale_inv".into())
            })?,
    };
    let weight_scale = normalize_sharded_2d(
        weight_scale,
        shard,
        scale_dim0,
        scale_dim1,
        "weight_scale(_inv)",
    )?;

    #[cfg(feature = "cuda")]
    let sm_version =
        attention_rs::cuda_utils::sm_version(vb.device().as_cuda_device()?).unwrap_or(0) as usize;

    #[cfg(not(feature = "cuda"))]
    let sm_version = 0;

    #[cfg(feature = "cutlass")]
    let weight_scale = if sm_version >= 100 {
        weight_scale.t()?
    } else if sm_version >= 90 {
        weight_scale.t()?.contiguous()?
    } else {
        weight_scale
    };

    let bias = if load_bias {
        vb.get_with_hints_dtype((out_dim,), "bias", shard, DType::F32)
            .ok()
            .map(|b| normalize_sharded_1d(b, shard, out_dim, "bias"))
            .transpose()?
    } else {
        None
    };

    Ok(LnFp8 {
        weight,
        weight_scale,
        bias,
        weight_block_size: block_size,
        sm_version,
    })
}

impl Module for LnFp8 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, in_dim) = match x.dims() {
            [b, s, d] => (*b, *s, *d),
            [b, d] => (*b, 1, *d),
            _ => candle_core::bail!("LnFp8: Input should be 2D or 3D"),
        };

        let m = b_sz * seq_len;
        let k = in_dim;

        let x_2d = x.reshape((m, k))?;

        #[cfg(feature = "cutlass")]
        let out = if self.sm_version >= 90 {
            attention_rs::fp8_linear::fp8_matmul_cutlass(
                &x_2d,
                &self.weight.t()?,
                &self.weight_scale,
                &self.weight_block_size,
            )?
        } else {
            // slower path
            attention_rs::fp8_linear::fp8_matmul(
                &x_2d,
                &self.weight,
                &self.weight_scale,
                &self.weight_block_size,
            )?
        };

        #[cfg(not(feature = "cutlass"))]
        let out = attention_rs::fp8_linear::fp8_matmul(
            &x_2d,
            &self.weight,
            &self.weight_scale,
            &self.weight_block_size,
        )?;

        let (_, out_dim) = out.dims2()?;
        let out = if seq_len > 1 {
            out.reshape((b_sz, seq_len, out_dim))?
        } else {
            out
        };

        match &self.bias {
            None => Ok(out),
            Some(bias) => out.broadcast_add(bias),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LinearX {
    Linear(Linear),
    QLinear(QLinear),
    LnFp8(LnFp8),
}

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            LinearX::Linear(ln) => ln.forward(x),
            LinearX::QLinear(ln) => ln.forward(x),
            LinearX::LnFp8(ln) => ln.forward(x),
        }
    }
}

impl LinearX {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        quant: &Option<String>,
        quant_config: &Option<QuantConfig>,
    ) -> Self {
        let ln = Linear::new(weight, bias);
        if let Some(quantized_type) = quant {
            LinearX::QLinear(QLinear::from_linear_x(
                ln,
                quantized_type.clone(),
                quant_config,
            ))
        } else {
            LinearX::Linear(ln)
        }
    }

    #[cfg(feature = "cuda")]
    pub fn offload(&mut self) -> Result<()> {
        match self {
            LinearX::Linear(ln) => ln.offload(),
            LinearX::QLinear(ln) => ln.offload(),
            LinearX::LnFp8(_) => Ok(()), // FP8 weights are already small
        }
    }

    #[cfg(feature = "cuda")]
    pub fn reload(&mut self) -> Result<()> {
        match self {
            LinearX::Linear(ln) => ln.reload(),
            LinearX::QLinear(ln) => ln.reload(),
            LinearX::LnFp8(_) => Ok(()), // FP8 weights are already small
        }
    }
}

fn should_bypass_quant_for_module(vb: &VarBuilder, quant_config: &Option<QuantConfig>) -> bool {
    fn module_path_matches_not_convert(module_path: &str, item: &str) -> bool {
        let module_path = module_path.trim_end_matches(".weight");
        let item = item.trim_end_matches(".weight");
        module_path == item
            || module_path.ends_with(item)
            || module_path.ends_with(&format!(".{item}"))
            || item.ends_with(module_path)
            || item.ends_with(&format!(".{module_path}"))
    }

    let Some(cfg) = quant_config else {
        return false;
    };
    if cfg.quant_method != "fp8" {
        return false;
    }
    let Some(skip_modules) = &cfg.modules_to_not_convert else {
        return false;
    };
    let prefix = vb.prefix();
    if prefix.is_empty() {
        return false;
    }
    skip_modules
        .iter()
        .any(|m| module_path_matches_not_convert(&prefix, m))
}

pub fn linear_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shard: Shard,
    quant: &Option<String>,
    quant_config: &Option<QuantConfig>,
    dtype: DType,
) -> Result<LinearX> {
    let bypass_quant = should_bypass_quant_for_module(&vb, quant_config);
    let quant_config_local = if bypass_quant {
        None
    } else {
        quant_config.clone()
    };
    let quant_local = if bypass_quant { None } else { quant.clone() };

    // Check for FP8 quantization first
    if let Some(cfg) = &quant_config_local {
        if cfg.quant_method == "fp8" {
            let has_fp8_scale =
                vb.contains_tensor("weight_scale") || vb.contains_tensor("weight_scale_inv");
            if !has_fp8_scale {
                let weight_probe = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
                if matches!(
                    weight_probe.dtype(),
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64
                ) {
                    let ln = linear(in_dim, out_dim, vb, shard)?;
                    return Ok(LinearX::Linear(ln));
                }
            }
            let ln = load_ln_fp8_with_hints(in_dim, out_dim, vb, shard, cfg, true)?;
            return Ok(LinearX::LnFp8(ln));
        }
        if matches!(cfg.quant_method.as_str(), "gptq" | "awq" | "marlin") {
            let ln = qlinear(in_dim, out_dim, vb, shard, &quant_config_local, true, dtype)?;
            return Ok(LinearX::QLinear(QLinear::from_linear_x(
                ln,
                cfg.quant_method.clone(),
                &quant_config_local,
            )));
        }
    }

    if let Some(quantized_type) = &quant_local {
        let ln = qlinear(in_dim, out_dim, vb, shard, &quant_config_local, true, dtype)?;
        Ok(LinearX::QLinear(QLinear::from_linear_x(
            ln,
            quantized_type.clone(),
            &quant_config_local,
        )))
    } else {
        let ln = linear(in_dim, out_dim, vb, shard)?;
        Ok(LinearX::Linear(ln))
    }
}

pub fn linear_no_bias_x(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    shards: Shard,
    quant: &Option<String>,
    quant_config: &Option<QuantConfig>,
    dtype: DType,
    merged_chunks: Option<(usize, usize)>, //(chunk_idx, num_of_chunks)
) -> Result<LinearX> {
    let bypass_quant = should_bypass_quant_for_module(&vb, quant_config);
    let quant_config_local = if bypass_quant {
        None
    } else {
        quant_config.clone()
    };
    let quant_local = if bypass_quant { None } else { quant.clone() };

    // Check for FP8 quantization first
    if let Some(cfg) = &quant_config_local {
        if cfg.quant_method == "fp8" {
            let has_fp8_scale =
                vb.contains_tensor("weight_scale") || vb.contains_tensor("weight_scale_inv");
            if !has_fp8_scale {
                let weight_probe = if let Some((chunk_idx, chunks)) = merged_chunks {
                    vb.get_with_hints(
                        (out_dim, in_dim),
                        "weight",
                        shard(
                            shards.dim,
                            chunk_idx * shards.world_size + shards.rank,
                            chunks * shards.world_size,
                        ),
                    )?
                } else {
                    vb.get_with_hints((out_dim, in_dim), "weight", shards)?
                };
                if matches!(
                    weight_probe.dtype(),
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64
                ) {
                    let ws = if let Some((chunk_idx, chunks)) = merged_chunks {
                        vb.get_with_hints(
                            (out_dim, in_dim),
                            "weight",
                            shard(
                                shards.dim,
                                chunk_idx * shards.world_size + shards.rank,
                                chunks * shards.world_size,
                            ),
                        )?
                    } else {
                        vb.get_with_hints((out_dim, in_dim), "weight", shards)?
                    };
                    return Ok(LinearX::Linear(Linear::new(ws, None)));
                }
            }
            let ln = load_ln_fp8_with_hints(in_dim, out_dim, vb, shards, cfg, false)?;
            return Ok(LinearX::LnFp8(ln));
        }
        if matches!(cfg.quant_method.as_str(), "gptq" | "awq" | "marlin") {
            let ln = qlinear(
                in_dim,
                out_dim,
                vb,
                shard(
                    if shards.world_size < 2 || shards.dim == 1 {
                        0
                    } else {
                        1
                    },
                    shards.rank,
                    shards.world_size,
                ),
                &quant_config_local,
                false,
                dtype,
            )?;
            return Ok(LinearX::QLinear(QLinear::from_linear_x(
                ln,
                cfg.quant_method.clone(),
                &quant_config_local,
            )));
        }
    }

    if quant_local.is_some()
        && matches!(
            quant_local.as_ref().unwrap().as_str(),
            "gptq" | "awq" | "marlin"
        )
    {
        let quantized_type = quant_local.as_ref().unwrap().clone();
        //quantized weight in k x n (shift dim in original shards)
        let ln = qlinear(
            in_dim,
            out_dim,
            vb,
            shard(
                if shards.world_size < 2 || shards.dim == 1 {
                    0
                } else {
                    1
                },
                shards.rank,
                shards.world_size,
            ),
            &quant_config_local,
            false,
            dtype,
        )
        .unwrap();
        Ok(LinearX::QLinear(QLinear::from_linear_x(
            ln,
            quantized_type.clone(),
            &quant_config_local,
        )))
    } else {
        //weight in n x k (use original shards)
        let ws = if merged_chunks.is_some() {
            let (chunk_idx, chunks) = merged_chunks.unwrap();
            vb.get_with_hints(
                (out_dim, in_dim),
                "weight",
                shard(
                    shards.dim,
                    chunk_idx * shards.world_size + shards.rank,
                    chunks * shards.world_size,
                ),
            )?
        } else {
            vb.get_with_hints((out_dim, in_dim), "weight", shards)?
        };

        let ln = Linear::new(ws, None);
        if quant_local.is_some() {
            let quantized_type = quant_local.as_ref().unwrap().clone();
            Ok(LinearX::QLinear(QLinear::from_linear_x(
                ln,
                quantized_type,
                &quant_config_local,
            )))
        } else {
            Ok(LinearX::Linear(ln))
        }
    }
}

pub fn linear_b_x(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    shard: Shard,
    quant: &Option<String>,
    quant_config: &Option<QuantConfig>,
    dtype: DType,
    merged_chunks: Option<(usize, usize)>,
) -> Result<LinearX> {
    if bias {
        linear_x(in_dim, out_dim, vb, shard, quant, quant_config, dtype)
    } else {
        linear_no_bias_x(
            in_dim,
            out_dim,
            vb,
            shard,
            quant,
            quant_config,
            dtype,
            merged_chunks,
        )
    }
}
