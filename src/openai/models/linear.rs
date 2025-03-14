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
use crate::backend::gptq::{gptq_matmul, gptq_weight_repack};
use crate::candle::Module;
use crate::candle::{
    quantized::{gguf_file, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use crate::openai::distributed::shard;
use candle_core::quantized;
pub use candle_nn::var_builder::Shard;
pub use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use either::Either;
use std::sync::Arc;

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
            let marlin_compatible =
                if cfg.quant_method != "gptq" || (cfg.bits != 4 && cfg.bits != 8) {
                    false
                } else {
                    true
                };
            let marlin_format = if cfg.checkpoint_format.is_some()
                && cfg.checkpoint_format.as_ref().unwrap() == "marlin"
            {
                true
            } else {
                false
            };
            let ws = vb.get_with_hints_dtype(
                (
                    in_dim / (32 / cfg.bits) / if marlin_format { 2 } else { 1 },
                    out_dim * if marlin_format { 2 } else { 1 },
                ),
                if marlin_format { "B" } else { "qweight" },
                Default::default(),
                DType::U32,
            )?;

            let ws = if shards.world_size > 1 {
                let dim_size = ws.dims()[shards.dim];
                let start = shards.rank * (dim_size / shards.world_size);
                ws.narrow(shards.dim, start, dim_size / shards.world_size)?
                    .contiguous()?
            } else {
                ws
            };

            let scale_and_zero_size = in_dim / (cfg.group_size as usize);
            let scales = vb.get_with_hints_dtype(
                (scale_and_zero_size, out_dim),
                if marlin_format { "s" } else { "scales" },
                shards,
                DType::F16,
            )?;

            let scales = if dtype != scales.dtype() {
                scales.to_dtype(dtype)?
            } else {
                scales
            };

            let in_dim_partition = if shards.world_size > 1 && shards.dim == 0 {
                in_dim / shards.world_size
            } else {
                in_dim
            };

            let scale_and_zero_size_partition = if shards.world_size > 1 && shards.dim == 0 {
                scale_and_zero_size / shards.world_size
            } else {
                scale_and_zero_size
            };

            let out_dim_partition = if shards.world_size > 1 && shards.dim == 1 {
                out_dim / shards.world_size
            } else {
                out_dim
            };

            let bs = if bias {
                let mut bs = vb
                    .get_with_hints_dtype((out_dim,), "bias", Default::default(), DType::F16)?
                    .to_dtype(dtype)?;
                bs = if out_dim_partition < out_dim {
                    bs.narrow(0, shards.rank * out_dim_partition, out_dim_partition)?
                        .contiguous()?
                } else {
                    bs
                };
                Some(bs)
            } else {
                None
            };

            if marlin_format {
                let workspace =
                    Tensor::zeros(out_dim_partition / (32 / cfg.bits), DType::U32, vb.device())?;
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
                let mut qzeros = vb.get_with_hints_dtype(
                    (scale_and_zero_size, out_dim / (32 / cfg.bits)),
                    "qzeros",
                    Default::default(),
                    DType::U32,
                )?;
                qzeros = if scale_and_zero_size_partition < scale_and_zero_size
                    || out_dim_partition < out_dim
                {
                    let dim_size = qzeros.dims()[shards.dim];
                    let start = shards.rank * (dim_size / shards.world_size);
                    qzeros
                        .narrow(shards.dim, start, dim_size / shards.world_size)?
                        .contiguous()?
                } else {
                    qzeros
                };
                let mut g_idx =
                    vb.get_with_hints_dtype((in_dim,), "g_idx", Default::default(), DType::U32)?;
                g_idx = if in_dim_partition < in_dim {
                    g_idx
                        .narrow(0, shards.rank * in_dim_partition, in_dim_partition)?
                        .contiguous()?
                } else {
                    g_idx
                };

                if !cfg.sym
                    || cfg.bits != 4
                    || (cfg.group_size != 128 && cfg.group_size != -1)
                    || (cfg.desc_act.is_some() && cfg.desc_act.unwrap() == true)
                {
                    //only model with 4-bit and desc_act==false can be repacked to marlin format
                    if cfg.quant_method == "marlin" {
                        println!("The current GPTQ model does no compatible with marlin format because one of the following conditions: !cfg.sym || cfg.bits != 4 || (cfg.group_size != 128 && cfg.group_size != -1) || (cfg.desc_act == true)");
                    }
                    //conventional gptq format
                    Ok(Linear {
                        weight: ws,
                        bias: bs,
                        scales: Some(scales),
                        qzeros: Some(qzeros),
                        g_idx: Some(g_idx),
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
                        gptq_weight_repack(&ws)?
                    } else {
                        ws
                    }; //repack to marlin format

                    let scales = if marlin_compatible {
                        marlin_permute_scales(
                            &scales,
                            in_dim_partition as usize,
                            out_dim_partition as usize,
                            cfg.group_size,
                            cfg.bits as u32,
                        )?
                    } else {
                        scales
                    };

                    let workspace = Tensor::zeros(
                        out_dim_partition / (32 / cfg.bits),
                        DType::U32,
                        vb.device(),
                    )?;
                    Ok(Linear {
                        weight: ws,
                        bias: bs,
                        scales: Some(scales),
                        qzeros: Some(qzeros),
                        g_idx: Some(g_idx),
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
        })
    }

    pub fn from_linear(linear: Linear, group_size: i32, bits: i32) -> Self {
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
        }
    }

    pub fn from_qparts_x(w: QTensor, b: Option<Tensor>, dtype: DType) -> Self {
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
        }
    }

    pub fn from_linear_x(
        linear: Linear,
        quant: String,
        quant_config: &Option<QuantConfig>,
    ) -> Self {
        let weight = linear.weight();
        let qbias = linear.bias().cloned();
        let dtype = weight.dtype();
        match quant_config {
            Some(cfg) => {
                assert!(
                    cfg.quant_method == "gptq" || cfg.quant_method == "marlin" || quant == "marlin"
                );
                QLinear::from_linear(linear, cfg.group_size as i32, cfg.bits as i32)
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
                let qtensor = QTensor::quantize(weight, ggml_dtype).unwrap();
                QLinear::from_qparts_x(qtensor, qbias, dtype)
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

    pub fn forward_via_dequant(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        // let in_dtype = x.dtype();
        // let w = self.inner.dequantize_f16()?;
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
                        )?;
                        o.reshape((bsize, seq_len, dim1, ()))?
                    }
                    [_, _, _] => gptq_matmul(
                        &x,
                        qw,
                        scale,
                        qzeros,
                        g_idx,
                        workspace,
                        self.bits,
                        self.group_size,
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
            _ => self.forward_no_dequant(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearX(Either<Linear, QLinear>);

impl Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.0 {
            Either::Left(ln) => ln.forward(x),
            Either::Right(ln) => ln.forward(x),
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
        if let Some(quatized_type) = quant {
            LinearX(Either::Right(QLinear::from_linear_x(
                ln,
                quatized_type.clone(),
                quant_config,
            )))
        } else {
            LinearX(Either::Left(ln))
        }
    }
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
    if let Some(quatized_type) = quant {
        let ln = qlinear(in_dim, out_dim, vb, shard, quant_config, true, dtype).unwrap();
        Ok(LinearX(Either::Right(QLinear::from_linear_x(
            ln,
            quatized_type.clone(),
            quant_config,
        ))))
    } else {
        let ln = linear(in_dim, out_dim, vb, shard).unwrap();
        Ok(LinearX(Either::Left(ln)))
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
) -> Result<LinearX> {
    if let Some(quatized_type) = quant {
        //quantized weight in k x n (shift dim in original shards)
        let ln = qlinear(
            in_dim,
            out_dim,
            vb,
            shard(
                if shards.world_size < 2 {
                    0
                } else {
                    if shards.dim == 1 {
                        0
                    } else {
                        1
                    }
                },
                shards.rank,
                shards.world_size,
            ),
            quant_config,
            false,
            dtype,
        )
        .unwrap();
        Ok(LinearX(Either::Right(QLinear::from_linear_x(
            ln,
            quatized_type.clone(),
            quant_config,
        ))))
    } else {
        //weight in n x k (use original shards)
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", shards)?;
        let ln = Linear::new(ws, None);
        Ok(LinearX(Either::Left(ln)))
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
) -> Result<LinearX> {
    if bias {
        linear_x(in_dim, out_dim, vb, shard, quant, quant_config, dtype)
    } else {
        linear_no_bias_x(in_dim, out_dim, vb, shard, quant, quant_config, dtype)
    }
}
