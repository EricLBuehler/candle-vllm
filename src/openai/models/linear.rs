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
use crate::candle::Module;
use crate::candle::{
    quantized::{gguf_file, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use candle_core::quantized;
use candle_nn::init;
use either::Either;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
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

/// Create or initialize a new linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vb: candle_nn::VarBuilder) -> Result<Linear> {
    let init_ws = init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = init::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

/// Create or initialize a new linear layer without biases.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: candle_nn::VarBuilder) -> Result<Linear> {
    let init_ws = init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: candle_nn::VarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug, Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
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
            dtype: DType::F32,
        })
    }

    pub fn from_linear(linear: Linear) -> Self {
        Self {
            inner: QMatMul::Tensor(linear.weight().clone()),
            bias: linear.bias().cloned(),
            dtype: linear.weight().dtype(),
        }
    }

    pub fn from_parts(w: Tensor, b: Option<Tensor>) -> Self {
        let dtype = w.dtype();
        Self {
            inner: QMatMul::Tensor(w),
            bias: b,
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
            dtype: dtype,
        }
    }

    pub fn from_linear_x(linear: Linear, quant: String) -> Self {
        let weight = linear.weight();
        let dtype = weight.dtype();
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
        let qbias = match linear.bias() {
            Some(b) => Some(b.clone()),
            _ => None,
        };

        QLinear::from_qparts_x(qtensor, qbias, dtype)
    }

    pub fn from_old_and_qmatmul(inner: QMatMul, old: &Self) -> Self {
        Self {
            inner,
            bias: old.bias.clone(),
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

    pub fn forward_via_f16(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let w = self.inner.dequantize_f16()?;
        let w = match *x.dims() {
            [b1, seq_len, _, _] => {
                if seq_len > 1 {
                    w.broadcast_left((b1, seq_len))?.t()?
                } else {
                    w.t()?
                }
            }
            [bsize, seq_len, _] => {
                if seq_len > 1 {
                    w.broadcast_left(bsize)?.t()?
                } else {
                    w.t()?
                }
            }
            _ => w.t()?,
        };
        let x = x.to_dtype(DType::F16)?;
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

        if let Some(bias) = &self.bias {
            x.broadcast_add(bias)?.to_dtype(in_dtype)
        } else {
            x.to_dtype(in_dtype)
        }
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dims()[0];
        if batch > 4 {
            self.forward_via_f16(x) //suitable for batched
        } else {
            self.forward_no_dequant(x) //faster in single-query
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
    pub fn new(weight: Tensor, bias: Option<Tensor>, quant: &Option<String>) -> Self {
        let ln = Linear::new(weight, bias);
        if let Some(quatized_type) = quant {
            LinearX(Either::Right(QLinear::from_linear_x(
                ln,
                quatized_type.clone(),
            )))
        } else {
            LinearX(Either::Left(ln))
        }
    }
}

pub fn linear_x(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
    quant: &Option<String>,
) -> Result<LinearX> {
    let ln = linear(in_dim, out_dim, vb).unwrap();
    if let Some(quatized_type) = quant {
        Ok(LinearX(Either::Right(QLinear::from_linear_x(
            ln,
            quatized_type.clone(),
        ))))
    } else {
        Ok(LinearX(Either::Left(ln)))
    }
}

pub fn linear_no_bias_x(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
    quant: &Option<String>,
) -> Result<LinearX> {
    let init_ws = init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let ln = Linear::new(ws, None);
    if let Some(quatized_type) = quant {
        Ok(LinearX(Either::Right(QLinear::from_linear_x(
            ln,
            quatized_type.clone(),
        ))))
    } else {
        Ok(LinearX(Either::Left(ln)))
    }
}

pub fn linear_b_x(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: candle_nn::VarBuilder,
    quant: &Option<String>,
) -> Result<LinearX> {
    if bias {
        linear_x(in_dim, out_dim, vb, quant)
    } else {
        linear_no_bias_x(in_dim, out_dim, vb, quant)
    }
}
