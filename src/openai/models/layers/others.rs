use crate::openai::distributed::VarBuilder;
use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Module, RmsNorm};

pub enum NormX {
    Rms(RmsNorm, candle_core::DType),
    Layer(LayerNorm, candle_core::DType),
}

impl NormX {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm_dtype = match self {
            Self::Rms(_, dt) | Self::Layer(_, dt) => *dt,
        };
        let in_dtype = xs.dtype();
        let xs = if in_dtype != norm_dtype {
            xs.to_dtype(norm_dtype)?
        } else {
            xs.clone()
        };
        let out = match self {
            Self::Rms(norm, _) => norm.forward(&xs)?,
            Self::Layer(norm, _) => norm.forward(&xs)?,
        };
        if out.dtype() != in_dtype {
            out.to_dtype(in_dtype)
        } else {
            Ok(out)
        }
    }
}

pub fn rms_norm(
    size: usize,
    eps: f64,
    vb: VarBuilder,
    dtype: candle_core::DType,
    is_gemma: bool,
) -> Result<NormX> {
    let weight = vb.get(size, "weight")?;
    let weight = if weight.dtype() != dtype {
        weight.to_dtype(dtype)?
    } else {
        weight
    };
    let weight = if is_gemma { (weight + 1.0)? } else { weight };
    let dt = weight.dtype();
    Ok(NormX::Rms(RmsNorm::new(weight, eps), dt))
}

pub fn layer_norm(
    size: usize,
    eps: f64,
    affine: bool,
    vb: VarBuilder,
    _dtype: candle_core::DType,
) -> Result<NormX> {
    let weight = vb.get(size, "weight")?;
    let dt = weight.dtype();
    if affine {
        let bias = vb.get(size, "bias")?;
        Ok(NormX::Layer(LayerNorm::new(weight, bias, eps), dt))
    } else {
        Ok(NormX::Layer(LayerNorm::new_no_bias(weight, eps), dt))
    }
}

pub fn embedding(
    vocab_size: Option<usize>,
    hidden_size: usize,
    vb: VarBuilder,
    _dtype: candle_core::DType,
) -> Result<(Embedding, usize)> {
    let vocab_size = vocab_size.expect("embedding requires an explicit vocab size");
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok((Embedding::new(embeddings, hidden_size), vocab_size))
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vb: VarBuilder,
    bias: bool,
) -> Result<candle_nn::Conv2d> {
    let ws = vb.get(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    let bs = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };

    Ok(candle_nn::Conv2d::new(ws, bs, cfg))
}

pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.avg_pool2d_with_stride(self.kernel_size, self.stride)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

pub struct Conv3dNoBias {
    conv2d_1: candle_nn::Conv2d,
    conv2d_2: candle_nn::Conv2d,
}

impl Conv3dNoBias {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        use candle_nn::Conv2dConfig;

        let ws = vb.get(
            (
                out_channels,
                in_channels / cfg.groups,
                kernel_sizes[0],
                kernel_sizes[1],
                kernel_sizes[2],
            ),
            "weight",
        )?;

        let w1 = ws.i((.., .., 0, .., ..))?;
        let w2 = ws.i((.., .., 1, .., ..))?;

        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
        };

        Ok(Self {
            conv2d_1: candle_nn::Conv2d::new(w1.contiguous()?, None, cfg),
            conv2d_2: candle_nn::Conv2d::new(w2.contiguous()?, None, cfg),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x1 = xs.i((.., .., 0, .., ..))?;
        let x2 = xs.i((.., .., 1, .., ..))?;
        let y1 = self.conv2d_1.forward(&x1)?;
        let y2 = self.conv2d_2.forward(&x2)?;
        y1.broadcast_add(&y2)
    }
}
