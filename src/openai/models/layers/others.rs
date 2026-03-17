use crate::openai::distributed::VarBuilder;
use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Module, RmsNorm};

pub enum NormX {
    Rms(RmsNorm),
    Layer(LayerNorm),
}

impl NormX {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(norm) => norm.forward(xs),
            Self::Layer(norm) => norm.forward(xs),
        }
    }
}

pub fn rms_norm(
    size: usize,
    eps: f64,
    vb: VarBuilder,
    _dtype: candle_core::DType,
    is_gemma: bool,
) -> Result<NormX> {
    let weight = vb.get(size, "weight")?;
    let weight = if is_gemma { (weight + 1.0)? } else { weight };
    Ok(NormX::Rms(RmsNorm::new(weight, eps)))
}

pub fn layer_norm(
    size: usize,
    eps: f64,
    affine: bool,
    vb: VarBuilder,
    _dtype: candle_core::DType,
) -> Result<NormX> {
    let weight = vb.get(size, "weight")?;
    if affine {
        let bias = vb.get(size, "bias")?;
        Ok(NormX::Layer(LayerNorm::new(weight, bias, eps)))
    } else {
        Ok(NormX::Layer(LayerNorm::new_no_bias(weight, eps)))
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
