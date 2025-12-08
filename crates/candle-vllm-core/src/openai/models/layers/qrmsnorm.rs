use super::quantized_var_builder::VarBuilder;
use candle_core::quantized::QTensor;
use candle_core::{Module, Result, Tensor};
#[derive(Debug, Clone)]
pub struct QRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
        Ok(Self { weight, eps })
    }

    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for QRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}
