pub mod attention;
pub mod deepstack;
pub mod deltanet;
pub mod mask;
pub mod mla_attention;
pub mod mlp;
pub mod moe;
pub mod others;
pub mod qrmsnorm;
pub mod quantized_var_builder;
pub mod rotary_emb;

use candle_core::quantized::GgmlDType;

pub use crate::openai::distributed::VarBuilder as VarBuilderX;

pub fn isq_high_precision_quant(quant: &str) -> &'static str {
    match quant {
        "q8" | "q80" | "q8_0" => "q8_0",
        _ => "q6k",
    }
}

pub fn isq_high_precision_dtype(quant: Option<&str>) -> GgmlDType {
    match quant {
        Some("q8" | "q80" | "q8_0") => GgmlDType::Q8_0,
        _ => GgmlDType::Q6K,
    }
}
