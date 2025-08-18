use super::{Config, RopeScaling, ScalingValue};
use candle::{DType, Device, Result, Tensor};
use candle_core as candle;
use either::Either;
use std::iter::zip;
pub use std::rc::Rc;
#[derive(Debug, Clone)]
pub struct DefaultRotaryEmbedding {
    pub cos: Tensor,
    pub sin: Tensor,
    pub is_gpt_neox: bool,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let dim = cfg
        .head_dim
        .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
    (0..dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
        .collect()
}

impl DefaultRotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, device: &Device, is_gpt_neox: bool) -> Result<Self> {
        let theta: Vec<_> = calculate_default_inv_freq(cfg);
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, cfg.max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            cos,
            sin,
            is_gpt_neox,
        })
    }
}

impl DefaultRotaryEmbedding {
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &[Vec<usize>],
    ) -> Result<(Tensor, Tensor)> {
        let (b_size, _h, seq_len, _n_embd) = q.dims4()?;
        let mut q_embeds = Vec::new();
        let mut k_embeds = Vec::new();
        let rope_func = if self.is_gpt_neox {
            candle_nn::rotary_emb::rope
        } else {
            candle_nn::rotary_emb::rope_i
        };

        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let cos = self.cos.narrow(0, seqlen_offset[0], seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offset[0], seq_len)?;
            let x_q = q.narrow(0, b, 1)?;
            let x_k = k.narrow(0, b, 1)?;
            let q_embed = rope_func(&x_q, &cos, &sin)?;
            let k_embed = rope_func(&x_k, &cos, &sin)?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }
}
#[derive(Debug, Clone)]
pub struct Llama3RotaryEmbedding(DefaultRotaryEmbedding);

impl Llama3RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device, is_gpt_neox: bool) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(rope_scaling) => {
                match (
                    &rope_scaling["rope_type"],
                    &rope_scaling["factor"],
                    &rope_scaling["low_freq_factor"],
                    &rope_scaling["high_freq_factor"],
                    &rope_scaling["original_max_position_embeddings"],
                ) {
                    (
                        RopeScaling(Either::Right(rope_type)),
                        RopeScaling(Either::Left(ScalingValue(Either::Left(factor)))),
                        RopeScaling(Either::Left(ScalingValue(Either::Left(low_freq_factor)))),
                        RopeScaling(Either::Left(ScalingValue(Either::Left(high_freq_factor)))),
                        RopeScaling(Either::Left(ScalingValue(Either::Left(
                            original_max_position_embeddings,
                        )))),
                    ) => {
                        if rope_type == "llama3" {
                            let low_freq_wavelen =
                                (original_max_position_embeddings / low_freq_factor) as f32;
                            let high_freq_wavelen =
                                (original_max_position_embeddings / high_freq_factor) as f32;

                            let inv_freq = calculate_default_inv_freq(cfg)
                                .into_iter()
                                .map(|freq| {
                                    let wavelen = 2. * std::f32::consts::PI / freq;
                                    if wavelen < high_freq_wavelen {
                                        freq
                                    } else if wavelen > low_freq_wavelen {
                                        freq / *factor as f32
                                    } else {
                                        let smooth = (*original_max_position_embeddings as f32
                                            / wavelen
                                            - *low_freq_factor as f32)
                                            / (*high_freq_factor - *low_freq_factor) as f32;
                                        (1. - smooth) * freq / *factor as f32 + smooth * freq
                                    }
                                })
                                .collect::<Vec<_>>();
                            let inv_freq_len = inv_freq.len();
                            let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
                            let t = Tensor::arange(0u32, cfg.max_seq_len as u32, dev)?
                                .to_dtype(DType::F32)?
                                .reshape((cfg.max_seq_len, 1))?;
                            let freqs = t.matmul(&inv_freq)?;
                            let sin = freqs.sin()?.to_dtype(dtype)?;
                            let cos = freqs.cos()?.to_dtype(dtype)?;
                            return Ok(Self(DefaultRotaryEmbedding {
                                sin,
                                cos,
                                is_gpt_neox,
                            }));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        let default_rope = DefaultRotaryEmbedding::new(dtype, cfg, dev, is_gpt_neox)?;
        Ok(Self(DefaultRotaryEmbedding {
            sin: default_rope.sin,
            cos: default_rope.cos,
            is_gpt_neox,
        }))
    }

    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &[Vec<usize>],
    ) -> Result<(Tensor, Tensor)> {
        self.0.apply_rotary_emb(q, k, input_positions)
    }
}
