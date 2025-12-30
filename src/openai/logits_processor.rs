use crate::candle::D;
use crate::candle::{DType, Error, Result, Tensor};
use crate::openai::sampling_params::SamplingParams;
#[cfg(feature = "cuda")]
use attention_rs::sort::ArgSortOp; //Use our custom sort kernel, fix kernel crash on A100
use rand::{distr::Distribution, SeedableRng};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::sync::Arc;
use std::sync::Mutex;
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All {
        temperature: f32,
    },
    TopK {
        k: usize,
        temperature: f32,
    },
    TopP {
        p: f32,
        minp: f32,
        temperature: f32,
    },
    TopKThenTopP {
        k: usize,
        p: f32,
        minp: f32,
        temperature: f32,
    },
}

pub struct LogitsProcessor {
    rng: Arc<Mutex<rand::rngs::StdRng>>,
    pub sampling: Sampling,
    #[cfg(feature = "cuda")]
    fast_sampler: Arc<std::sync::Mutex<attention_rs::sampler::Sampler>>,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng: Arc::new(Mutex::new(rng)),
            sampling,
            #[cfg(feature = "cuda")]
            fast_sampler: Arc::new(std::sync::Mutex::new(attention_rs::sampler::Sampler::new())),
        }
    }

    pub fn new(
        seed: u64,
        temperature: Option<f32>,
        top_k: Option<isize>,
        top_p: Option<f32>,
        min_p: Option<f32>,
    ) -> Self {
        let strategy = LogitsProcessor::get_strategy(temperature, top_k, top_p, min_p);
        Self::from_sampling(seed, strategy)
    }

    pub fn get_strategy(
        temperature: Option<f32>,
        top_k: Option<isize>,
        top_p: Option<f32>,
        min_p: Option<f32>,
    ) -> Sampling {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let top_k: Option<usize> = top_k.filter(|&k| k > 0).map(|k| k as usize);

        let temperature: Option<f32> = temperature.filter(|&t| t > 0.0);

        match (temperature, top_k, top_p) {
            (None, _, _) => Sampling::ArgMax,
            (Some(temperature), None, None) => Sampling::All { temperature },
            (Some(temperature), Some(k), None) => Sampling::TopK { k, temperature },
            (Some(temperature), None, Some(p)) => Sampling::TopP {
                p,
                minp: min_p.unwrap_or(0.),
                temperature,
            },
            (Some(temperature), Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                minp: min_p.unwrap_or(0.),
                temperature,
            },
        }
    }

    fn sample_argmax(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let next_tokens = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        Ok(next_tokens)
    }

    fn sample_multinomial(&self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let mut rng = self.rng.lock().unwrap();
        let next_token = distr.sample(&mut *rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&self, logits: &Tensor, top_p: f32, min_p: Option<f32>) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let asort = logits.arg_sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let asort = logits
            .to_device(&candle_core::Device::Cpu)?
            .arg_sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = logits.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let min_p = min_p.unwrap_or(0.);
        let vec_ret: Vec<u32> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let indices: Vec<u32> = asort[b].to_vec();
                let mut prs: Vec<f32> = sorted[b].to_vec();
                let min_threshold = if min_p > 0. {
                    let max_prob = prs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    min_p * max_prob
                } else {
                    0.
                };
                let mut error_prompt = false;
                // Clamp smaller probabilities to zero.
                let mut cumsum = 0.;
                for &index in &indices {
                    if let Some(p) = prs.get_mut(index as usize) {
                        if cumsum > top_p || *p < min_threshold {
                            *p = 0.0;
                        } else {
                            cumsum += *p;
                        }
                    } else {
                        if !error_prompt {
                            error_prompt = true;
                            tracing::error!("Index out of bound during sampling, garbage model output detected!")
                        }
                    }
                }
                // Sample with clamped probabilities.
                self.sample_multinomial(&prs).unwrap()
            })
            .collect();
        Ok(vec_ret)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&self, logits: &Tensor, top_k: usize) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let (sorted, asort) = logits
            .to_device(&candle_core::Device::Cpu)?
            .sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let vec_ret: Vec<u32> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let indices: Vec<u32> = asort[b][0..top_k].to_vec();
                let prs: Vec<f32> = sorted[b][0..top_k].to_vec();
                let index = self.sample_multinomial(&prs).unwrap();
                indices[index as usize] as u32
            })
            .collect();
        Ok(vec_ret)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(
        &self,
        logits: &Tensor,
        top_k: usize,
        top_p: f32,
        min_p: Option<f32>,
    ) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        let (sorted, asort) = logits.sort(false)?;
        #[cfg(not(feature = "cuda"))]
        let (sorted, asort) = logits
            .to_device(&candle_core::Device::Cpu)?
            .sort_last_dim(false)?;
        let asort: Vec<Vec<u32>> = asort.to_vec2()?;
        let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;
        let batch = logits.layout().dims()[0];
        let min_p = min_p.unwrap_or(0.);
        let vec_ret: Vec<u32> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let indices: Vec<u32> = asort[b][0..top_k].to_vec();
                let mut prs: Vec<f32> = sorted[b][0..top_k].to_vec();
                let sum_p = prs.iter().sum::<f32>();
                let index = if top_p <= 0.0 || top_p >= sum_p {
                    self.sample_multinomial(&prs).unwrap()
                } else {
                    let min_threshold = if min_p > 0. {
                        let max_prob = prs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        min_p * max_prob
                    } else {
                        0.
                    };
                    let mut cumsum = 0.;
                    for i in 0..prs.len() {
                        if cumsum >= top_p || prs[i] < min_threshold {
                            prs[i] = 0.0;
                        } else {
                            cumsum += prs[i];
                        }
                    }
                    // Sample with clamped probabilities.
                    self.sample_multinomial(&prs).unwrap()
                };
                indices[index as usize] as u32
            })
            .collect();
        Ok(vec_ret)
    }

    pub fn sample(
        &self,
        logits: &Tensor,
        sampling_params: &Option<SamplingParams>,
    ) -> Result<Vec<u32>> {
        let sampling = sampling_params.as_ref().map_or_else(
            || self.sampling.to_owned(),
            |param| {
                LogitsProcessor::get_strategy(
                    param.temperature,
                    param.top_k,
                    param.top_p,
                    param.min_p,
                )
            },
        );

        // CUDA fast path for supported sampling strategies
        #[cfg(feature = "cuda")]
        {
            // Extract k, p, temperature based on sampling strategy
            // TopK: use p=1.0 to disable top-p filtering
            // TopP: use k=128 (kernel max) so top-p can consider enough candidates
            let (k, p, t) = match &sampling {
                Sampling::TopKThenTopP {
                    k, p, temperature, ..
                } => (*k, *p, *temperature),
                Sampling::TopK { k, temperature } => (*k, 1.0, *temperature),
                Sampling::TopP { p, temperature, .. } => (128, *p, *temperature),
                _ => (0, 0.0, 0.0), // Marker for unsupported strategies
            };

            let should_run = matches!(
                sampling,
                Sampling::TopKThenTopP { .. } | Sampling::TopK { .. } | Sampling::TopP { .. }
            );

            if should_run && k > 0 {
                let seed = {
                    use rand::RngCore;
                    self.rng.lock().unwrap().next_u64()
                };
                let sampler = self.fast_sampler.lock().unwrap();
                return sampler.sample_cuda(logits, k, p, t, seed);
            }
        }

        // CPU fallback
        let logits = logits.to_dtype(DType::F32)?;
        let batch = logits.layout().dims()[0];
        let prs = |temperature: f64| -> Result<Tensor> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            Ok(prs)
        };

        let next_tokens = match &sampling {
            Sampling::ArgMax => self.sample_argmax(&logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature as f64)?.to_vec2()?;
                (0..batch)
                    .map(|b| self.sample_multinomial(&prs[b]).unwrap())
                    .collect()
            }
            Sampling::TopP {
                p,
                minp,
                temperature,
            } => {
                let prs = prs(*temperature as f64)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    let prs = prs.to_vec2()?;
                    (0..batch)
                        .map(|b| self.sample_multinomial(&prs[b]).unwrap())
                        .collect()
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&prs, *p, Some(*minp))?
                }
            }
            Sampling::TopK { k, temperature } => {
                let prs = prs(*temperature as f64)?;
                self.sample_topk(&prs, *k)?
            }
            Sampling::TopKThenTopP {
                k,
                p,
                minp,
                temperature,
            } => {
                let prs = prs(*temperature as f64)?;
                self.sample_topk_topp(&prs, *k, *p, Some(*minp))?
            }
        };
        Ok(next_tokens)
    }

    fn apply_penalties(
        &self,
        logits: &mut [f32],
        context: &[u32],
        frequency_penalty: f32,
        presence_penalty: f32,
    ) {
        let mut counts = vec![0.0f32; logits.len()];
        for ctx in context.iter() {
            if *ctx as usize >= logits.len() {
                continue;
            }
            counts[*ctx as usize] += 1.0;
        }
        for (token_id, logit) in logits.iter_mut().enumerate() {
            let count = counts[token_id];
            *logit = *logit
                - count * frequency_penalty
                - if count > 0.0 { 1. } else { 0. } * presence_penalty;
        }
    }

    pub fn apply_batch_repeat_penalty(
        &self,
        logits: &Tensor,
        frequency_penalties: Vec<f32>,
        presence_penalties: Vec<f32>,
        context: Vec<Vec<u32>>,
    ) -> Result<Tensor> {
        let device = logits.device();
        let batch = logits.layout().dims()[0];
        let logits_len = logits.layout().dims()[1];
        let logits: Vec<Vec<f32>> = if logits.dtype() != candle_core::DType::F32 {
            logits.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?
        } else {
            logits.to_vec2::<f32>()?
        };
        let vec_ret: Vec<Vec<f32>> = (0..batch)
            .into_par_iter()
            .map(|b| {
                let mut logits = logits[b].to_vec();
                if context[b].len() > 1
                    && ((frequency_penalties[b] != 1.0 && frequency_penalties[b] != 0.)
                        || (presence_penalties[b] != 1.0 && presence_penalties[b] != 0.))
                {
                    self.apply_penalties(
                        &mut logits,
                        &context[b],
                        frequency_penalties[b],
                        presence_penalties[b],
                    );
                }
                logits
            })
            .collect();

        let logits = vec_ret.into_iter().flatten().collect();
        Tensor::from_vec(logits, (batch, logits_len), device)
    }
}
