use crate::candle::D;
use crate::candle::{DType, Error, Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};
use std::sync::Arc;
use std::sync::Mutex;
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: Arc<Mutex<rand::rngs::StdRng>>,
    pub sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng: Arc::new(Mutex::new(rng)),
            sampling,
        }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&self, logits: Tensor) -> Result<u32> {
        let next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        Ok(next_token)
    }

    fn sample_argmax_batch(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let next_tokens = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        Ok(next_tokens)
    }

    fn sample_multinomial(&self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let mut rng = self.rng.lock().unwrap();
        let next_token = distr.sample(&mut *rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&self, prs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&self, prs: &mut Vec<f32>, top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f_batch(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let logits = logits.to_dtype(DType::F32)?;
        let tokens = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax_batch(&logits),
            Sampling::All { temperature }
            | Sampling::TopP { temperature, .. }
            | Sampling::TopK { temperature, .. }
            | Sampling::TopKThenTopP { temperature, .. } => {
                let temper = if *temperature > 0. { *temperature } else { 1.0 };
                let logits = (&logits / temper)?;
                let prs = candle_nn::ops::softmax_last_dim(&logits)?;
                let prs_vec = prs.to_vec2()?;
                let mut vec_ret = Vec::<u32>::new();
                for idx in 0..prs_vec.len() {
                    let next_token = match &self.sampling {
                        Sampling::All { .. } => {
                            let prs = prs_vec[idx].clone();
                            self.sample_multinomial(&prs)?
                        }
                        Sampling::TopP { p, .. } => {
                            let mut prs = prs_vec[idx].clone();
                            if *p <= 0.0 || *p >= 1.0 {
                                // simply sample from the predicted probability distribution
                                self.sample_multinomial(&prs)?
                            } else {
                                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                                self.sample_topp(&mut prs, *p as f32)?
                            }
                        }
                        Sampling::TopK { k, .. } => {
                            let mut prs = prs_vec[idx].clone();
                            self.sample_topk(&mut prs, *k)?
                        }
                        Sampling::TopKThenTopP { k, p, .. } => {
                            let mut prs = prs_vec[idx].clone();
                            self.sample_topk_topp(&mut prs, *k, *p as f32)?
                        }
                        _ => {
                            unreachable!();
                        }
                    };
                    vec_ret.push(next_token);
                }

                Ok(vec_ret)
            }
        };
        tokens
    }

    pub fn sample_batch(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let next_tokens = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax_batch(logits)?,
            _ => self.sample_f_batch(logits)?,
        };
        Ok(next_tokens)
    }

    pub fn sample_f(&self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            let mut prs = prs.to_vec1()?;
            f(&mut prs);
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}
