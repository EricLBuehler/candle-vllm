use super::{requests::StopTokens, responses::APIError};
use serde::{Deserialize, Serialize};
use std::ops::Range;

const SAMPLING_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Serialize, Deserialize)]
// Top-n logprobs element
pub struct TopLogprob {
    pub token: usize,
    pub logprob: f32,
    pub bytes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprobs {
    pub token: usize,
    pub logprob: f32,
    pub bytes: String,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EarlyStoppingCondition {
    ///True
    BestOfCompleteCandidates,
    ///False
    UnlikelyBetterCandidates,
    ///"never"
    CanonicalNoBetterCandidates,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SamplingType {
    BEAM,
    GREEDY,
    RANDOM,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Number of output seqs to return for a prompt.
    pub n: usize,
    /// Number of output seqs that are generated from the prompt, from these `best_of` seqs, the top `n` sequences are returned. `best_of` must be `>=n`. Default = `n`.
    /// Beam width when `use_beam_search` is true.
    pub best_of: usize,
    /// Penalize new tokens based upon whether they appear in the generated text so far, >0 encourage new, <0 encourage repeat.
    /// rec. default = 0
    pub presence_penalty: f32,
    /// Penalize new tokens based upon whether their frequency in the generated text so far, >0 encourage new, <0 encourage repeat.
    /// rec. default = 0
    pub frequency_penalty: f32,
    /// Penalize new tokens based upon whether their frequency in the generated text so far, >1 encourage new, <1 encourage repeat
    /// rec. default = 1
    pub repetition_penalty: f32,
    /// Randomness of sampling.
    /// rec. default = 1
    pub temperature: Option<f32>,
    /// Cumulative prob of the top tokens to consider, must be in (0, 1]. Set 1 to consider all toks.  
    /// rec. default = 1    
    pub top_p: Option<f32>,
    /// Control the number of top tokens to consider, set -1 to consider all.
    /// rec. default = -1
    pub top_k: Option<isize>,
    /// Use beam search instead of sampling.
    /// rec. default = false
    pub use_beam_search: bool,
    /// Penalize based on length.
    /// rec. default = 1
    pub length_penalty: f32,
    /// Control stopping for beam search.
    /// rec. default = EarlyStoppingCondition::UnlikelyBetterCandidates
    pub early_stopping: EarlyStoppingCondition,
    /// Strings that stop generation when generated.
    pub stop: Option<StopTokens>,
    /// Tokens to stop on.
    pub stop_token_ids: Vec<usize>,
    /// Whether to ignore EOS token.
    pub ignore_eos: bool,
    /// Max number of toks to gen per output seq.
    /// rec. default = 16
    pub max_tokens: usize,
    /// Num of log probs to return per output token. Follows OpenAI API, return result include the log probabilities on the `logprobs` most likely tokens.
    /// will always return the log prob of the sampled token, so there may be up to `logprobs+1` elements in the response.
    /// Default = 1
    pub logprobs: Option<usize>,
    /// Num of log probs to return per prompt token.
    pub prompt_logprobs: Option<usize>,
    /// Skip special toks in output.
    /// rec. default = true
    pub skip_special_tokens: bool,
    /// Thinking flag for reasoning models
    //  default = False
    pub thinking: Option<bool>,
}

impl SamplingParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        best_of: Option<usize>,
        presence_penalty: f32,
        frequency_penalty: f32,
        repetition_penalty: f32,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<isize>,
        use_beam_search: bool,
        length_penalty: f32,
        early_stopping: EarlyStoppingCondition,
        stop: Option<StopTokens>,
        stop_token_ids: Vec<usize>,
        ignore_eos: bool,
        max_tokens: usize,
        logprobs: Option<usize>,
        prompt_logprobs: Option<usize>,
        skip_special_tokens: bool,
        thinking: Option<bool>,
    ) -> Result<Self, APIError> {
        let this = Self {
            n,
            best_of: best_of.unwrap_or(n),
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            temperature,
            top_p,
            top_k,
            use_beam_search,
            length_penalty,
            early_stopping,
            stop,
            stop_token_ids,
            ignore_eos,
            max_tokens,
            logprobs,
            prompt_logprobs,
            skip_special_tokens,
            thinking,
        };

        this.verify_args()?;
        if this.use_beam_search {
            this.verify_beam_search()?;
        } else {
            this.verify_non_beam_search()?;
            if this.temperature.unwrap_or(0.0f32) < SAMPLING_EPS {
                this.verify_greedy_sampling()?;
            }
        }

        Ok(this)
    }

    // pub fn get_logits_processor<'a>(
    //     &self,
    //     seed: u64,
    //     tokenizer: &'a Tokenizer,
    //     top_n_logprobs: usize,
    // ) -> LogitsProcessor<'a> {
    //     if self.top_k == -1 && self.top_p == 1. {
    //         // Greedy
    //         LogitsProcessor::new(
    //             seed,
    //             Some(self.temperature.into()),
    //             SamplingMethod::Multinomial,
    //             top_n_logprobs,
    //             tokenizer,
    //         )
    //     } else if self.top_k > 0 && self.top_p == 1. {
    //         // Top-k
    //         LogitsProcessor::new(
    //             seed,
    //             Some(self.temperature.into()),
    //             SamplingMethod::TopK(self.top_k.try_into().unwrap()),
    //             top_n_logprobs,
    //             tokenizer,
    //         )
    //     } else if self.top_k == -1 && self.top_p != 1. {
    //         // Top-p
    //         LogitsProcessor::new(
    //             seed,
    //             Some(self.temperature.into()),
    //             SamplingMethod::TopP(self.top_p.into()),
    //             top_n_logprobs,
    //             tokenizer,
    //         )
    //     } else {
    //         unreachable!()
    //     }
    // }

    fn verify_args(&self) -> Result<(), APIError> {
        if self.n < 1 {
            return Err(APIError::new(format!(
                "n must be at leas 1, got {}.",
                self.n
            )));
        }
        if self.best_of < self.n {
            return Err(APIError::new(format!(
                "best_of must be greater than or equal to n, got n={} and best_of={}",
                self.n, self.best_of
            )));
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(APIError::new(format!(
                "presence_penalty must be in [-2, 2], got {}",
                self.presence_penalty
            )));
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(APIError::new(format!(
                "frequency_penalty must be in [-2, 2], got {}",
                self.frequency_penalty
            )));
        }
        if !(Range {
            start: 0.0,
            end: 2.0,
        })
        .contains(&self.repetition_penalty)
            || self.repetition_penalty == 0.0
        {
            return Err(APIError::new(format!(
                "repetition_penalty must be in (0, 2], got {}",
                self.repetition_penalty
            )));
        }
        if self.temperature.unwrap_or(0.0f32) < 0.0f32 {
            return Err(APIError::new(format!(
                "temperature must be non-negative, got {}",
                self.temperature.unwrap_or(0.0f32)
            )));
        }
        if self.max_tokens < 1 {
            return Err(APIError::new(format!(
                "max_tokens must be at least 1, got {}",
                self.max_tokens
            )));
        }
        Ok(())
    }

    fn verify_beam_search(&self) -> Result<(), APIError> {
        if self.best_of <= 1 {
            return Err(APIError::new(format!(
                "best_of must be greater than 1 when using beam search. Got {}",
                self.best_of
            )));
        }
        if self.temperature.is_some() && self.temperature.unwrap() > SAMPLING_EPS {
            return Err(APIError::new_str(
                "temperature must be 0 when using beam search",
            ));
        }
        if self.top_p.is_some() && self.top_p.unwrap() < 1.0f32 - SAMPLING_EPS {
            return Err(APIError::new_str("top_p must be 1 when using beam search"));
        }
        if self.top_k.is_some() && self.top_k.unwrap() != -1 {
            return Err(APIError::new_str("top_k must be -1 when using beam search"));
        }
        Ok(())
    }

    fn verify_non_beam_search(&self) -> Result<(), APIError> {
        if self.early_stopping != EarlyStoppingCondition::UnlikelyBetterCandidates {
            return Err(APIError::new_str("early_stopping is not effective and must be UnlikelyBetterCandidates when not using beam search."));
        }
        if self.length_penalty < 1.0f32 - SAMPLING_EPS
            || self.length_penalty > 1.0f32 + SAMPLING_EPS
        {
            return Err(APIError::new_str("length_penalty is not effective and must be the default value of 1.0 when not using beam search."));
        }
        Ok(())
    }

    fn verify_greedy_sampling(&self) -> Result<(), APIError> {
        if self.best_of > 1 {
            return Err(APIError::new(format!(
                "best_of must be 1 when using greedy sampling. Got {}.",
                self.best_of
            )));
        }
        if self.top_p.is_some() && self.top_p.unwrap() < 1.0f32 - SAMPLING_EPS {
            return Err(APIError::new_str(
                "top_p must be 1 when using greedy sampling (no temperature specified).",
            ));
        }
        if self.top_k.is_some() && self.top_k.unwrap() != -1 {
            return Err(APIError::new_str(
                "top_k must be -1 when using greedy sampling (no temperature specified).",
            ));
        }
        Ok(())
    }
}
