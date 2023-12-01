use candle_core::Tensor;

use crate::openai::responses::APIError;

use super::attn_bias::AttentionBias;

pub struct InputMetadata {
    pub max_context_len: usize,
    pub block_tables: Tensor,
    pub context_lens: Tensor,
    pub num_prompt_tokens: usize,
    pub num_generation_tokens: usize,
    pub slot_mappinng: Tensor,
    pub to_cache: Option<Tensor>,
    pub attn_bias: Option<Box<dyn AttentionBias>>,
    pub max_prompt_len: usize,
    pub num_prompts: usize,
}

impl InputMetadata {
    /// prompt_lens: Lengths of prompts.
    /// slot_mapping: The address to write the new KV to of each token.
    /// context_lens: the length of attention context for each generation token.
    /// max_context_len: The maximum context length.
    /// block_tables: The block tables. (Seq id -> list of physical block)
    pub fn new(
        max_context_len: usize,
        block_tables: Tensor,
        context_lens: Tensor,
        prompt_lens: &[u32],
        slot_mappinng: Tensor,
        sliding_window: Option<u32>,
    ) -> Result<Self, APIError> {
        let num_prompts = prompt_lens.len();
        let max_prompt_len = if !prompt_lens.is_empty() {
            prompt_lens.iter().fold(u32::MIN, |a, b| a.max(*b))
        } else {
            0
        };
        let to_cache = if let Some(sliding_window) = sliding_window {
            let mut to_cache: Vec<u32> = Vec::new();
            let mut start_idx: u32 = 0;
            for prompt_len in prompt_lens {
                to_cache.extend(
                    (start_idx + 0.max(prompt_len - sliding_window))..(start_idx + prompt_len),
                );
                start_idx += max_prompt_len;
            }
            to_cache.extend(start_idx..*slot_mappinng.shape().dims().get(0).unwrap() as u32);
            let len = to_cache.len();
            let out = Tensor::from_vec(to_cache, (len,), slot_mappinng.device())
                .map_err(APIError::from)?;
            out.to_dtype(candle_core::DType::U32)
                .map_err(APIError::from)?;
            Some(out)
        } else {
            None
        };

        let num_generation_tokens = *context_lens.shape().dims().get(0).unwrap();
        Ok(Self {
            max_context_len,
            block_tables,
            context_lens,
            num_prompt_tokens: num_prompts * max_context_len,
            num_generation_tokens,
            slot_mappinng,
            to_cache,
            attn_bias: None,
            max_prompt_len: max_prompt_len as usize,
            num_prompts: prompt_lens.len(),
        })
    }
}
