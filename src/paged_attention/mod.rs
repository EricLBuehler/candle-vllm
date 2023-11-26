use candle_core::{DType, Device, Tensor};
use tch::Kind;

use crate::{
    openai::responses::APIError,
    paged_attention::bindings::{paged_attention_v1, paged_attention_v2, Optional, Storage},
};

use self::input_metadata::InputMetadata;
mod bindings;
mod input_metadata;

const _PARTION_SIZE: usize = 512;

pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    head_mapping: Tensor,
}

fn convert_candle_to_tch(candle: &mut Tensor) -> tch::Tensor {
    let output_kind = match candle.dtype() {
        DType::BF16 => Kind::BFloat16,
        DType::F16 => Kind::Float,
        DType::F32 => Kind::Float,
        DType::F64 => Kind::Float,
        DType::I64 => Kind::Int64,
        DType::U8 => Kind::Uint8,
        DType::U32 => Kind::Int,
    };

    let mut dims = Vec::new();
    for dim in candle.dims() {
        dims.push(*dim as i64);
    }

    tch::Tensor::from_data_size(
        &candle
            .to_vec3::<u8>()
            .unwrap()
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>()[..],
        &dims[..],
        output_kind,
    )
}

fn convert_tch_to_ptr(
    tch: &mut tch::Tensor,
) -> (*mut torch_sys::C_tensor, &mut torch_sys::C_tensor) {
    (tch.as_mut_ptr(), unsafe { &mut *tch.as_mut_ptr() })
}

fn convert_option_to_optional(option: Option<Tensor>) -> Optional<torch_sys::C_tensor> {
    let storage: Storage<torch_sys::C_tensor> = if let Some(mut value) = option {
        let mut slopes_tchtensor = convert_candle_to_tch(&mut value);
        let (_, slopes_ctensor) = convert_tch_to_ptr(&mut slopes_tchtensor);
        Storage {
            value_: *slopes_ctensor,
        }
    } else {
        Storage {
            dummy_: std::os::raw::c_uchar::from(0),
        }
    };

    Optional {
        init_: true,
        storage_: storage,
    }
}

impl PagedAttention {
    pub fn new(
        num_attention_heads: usize,
        head_dim: usize,
        scale: f32,
        num_key_value_heads: Option<usize>,
        sliding_window: Option<usize>,
        device: Device,
    ) -> Result<Self, APIError> {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_key_value_heads;
        Ok(Self {
            num_attention_heads,
            head_dim,
            num_key_value_heads,
            scale,
            sliding_window,
            num_queries_per_kv,
            head_mapping: Tensor::arange(0u32, num_key_value_heads as u32, &device)
                .map_err(APIError::from)?
                .repeat(num_queries_per_kv)
                .map_err(APIError::from)?,
        })
    }

    /// Args:
    /// output: shape = [num_generation_tokens, num_heads, head_size]
    ///
    /// query: shape = [num_generation_tokens, num_heads, head_size]
    ///
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    ///
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    ///
    /// input_metadata: metadata for paged attention.
    ///
    /// alibi_slopes: shape = [num_heads]
    pub fn single_query_kv_attention(
        &mut self,
        output: &mut Tensor,
        mut query: Tensor,
        mut key_cache: Tensor,
        mut value_cache: Tensor,
        input_metadata: &mut InputMetadata,
        alibi_slopes: Option<Tensor>,
    ) -> Result<(), APIError> {
        let block_size = *value_cache.shape().dims().get(3).unwrap();
        let (num_seqs, num_heads, head_size) = query.shape().dims3().map_err(APIError::from)?;
        let max_num_partions = (input_metadata.max_context_len + _PARTION_SIZE - 1) / _PARTION_SIZE;

        let use_v1 = input_metadata.max_context_len <= 8192
            && (max_num_partions == 1 || num_seqs * num_heads > 512);
        if use_v1 {
            //Run PagedAttention V1
            let mut output_tch = convert_candle_to_tch(output);
            let (output_ptr, _) = convert_tch_to_ptr(&mut output_tch);

            let mut query_tch = convert_candle_to_tch(&mut query);
            let (query_ptr, _) = convert_tch_to_ptr(&mut query_tch);

            let mut keycache_tch = convert_candle_to_tch(&mut key_cache);
            let (keycache_ptr, _) = convert_tch_to_ptr(&mut keycache_tch);

            let mut valuecache_tch = convert_candle_to_tch(&mut value_cache);
            let (valuecache_ptr, _) = convert_tch_to_ptr(&mut valuecache_tch);

            let mut head_mapping_tch = convert_candle_to_tch(&mut self.head_mapping);
            let (head_mapping_ptr, _) = convert_tch_to_ptr(&mut head_mapping_tch);

            let mut block_tbl_tch = convert_candle_to_tch(&mut input_metadata.block_tables);
            let (block_tbl_ptr, _) = convert_tch_to_ptr(&mut block_tbl_tch);

            let mut ctxt_lens_tch = convert_candle_to_tch(&mut input_metadata.context_lens);
            let (ctxt_lens_ptr, _) = convert_tch_to_ptr(&mut ctxt_lens_tch);

            let optional = convert_option_to_optional(alibi_slopes);

            unsafe {
                paged_attention_v1(
                    output_ptr,
                    query_ptr,
                    keycache_ptr,
                    valuecache_ptr,
                    head_mapping_ptr,
                    self.scale,
                    block_tbl_ptr,
                    ctxt_lens_ptr,
                    block_size.try_into().unwrap(),
                    input_metadata.max_context_len.try_into().unwrap(),
                    &optional as *const Optional<torch_sys::C_tensor>,
                )
            };
        } else {
            //Run PagedAttention V2
            assert_eq!(_PARTION_SIZE % block_size, 0);

            let mut tmp_output = Tensor::zeros(
                //use Tensor::empty, huggingface/candle#1374
                (num_seqs, num_heads, max_num_partions, head_size),
                output.dtype(),
                output.device(),
            )
            .map_err(APIError::from)?;
            let mut exp_sums = Tensor::zeros(
                //use Tensor::empty, huggingface/candle#1374
                (num_seqs, num_heads, max_num_partions),
                DType::F32,
                output.device(),
            )
            .map_err(APIError::from)?;
            let mut max_logits = exp_sums.zeros_like().map_err(APIError::from)?; //use Tensor::empty, huggingface/candle#1374

            let mut output_tch = convert_candle_to_tch(output);
            let (output_ptr, _) = convert_tch_to_ptr(&mut output_tch);

            let mut exp_sums_tch = convert_candle_to_tch(&mut exp_sums);
            let (exp_sums_ptr, _) = convert_tch_to_ptr(&mut exp_sums_tch);

            let mut max_logits_tch = convert_candle_to_tch(&mut max_logits);
            let (max_logits_ptr, _) = convert_tch_to_ptr(&mut max_logits_tch);

            let mut tmp_output_tch = convert_candle_to_tch(&mut tmp_output);
            let (tmp_output_ptr, _) = convert_tch_to_ptr(&mut tmp_output_tch);

            let mut query_tch = convert_candle_to_tch(&mut query);
            let (query_ptr, _) = convert_tch_to_ptr(&mut query_tch);

            let mut keycache_tch = convert_candle_to_tch(&mut key_cache);
            let (keycache_ptr, _) = convert_tch_to_ptr(&mut keycache_tch);

            let mut valuecache_tch = convert_candle_to_tch(&mut value_cache);
            let (valuecache_ptr, _) = convert_tch_to_ptr(&mut valuecache_tch);

            let mut head_mapping_tch = convert_candle_to_tch(&mut self.head_mapping);
            let (head_mapping_ptr, _) = convert_tch_to_ptr(&mut head_mapping_tch);

            let mut block_tbl_tch = convert_candle_to_tch(&mut input_metadata.block_tables);
            let (block_tbl_ptr, _) = convert_tch_to_ptr(&mut block_tbl_tch);

            let mut ctxt_lens_tch = convert_candle_to_tch(&mut input_metadata.context_lens);
            let (ctxt_lens_ptr, _) = convert_tch_to_ptr(&mut ctxt_lens_tch);

            let optional = convert_option_to_optional(alibi_slopes);

            unsafe {
                paged_attention_v2(
                    output_ptr,
                    exp_sums_ptr,
                    max_logits_ptr,
                    tmp_output_ptr,
                    query_ptr,
                    keycache_ptr,
                    valuecache_ptr,
                    head_mapping_ptr,
                    self.scale,
                    block_tbl_ptr,
                    ctxt_lens_ptr,
                    block_size.try_into().unwrap(),
                    input_metadata.max_context_len.try_into().unwrap(),
                    &optional as *const Optional<torch_sys::C_tensor>,
                );
            };
        }
        Ok(())
    }
}
