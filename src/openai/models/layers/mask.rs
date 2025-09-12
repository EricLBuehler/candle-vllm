use candle_core::{DType, Device, Tensor};

#[cfg(feature = "flash-attn")] // If flash-attn or metal is enabled, we don't implement this function.
                               // The actual implementation would be embedded in the flash or metal attention kernel.
pub fn get_attention_casual_mask(
    _: &Device,
    _: DType,
    _: &Tensor,
    _: &Vec<u32>,
    _: Option<usize>,
    _: bool,
) -> Option<Vec<Tensor>> {
    None
}

#[cfg(not(feature = "flash-attn"))]
fn get_casual_mask_internal(
    device: &Device,
    dtype: DType,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
) -> candle_core::Result<Tensor> {
    use candle_core::D;
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((1, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

#[cfg(not(feature = "flash-attn"))]
pub fn get_attention_casual_mask(
    device: &Device,
    dtype: DType,
    _: &Tensor,
    seqlens: &Vec<u32>,
    sliding_window: Option<usize>,
    is_prefill: bool,
) -> Option<Vec<Tensor>> {
    if !is_prefill {
        return None;
    }
    // let vec_positions = positions.to_vec1::<i64>().unwrap();
    let mut offsets = vec![0u32];
    offsets.extend(seqlens.clone());
    let mut vec_mask = Vec::new();
    let mut start = 0;
    for (_, seq_offset) in seqlens.iter().enumerate() {
        let seq_len = seq_offset - start;
        let mask = get_casual_mask_internal(
            device,
            dtype,
            seq_len as usize,
            0,
            // vec_positions[offsets[i] as usize] as usize,
            sliding_window,
        )
        .unwrap();
        vec_mask.push(mask);
        start = *seq_offset;
    }
    Some(vec_mask)
}
