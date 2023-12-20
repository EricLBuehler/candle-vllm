use candle_core::{DType, Device, Shape, Tensor};

use crate::{openai::responses::APIError, try_api};

// https://github.com/mokeyish/candle-ext/blob/main/src/triangular.rs
pub(crate) fn apply_triangular(
    xs: &Tensor,
    diagonal: isize,
    upper: bool,
) -> Result<Tensor, APIError> {
    let device = xs.device();
    let (l, s) = try_api!(xs.dims2());
    let mut xs_tri = vec![];
    for i in 0..l.try_into().unwrap() {
        for j in 0..s.try_into().unwrap() {
            let cond = if upper {
                i + diagonal > j
            } else {
                i + diagonal < j
            };
            xs_tri.push(if cond { 0u8 } else { 1u8 });
        }
    }
    (xs * try_api!(try_api!(Tensor::from_vec(xs_tri, (l * s,), device)).to_dtype(xs.dtype())))
        .map_err(APIError::from)
}

pub(crate) fn materialize_causal_mask(
    shape: &Shape,
    dtype: DType,
    device: &Device,
    window_size: Option<usize>,
    from_bottomright: bool,
) -> Result<Tensor, APIError> {
    let create_as = if dtype != DType::BF16 {
        dtype
    } else {
        DType::F32
    };
    let tensor = try_api!(Tensor::ones(shape, create_as, device));

    let mut shift = 0usize;
    if from_bottomright {
        let num_queries = shape.dims()[shape.dims().len() - 2];
        let num_keys = shape.dims()[shape.dims().len() - 1];
        shift = num_keys - num_queries;
    }

    let mut mask = try_api!(apply_triangular(&tensor, shift.try_into().unwrap(), false));
    if let Some(window_size) = window_size {
        mask = try_api!(apply_triangular(
            &mask,
            (shift - window_size + 1).try_into().unwrap(),
            false
        ));
    }
    try_api!(mask.log()).to_dtype(dtype).map_err(APIError::from)
}
