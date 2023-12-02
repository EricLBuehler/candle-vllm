use std::iter::zip;

use candle_core::{DType, Device, Shape, Tensor};

use crate::openai::responses::APIError;

pub trait AttentionBiasBlockDiagonal {
    /// Queries and Keys are each divided into the same number of blocks.
    /// A query Q in block i cannot attend to a key which is not in block i,
    /// nor one which is farther from the initial key in block i than Q
    /// is from the initial query in block i.
    fn materialize(&self, shape: &Shape, dtype: DType, device: Device) -> Result<Tensor, APIError> {
        //use Tensor::empty, huggingface/candle#1374
        let mut mask = Tensor::new(
            &shape.dims().iter().map(|x| (*x) as u32).collect::<Vec<_>>()[2..],
            &device,
        )
        .map_err(APIError::from)?
        .to_dtype(dtype)
        .map_err(APIError::from)?;

        for (_, ((q_start, q_end), (k_start, k_end))) in zip(
            self.get_q_seqinfo().intervals(),
            self.get_k_seqinfo().intervals(),
        )
        .enumerate()
        {
            mask.slice_assign(
                &[
                    q_start as usize..q_end as usize,
                    k_start as usize..k_end as usize,
                ],
                &self._create_block_mask(
                    &Shape::from_dims(&[(q_end - q_start) as usize, (k_end - k_start) as usize]),
                    dtype,
                    &device,
                )?,
            )
            .map_err(APIError::from)?;
        }

        for _ in 0..shape.dims().len() - 2 {
            mask = mask.unsqueeze(0).map_err(APIError::from)?;
        }
        Ok(mask.expand(shape).map_err(APIError::from)?)
    }

    fn get_q_seqinfo(&self) -> &SeqLenInfo;

    fn get_k_seqinfo(&self) -> &SeqLenInfo;

    fn _create_block_mask(
        &self,
        shape: &Shape,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor, APIError>;

    fn make_local_attention(
        &self,
        _window_size: usize,
    ) -> Result<Box<dyn AttentionBiasBlockDiagonal>, APIError> {
        unimplemented!();
    }
}

#[derive(Clone)]
pub struct SeqLenInfo {
    seqstart_py: Vec<u32>,
}

impl SeqLenInfo {
    fn new(seqstart_py: Vec<u32>) -> Self {
        Self { seqstart_py }
    }

    fn from_seqlens<'a>(seqlens: impl Iterator<Item = &'a u32>) -> Result<Self, APIError> {
        let mut seqstart_py = vec![0];
        for seqlen in seqlens.into_iter() {
            seqstart_py.push(seqstart_py[seqstart_py.len() - 1] + seqlen);
        }
        Ok(Self::new(seqstart_py))
    }

    fn intervals(&self) -> Box<dyn Iterator<Item = (u32, u32)>> {
        Box::new(zip(
            self.seqstart_py.clone(),
            (&self.seqstart_py[1..]).iter().copied().collect::<Vec<_>>(),
        ))
    }
}

pub struct BlockDiagonalCausalMask {
    q_seqinfo: SeqLenInfo,
    k_seqinfo: SeqLenInfo,
    _batch_sizes: Option<Vec<usize>>,
}

impl BlockDiagonalCausalMask {
    fn new(q_seqinfo: SeqLenInfo, k_seqinfo: SeqLenInfo, _batch_sizes: Option<Vec<usize>>) -> Self {
        Self {
            q_seqinfo,
            k_seqinfo,
            _batch_sizes,
        }
    }

    pub fn from_seqlens(
        q_seqlen: Vec<u32>,
        kv_seqlen: Option<Vec<u32>>,
    ) -> Result<Box<dyn AttentionBiasBlockDiagonal>, APIError> {
        assert!(kv_seqlen.is_none() || q_seqlen.len() == kv_seqlen.as_ref().unwrap().len());
        let q_seqinfo = SeqLenInfo::from_seqlens(q_seqlen.iter()).map_err(APIError::from)?;
        let k_seqinfo = if kv_seqlen.is_none() || &q_seqlen == kv_seqlen.as_ref().unwrap() {
            q_seqinfo.clone()
        } else {
            SeqLenInfo::from_seqlens(kv_seqlen.unwrap().iter()).map_err(APIError::from)?
        };
        Ok(Box::new(Self::new(q_seqinfo, k_seqinfo, None)))
    }
}

impl AttentionBiasBlockDiagonal for BlockDiagonalCausalMask {
    fn _create_block_mask(
        &self,
        shape: &Shape,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor, APIError> {
        Ok(Tensor::zeros(shape, dtype, device).map_err(APIError::from)?)
    }

    fn get_k_seqinfo(&self) -> &SeqLenInfo {
        &self.k_seqinfo
    }

    fn get_q_seqinfo(&self) -> &SeqLenInfo {
        &self.q_seqinfo
    }

    fn make_local_attention(
        &self,
        window_size: usize,
    ) -> Result<Box<dyn AttentionBiasBlockDiagonal>, APIError> {
        Ok(Box::new(BlockDiagonalCausalLocalAttentionMask::new(
            self.q_seqinfo.clone(),
            self.k_seqinfo.clone(),
            self._batch_sizes.clone(),
            window_size,
        )))
    }
}

pub struct BlockDiagonalCausalLocalAttentionMask {
    q_seqinfo: SeqLenInfo,
    k_seqinfo: SeqLenInfo,
    _batch_sizes: Option<Vec<usize>>,
    _window_size: usize,
}

impl BlockDiagonalCausalLocalAttentionMask {
    fn new(
        q_seqinfo: SeqLenInfo,
        k_seqinfo: SeqLenInfo,
        _batch_sizes: Option<Vec<usize>>,
        window_size: usize,
    ) -> Self {
        Self {
            q_seqinfo,
            k_seqinfo,
            _batch_sizes,
            _window_size: window_size,
        }
    }
}

impl AttentionBiasBlockDiagonal for BlockDiagonalCausalLocalAttentionMask {
    fn _create_block_mask(
        &self,
        shape: &Shape,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor, APIError> {
        materialize_causal_mask(shape, dtype, device, Some(self._window_size), false)
    }

    fn get_k_seqinfo(&self) -> &SeqLenInfo {
        &self.k_seqinfo
    }

    fn get_q_seqinfo(&self) -> &SeqLenInfo {
        &self.q_seqinfo
    }
}

pub struct LowerTriangularMaskWithTensorBias {
    bias: Tensor,
}

impl LowerTriangularMaskWithTensorBias {
    pub fn new(bias: Tensor) -> Self {
        Self { bias }
    }
}

impl AttentionBiasBlockDiagonal for LowerTriangularMaskWithTensorBias {
    fn materialize(&self, shape: &Shape, dtype: DType, device: Device) -> Result<Tensor, APIError> {
        Ok(
            (materialize_causal_mask(shape, dtype, &device, None, false)
                .map_err(APIError::from)?
                + &self.bias)
                .map_err(APIError::from)?,
        )
    }
    fn _create_block_mask(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _device: &Device,
    ) -> Result<Tensor, APIError> {
        unimplemented!("should not be called");
    }
    fn get_k_seqinfo(&self) -> &SeqLenInfo {
        unimplemented!("should not be called");
    }
    fn get_q_seqinfo(&self) -> &SeqLenInfo {
        unimplemented!("should not be called");
    }
    fn make_local_attention(
        &self,
        _window_size: usize,
    ) -> Result<Box<dyn AttentionBiasBlockDiagonal>, APIError> {
        unimplemented!("should not be called");
    }
}

// https://github.com/mokeyish/candle-ext/blob/main/src/triangular.rs
fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> Result<Tensor, APIError> {
    let device = xs.device();
    let (l, s) = xs.dims2().map_err(APIError::from)?;
    let mut xs_tri = vec![];
    for i in 0..l as isize {
        for j in 0..s as isize {
            let cond = if upper {
                i + diagonal > j
            } else {
                i + diagonal < j
            };
            xs_tri.push(if cond { 0u8 } else { 1u8 });
        }
    }
    Ok((xs
        * Tensor::from_vec(xs_tri, (l, s), device)
            .map_err(APIError::from)?
            .to_dtype(xs.dtype())
            .map_err(APIError::from)?)
    .map_err(APIError::from)?)
}

fn materialize_causal_mask(
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
    let tensor = Tensor::ones(shape, create_as, device).map_err(APIError::from)?;

    let mut shift = 0usize;
    if from_bottomright {
        let num_queries = shape.dims()[shape.dims().len() - 2];
        let num_keys = shape.dims()[shape.dims().len() - 1];
        shift = num_keys - num_queries;
    }

    let mut mask =
        apply_triangular(&tensor, shift.try_into().unwrap(), false).map_err(APIError::from)?;
    if let Some(window_size) = window_size {
        mask = apply_triangular(&mask, (shift - window_size + 1).try_into().unwrap(), false)
            .map_err(APIError::from)?;
    }
    Ok(mask
        .log()
        .map_err(APIError::from)?
        .to_dtype(dtype)
        .map_err(APIError::from)?)
}
