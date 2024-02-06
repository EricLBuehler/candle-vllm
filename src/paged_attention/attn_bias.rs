#![allow(dead_code)]

use std::iter::zip;

use candle_core::{DType, Device, Shape, Tensor};

use crate::openai::responses::APIError;

use crate::paged_attention::utils;
use crate::try_api;

pub trait AttentionBiasBlockDiagonal {
    /// Queries and Keys are each divided into the same number of blocks.
    /// A query Q in block i cannot attend to a key which is not in block i,
    /// nor one which is farther from the initial key in block i than Q
    /// is from the initial query in block i.
    fn materialize(
        &self,
        shape: &Shape,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor, APIError> {
        let mut mask =
            try_api!(
                try_api!(Tensor::zeros(&shape.dims().to_vec()[2..], dtype, device,))
                    .to_dtype(dtype)
            );

        for ((q_start, q_end), (k_start, k_end)) in zip(
            self.get_q_seqinfo().intervals(),
            self.get_k_seqinfo().intervals(),
        ) {
            try_api!(mask.slice_assign(
                &[
                    q_start as usize..q_end as usize,
                    k_start as usize..k_end as usize,
                ],
                &self._create_block_mask(
                    &Shape::from_dims(&[(q_end - q_start) as usize, (k_end - k_start) as usize]),
                    dtype,
                    device,
                )?,
            ));
        }

        for _ in 0..shape.dims().len() - 2 {
            mask = try_api!(mask.unsqueeze(0));
        }
        mask.expand(shape).map_err(APIError::from)
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
            self.seqstart_py[1..].to_vec(),
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
        let q_seqinfo = try_api!(SeqLenInfo::from_seqlens(q_seqlen.iter()));
        let k_seqinfo = if kv_seqlen.is_none() || &q_seqlen == kv_seqlen.as_ref().unwrap() {
            q_seqinfo.clone()
        } else {
            try_api!(SeqLenInfo::from_seqlens(kv_seqlen.unwrap().iter()))
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
        Tensor::zeros(shape, dtype, device).map_err(APIError::from)
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
        utils::materialize_causal_mask(shape, dtype, device, Some(self._window_size), false)
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
    fn materialize(
        &self,
        shape: &Shape,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor, APIError> {
        (try_api!(utils::materialize_causal_mask(
            shape, dtype, device, None, false
        )) + &self.bias)
            .map_err(APIError::from)
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
