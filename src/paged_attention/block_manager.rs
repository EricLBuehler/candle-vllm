use std::collections::HashMap;

use candle_core::Device;
use range_checked::F64Bounded;

use crate::openai::responses::APIError;

pub struct PhysicalTokenBlock {
    device: Device,
    block_number: usize,
    block_size: usize,
    ref_count: usize,
}

impl PhysicalTokenBlock {
    pub fn new(device: Device, block_number: usize, block_size: usize) -> Self {
        Self {
            device,
            block_number,
            block_size,
            ref_count: 0,
        }
    }
}

pub struct BlockAllocator {
    device: Device,
    block_size: usize,
    num_blocks: usize,
    free_blocks: Vec<PhysicalTokenBlock>,
}

impl BlockAllocator {
    pub fn new(device: Device, block_size: usize, num_blocks: usize) -> Self {
        Self {
            device: device.clone(),
            block_size,
            num_blocks,
            free_blocks: {
                let mut blocks = Vec::new();
                for i in 0..num_blocks {
                    blocks.push(PhysicalTokenBlock::new(device.clone(), i, block_size));
                }
                blocks
            },
        }
    }
}

pub struct BlockSpaceManager {
    block_size: usize,
    num_total_gpu_blocks: usize,
    num_total_cpu_blocks: usize,
    block_sliding_window: Option<usize>,
    watermark: f64,
    watermark_bias: usize,
    gpu_allocator: BlockAllocator,
    cpu_allocator: BlockAllocator,
    block_tables: HashMap<usize, Vec<PhysicalTokenBlock>>,
}

impl BlockSpaceManager {
    pub fn new(
        block_size: usize,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        watermark: F64Bounded<0, { i32::MAX }, true>,
        sliding_window: Option<usize>,
    ) -> Result<Self, APIError> {
        Ok(Self {
            block_size,
            num_total_cpu_blocks: num_cpu_blocks,
            num_total_gpu_blocks: num_gpu_blocks,
            block_sliding_window: if let Some(sliding_window) = sliding_window {
                assert_eq!(sliding_window % block_size, 0);
                Some(sliding_window / block_size)
            } else {
                None
            },
            watermark: *watermark,
            watermark_bias: (*watermark * num_gpu_blocks as f64) as usize,
            gpu_allocator: BlockAllocator::new(
                Device::new_cuda(0).map_err(APIError::from)?,
                block_size,
                num_gpu_blocks,
            ),
            cpu_allocator: BlockAllocator::new(Device::Cpu, block_size, num_cpu_blocks),
            block_tables: HashMap::new(),
        })
    }
}
