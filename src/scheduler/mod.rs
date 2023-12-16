use std::collections::HashMap;

pub mod block_engine;
pub mod cache_engine;
pub mod scheduler;
pub mod sequence;

type CPUBlockFrom = usize;
type GPUBlockFrom = usize;
type CPUBlockTo = usize;
type GPUBlockTo = usize;
type SrcBlockFrom = usize;
type DstBlocksTo = Vec<usize>;

type SrcBlock = usize;
type DstBlock = usize;
type DstBlocks = Vec<DstBlock>;
