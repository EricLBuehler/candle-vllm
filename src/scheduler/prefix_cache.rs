use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::block_engine::PhysicalTokenBlock;

#[derive(Clone, Debug)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_cached_blocks: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_cached_blocks: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrefixMatch {
    pub matched_blocks: usize,
    pub last_hash: Option<u64>,
}

#[derive(Clone)]
struct PrefixEntry {
    parent: Option<u64>,
    block: Arc<PhysicalTokenBlock>,
    children: usize,
    access_id: u64,
}

pub struct PrefixCache {
    block_size: usize,
    config: PrefixCacheConfig,
    entries: HashMap<u64, PrefixEntry>,
    leaf_set: HashSet<u64>,
    leaf_lru: VecDeque<(u64, u64)>,
    access_counter: u64,
}

impl PrefixCache {
    pub fn new(block_size: usize, config: PrefixCacheConfig) -> Self {
        Self {
            block_size,
            config,
            entries: HashMap::new(),
            leaf_set: HashSet::new(),
            leaf_lru: VecDeque::new(),
            access_counter: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled && self.config.max_cached_blocks > 0
    }

    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        if !self.enabled() {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
            };
        }

        let mut matched = 0usize;
        let mut parent_hash = 0u64;
        let mut last_hash = None;
        for block_tokens in tokens.chunks(self.block_size).take(full_blocks) {
            let hash = Self::hash_block(parent_hash, block_tokens);
            if self.entries.contains_key(&hash) {
                matched += 1;
                parent_hash = hash;
                last_hash = Some(hash);
                self.touch(hash);
            } else {
                break;
            }
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash,
        }
    }

    pub fn blocks_for_match(&self, last_hash: u64) -> Vec<Arc<PhysicalTokenBlock>> {
        let mut blocks = Vec::new();
        let mut current = Some(last_hash);
        while let Some(hash) = current {
            let entry = match self.entries.get(&hash) {
                Some(entry) => entry,
                None => break,
            };
            blocks.push(entry.block.clone());
            current = entry.parent;
        }
        blocks.reverse();
        blocks
    }

    pub fn insert_prefix(
        &mut self,
        tokens: &[u32],
        blocks: &[Arc<PhysicalTokenBlock>],
    ) -> Vec<Arc<PhysicalTokenBlock>> {
        if !self.enabled() {
            return Vec::new();
        }

        let full_blocks = tokens.len() / self.block_size;
        let max_blocks = std::cmp::min(full_blocks, blocks.len());
        if max_blocks == 0 {
            return Vec::new();
        }

        let mut parent_hash = None;
        for (block, block_tokens) in blocks
            .iter()
            .zip(tokens.chunks(self.block_size))
            .take(max_blocks)
        {
            let hash = Self::hash_block(parent_hash.unwrap_or(0), block_tokens);
            if self.entries.contains_key(&hash) {
                let access_id = self.next_access_id();
                if let Some(entry) = self.entries.get_mut(&hash) {
                    entry.access_id = access_id;
                }
                self.touch_leaf(hash);
            } else {
                if let Some(parent) = parent_hash {
                    if let Some(parent_entry) = self.entries.get_mut(&parent) {
                        if parent_entry.children == 0 {
                            self.leaf_set.remove(&parent);
                        }
                        parent_entry.children += 1;
                    }
                }
                block.deref_mut().refcount += 1;
                let access_id = self.next_access_id();
                self.entries.insert(
                    hash,
                    PrefixEntry {
                        parent: parent_hash,
                        block: block.clone(),
                        children: 0,
                        access_id,
                    },
                );
                self.leaf_set.insert(hash);
                self.leaf_lru.push_back((hash, access_id));
            }
            parent_hash = Some(hash);
        }

        self.evict_if_needed()
    }

    fn touch(&mut self, hash: u64) {
        if self.entries.contains_key(&hash) {
            let access_id = self.next_access_id();
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.access_id = access_id;
            }
            self.touch_leaf(hash);
        }
    }

    fn touch_leaf(&mut self, hash: u64) {
        if self.leaf_set.contains(&hash) {
            if let Some(entry) = self.entries.get(&hash) {
                self.leaf_lru.push_back((hash, entry.access_id));
            }
        }
    }

    fn evict_if_needed(&mut self) -> Vec<Arc<PhysicalTokenBlock>> {
        let mut evicted = Vec::new();
        while self.entries.len() > self.config.max_cached_blocks {
            let (hash, access_id) = match self.leaf_lru.pop_front() {
                Some(candidate) => candidate,
                None => break,
            };
            if !self.leaf_set.contains(&hash) {
                continue;
            }
            let entry = match self.entries.get(&hash) {
                Some(entry) => entry,
                None => continue,
            };
            if entry.access_id != access_id || entry.children > 0 {
                continue;
            }
            let entry = self.entries.remove(&hash).unwrap();
            self.leaf_set.remove(&hash);
            evicted.push(entry.block.clone());
            if let Some(parent) = entry.parent {
                if let Some(parent_entry) = self.entries.get_mut(&parent) {
                    if parent_entry.children > 0 {
                        parent_entry.children -= 1;
                    }
                    if parent_entry.children == 0 {
                        self.leaf_set.insert(parent);
                        self.leaf_lru.push_back((parent, parent_entry.access_id));
                    }
                }
            }
        }
        evicted
    }

    fn next_access_id(&mut self) -> u64 {
        self.access_counter = self.access_counter.wrapping_add(1);
        self.access_counter
    }

    fn hash_block(parent_hash: u64, tokens: &[u32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_hash.hash(&mut hasher);
        tokens.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{PrefixCache, PrefixCacheConfig};
    use crate::scheduler::block_engine::{PhysicalTokenBlock, _PhysicalTokenBlock};
    use std::sync::{Arc, Mutex};

    fn block(block_id: usize, block_size: usize) -> Arc<PhysicalTokenBlock> {
        Arc::new(PhysicalTokenBlock(Mutex::new(_PhysicalTokenBlock {
            block_id,
            block_size,
            refcount: 1,
            is_gpu: true,
        })))
    }

    #[test]
    fn prefix_cache_matches_full_blocks() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 8,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![block(0, 4), block(1, 4)];
        let evicted = cache.insert_prefix(&tokens, &blocks);
        assert!(evicted.is_empty());

        let match_info = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(match_info.matched_blocks, 2);

        let matched_blocks = cache.blocks_for_match(match_info.last_hash.unwrap());
        assert_eq!(matched_blocks.len(), 2);
        assert_eq!(matched_blocks[0].deref_mut().block_id, 0);
        assert_eq!(matched_blocks[1].deref_mut().block_id, 1);
    }

    #[test]
    fn prefix_cache_evicts_leaf_blocks() {
        let mut cache = PrefixCache::new(
            4,
            PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: 1,
            },
        );

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let blocks = vec![block(5, 4), block(6, 4)];
        let evicted = cache.insert_prefix(&tokens, &blocks);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].deref_mut().block_id, 6);

        let match_info = cache.match_prefix(&tokens);
        assert_eq!(match_info.matched_blocks, 1);
    }
}
