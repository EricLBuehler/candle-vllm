use super::DefaultPipeline;
#[cfg(feature = "nccl")]
use crate::openai::communicator::{DaemonManager, MessageType, TaskSampleData};
#[cfg(feature = "nccl")]
use crate::openai::pipelines::TokenOrFinishReason;
use crate::openai::streaming::ChatResponse;
use crate::openai::TaskData;
use crate::scheduler::Scheduler;
use crate::tools::stream_parser::{
    BufferedFinalizeResult, ParserState, StreamResult, StreamToolParser,
};
#[cfg(feature = "flashinfer")]
use crate::FlashInferKvParams;
use crate::{
    openai::{
        models::Config,
        responses::{
            ChatChoice, ChatChoiceData, ChatCompletionChunk, ChatCompletionUsageResponse, Choice,
            ChoiceData, EmbeddingData, EmbeddingOutput, EmbeddingResponse, EmbeddingUsage,
            WrapperLogprobs,
        },
        sampling_params::Logprobs,
        sampling_params::SamplingParams,
        utils::get_created_time_secs,
    },
    scheduler::{
        cache_engine::{CacheConfig, CacheEngine},
        sequence::{Sequence, SequenceGroup, _Sequence},
        SchedulerConfig, SchedulerOutput,
    },
    InputMetadata,
};
#[cfg(feature = "flashinfer")]
use attention_rs::FlashInferMetadata;
use candle_core::{Device, Result, Tensor};
use either::Either;
use flume::Sender;
use parking_lot::RwLock;
#[cfg(feature = "nccl")]
use rayon::iter::IntoParallelRefIterator;
#[cfg(feature = "nccl")]
use rayon::iter::ParallelIterator;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    iter::zip,
    sync::Arc,
};
use tokio::sync::Notify;
#[allow(unused_imports)]
use tracing::{debug, info, warn};
#[allow(dead_code)]
struct PreparedInputs {
    tokens: Tensor,
    positions: Tensor,
    metadata: InputMetadata,
}

struct StreamEmission {
    content: Option<String>,
    tool_calls: Option<Vec<crate::tools::ToolCall>>,
}

const _PAD_SLOT_ID: i64 = -1;
const PREFILL_CHUNK_SIZE: usize = 8192;

#[allow(unused)]
pub struct LLMEngine {
    pub pipelines: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)>,
    pub scheduler: Scheduler,
    seq_id: usize,
    cache_config: CacheConfig,
    pub config: Config,
    group_id: usize,
    pub notify: Arc<Notify>,
    pub sync_notifies: HashMap<String, Option<Arc<Notify>>>,
    pub senders: HashMap<String, Option<Arc<Sender<ChatResponse>>>>,
    pub completion_records: HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>,
    sequence_groups: RwLock<VecDeque<Arc<SequenceGroup>>>,
    multi_process: bool,
    num_shards: usize,
    waiting_tasks: RwLock<Vec<TaskData>>,
    #[cfg(feature = "nccl")]
    pub daemon_manager: RwLock<Option<DaemonManager>>,
    prefill_chunk_size: Option<usize>,
    restored_prefix_sequences: RwLock<HashSet<usize>>,
    pub exit_flag: Arc<AtomicBool>,
    last_decoding_throughput_log_ms: usize,
}

impl LLMEngine {
    fn ordered_group_sequences(group: &Arc<SequenceGroup>) -> Vec<Arc<Sequence>> {
        let mut seqs = group
            .get_seqs()
            .iter()
            .map(|(seq_id, seq)| (*seq_id, Arc::clone(seq)))
            .collect::<Vec<_>>();
        seqs.sort_unstable_by_key(|(seq_id, _)| *seq_id);
        seqs.into_iter().map(|(_, seq)| seq).collect()
    }

    fn primary_sequence(group: &Arc<SequenceGroup>) -> Arc<Sequence> {
        Self::ordered_group_sequences(group)
            .into_iter()
            .next()
            .expect("sequence group must contain at least one sequence")
    }

    #[cfg(feature = "nccl")]
    fn primary_sequence_id(group: &Arc<SequenceGroup>) -> usize {
        Self::primary_sequence(group).deref().get_id()
    }

    fn fallback_sequence_to_full_prefill(
        &mut self,
        sequence: &Sequence,
        seq_id: usize,
        cached_tokens: usize,
        reason: &str,
    ) {
        let _ = self.restored_prefix_sequences.write().remove(&seq_id);
        let rebuilt = self
            .scheduler
            .block_engine
            .fallback_sequence_to_full_prefill(sequence);
        if rebuilt {
            tracing::warn!(
                "Seq {} {} (cached {} tokens); rebuilt block table and falling back to full prefill",
                seq_id,
                reason,
                cached_tokens
            );
        } else {
            tracing::warn!(
                "Seq {} {} (cached {} tokens); unable to rebuild block table due memory pressure, keeping cached prefill",
                seq_id,
                reason,
                cached_tokens
            );
        }
    }

    fn capture_mamba_prefix_states_for_prefill_progress(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        chunk_size: usize,
        rank: usize,
    ) -> Result<()> {
        if groups.is_empty() || chunk_size == 0 {
            return Ok(());
        }

        let mut captures = Vec::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                let seq_id = seq.deref().get_id();
                let prompt_len = seq.deref().get_prompt_len();
                let num_cached_tokens = seq.deref().get_num_cached_tokens();
                if prompt_len == 0 || num_cached_tokens >= prompt_len {
                    continue;
                }

                let processed_tokens =
                    if prompt_len < chunk_size || num_cached_tokens + chunk_size >= prompt_len {
                        prompt_len
                    } else {
                        num_cached_tokens + chunk_size
                    };

                if let Some(hash) = self
                    .scheduler
                    .block_engine
                    .prefix_hash_for_sequence(&seq, processed_tokens)
                {
                    captures.push((seq_id, hash));
                }
            }
        }

        if captures.is_empty() {
            return Ok(());
        }

        let mut captured_hashes = Vec::new();
        {
            let pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            for (seq_id, hash) in captures {
                match pipeline.capture_mamba_prefix_state(seq_id, hash) {
                    Ok(true) => captured_hashes.push(hash),
                    Ok(false) => {}
                    Err(e) => {
                        tracing::warn!(
                            "Failed to capture prefill mamba prefix state for seq {} hash {}: {}",
                            seq_id,
                            hash,
                            e
                        );
                    }
                }
            }
        }
        for hash in captured_hashes {
            self.scheduler.block_engine.record_mamba_prefix_hash(hash);
        }
        Ok(())
    }

    fn capture_mamba_prefix_states_for_decode_progress(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        rank: usize,
    ) -> Result<()> {
        if groups.is_empty() {
            return Ok(());
        }

        let block_size = self.cache_config.block_size;
        let stride_blocks = crate::mamba_snapshot_block_stride_blocks();
        let mut captures = Vec::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                if seq.deref().is_finished() {
                    continue;
                }
                let seq_id = seq.deref().get_id();
                let seq_len = seq.deref().get_len();
                if seq_len < block_size || seq_len % block_size != 0 {
                    continue;
                }
                let full_blocks = seq_len / block_size;
                if stride_blocks > 1 && full_blocks % stride_blocks != 0 {
                    continue;
                }
                if let Some(hash) = self
                    .scheduler
                    .block_engine
                    .prefix_hash_for_sequence(&seq, seq_len)
                {
                    captures.push((seq_id, hash));
                }
            }
        }

        if captures.is_empty() {
            return Ok(());
        }

        let mut captured_hashes = Vec::new();
        {
            let pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            for (seq_id, hash) in captures {
                match pipeline.capture_mamba_prefix_state(seq_id, hash) {
                    Ok(true) => captured_hashes.push(hash),
                    Ok(false) => {}
                    Err(e) => {
                        tracing::warn!(
                            "Failed to capture decode mamba prefix state for seq {} hash {}: {}",
                            seq_id,
                            hash,
                            e
                        );
                    }
                }
            }
        }
        for hash in captured_hashes {
            self.scheduler.block_engine.record_mamba_prefix_hash(hash);
        }
        Ok(())
    }

    fn restore_mamba_prefix_states_for_prompt(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        rank: usize,
    ) -> Result<()> {
        if groups.is_empty() {
            return Ok(());
        }

        let mut restore_candidates = Vec::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                let seq_id = seq.deref().get_id();
                let cached_tokens = seq.deref().get_num_cached_tokens();
                if cached_tokens == 0 {
                    continue;
                }
                if self.restored_prefix_sequences.read().contains(&seq_id) {
                    continue;
                }
                restore_candidates.push(seq_id);
            }
        }

        if !restore_candidates.is_empty() {
            let (pipeline, _) = self.get_pipeline(rank).unwrap();
            pipeline.ensure_mamba_slots_for_sequences(&restore_candidates)?;
        }

        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                let seq_id = seq.deref().get_id();
                let cached_tokens = seq.deref().get_num_cached_tokens();
                if cached_tokens == 0 {
                    continue;
                }
                if self.restored_prefix_sequences.read().contains(&seq_id) {
                    continue;
                }
                if let Some(hash) = self
                    .scheduler
                    .block_engine
                    .prefix_hash_for_sequence(&seq, cached_tokens)
                {
                    let has_snapshot = {
                        let (pipeline, _) = self.get_pipeline(rank).unwrap();
                        pipeline.has_mamba_prefix_state(hash)?
                    };
                    if !has_snapshot {
                        self.scheduler
                            .block_engine
                            .invalidate_mamba_prefix_hash(hash);
                        self.fallback_sequence_to_full_prefill(
                            &seq,
                            seq_id,
                            cached_tokens,
                            &format!("missing mamba snapshot for hash {}", hash),
                        );
                        continue;
                    }

                    let restored = {
                        let (pipeline, _) = self.get_pipeline(rank).unwrap();
                        pipeline.restore_mamba_prefix_state(seq_id, hash)?
                    };
                    if !restored {
                        self.fallback_sequence_to_full_prefill(
                            &seq,
                            seq_id,
                            cached_tokens,
                            &format!("failed to restore mamba snapshot for hash {}", hash),
                        );
                    } else {
                        self.restored_prefix_sequences.write().insert(seq_id);
                    }
                } else {
                    tracing::warn!(
                        "Seq {} has {} cached tokens but no prefix hash; fallback to full prefill",
                        seq_id,
                        cached_tokens
                    );
                    self.fallback_sequence_to_full_prefill(
                        &seq,
                        seq_id,
                        cached_tokens,
                        "has no prefix hash",
                    );
                    continue;
                };
            }
        }

        Ok(())
    }

    fn free_finished_sequence_groups_and_sync_mamba(&mut self, rank: usize) {
        let mut captures = Vec::new();
        let released_ids = self
            .scheduler
            .free_finished_sequence_groups_with(|seq_id, hash| {
                if let Some(hash) = hash {
                    captures.push((seq_id, hash));
                }
            });

        let mut captured_hashes = Vec::new();
        {
            let pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            for (seq_id, hash) in captures {
                match pipeline.capture_mamba_prefix_state(seq_id, hash) {
                    Ok(true) => captured_hashes.push(hash),
                    Ok(false) => {}
                    Err(e) => {
                        tracing::warn!(
                            "Failed to capture mamba prefix state for seq {} hash {}: {}",
                            seq_id,
                            hash,
                            e
                        );
                    }
                }
            }
        }
        for hash in captured_hashes {
            self.scheduler.block_engine.record_mamba_prefix_hash(hash);
        }
        {
            let mut restored = self.restored_prefix_sequences.write();
            for seq_id in &released_ids {
                let _ = restored.remove(seq_id);
            }
        }
        {
            let pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            for seq_id in released_ids {
                pipeline.release_sequence_state(seq_id);
            }
        }
    }

    fn may_log_decoding_throughput(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        prompt_finish_times: &HashMap<usize, SystemTime>,
        _rank: usize,
    ) {
        #[cfg(feature = "nccl")]
        let do_log = DaemonManager::is_master_rank();
        #[cfg(not(feature = "nccl"))]
        let do_log = true;
        if !do_log || groups.is_empty() {
            return;
        }

        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        if cur_time.saturating_sub(self.last_decoding_throughput_log_ms) < 5000 {
            return;
        }
        self.last_decoding_throughput_log_ms = cur_time;

        let now = SystemTime::now();
        let mut active_seq_ids = Vec::new();
        let mut total_decoded_tokens = 0usize;
        let mut total_decoding_time_ms = 0usize;

        for group in groups {
            if group.is_finished() {
                continue;
            }
            let Some(prompt_finish_time) = prompt_finish_times.get(group.get_id()) else {
                continue;
            };
            let seq = Self::primary_sequence(group);
            let decoded_tokens = seq
                .deref()
                .get_len()
                .saturating_sub(seq.deref().get_prompt_len());
            if decoded_tokens == 0 {
                continue;
            }
            let decoding_time_ms = now
                .duration_since(*prompt_finish_time)
                .unwrap_or_default()
                .as_millis() as usize;
            if decoding_time_ms == 0 {
                continue;
            }
            total_decoded_tokens += decoded_tokens;
            total_decoding_time_ms += decoding_time_ms;
            active_seq_ids.push(seq.deref().get_id());
        }

        if active_seq_ids.is_empty() || total_decoding_time_ms < 1000 {
            return;
        }

        let avg_tps = total_decoded_tokens as f64 * 1000.0 / total_decoding_time_ms as f64;
        let total_tps = avg_tps * active_seq_ids.len() as f64;
        info!(
            "Decoding: {} active request(s) [Seq: {:?}], avg. {:.2} tokens/s per request (total: {:.2} tokens/s)",
            active_seq_ids.len(),
            active_seq_ids,
            avg_tps,
            total_tps,
        );
        self.scheduler.print_free_blocks();
    }

    async fn generate_parallel(
        engine: &Arc<RwLock<LLMEngine>>,
        ranks: Vec<usize>,
        multi_process: bool,
    ) -> Vec<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        #[cfg(feature = "nccl")]
        let iterator = ranks.par_iter();
        #[cfg(not(feature = "nccl"))]
        let iterator = ranks.iter();

        let tasks: Vec<_> = iterator
            .map(|rank| {
                let engine_clone = engine.clone();
                Self::generate_once(engine_clone, *rank, multi_process).unwrap()
            })
            .collect();
        tasks
    }

    #[cfg(all(feature = "cuda", feature = "graph"))]
    fn graph_capture_all_pipelines(&mut self) -> Result<()> {
        let mut ranks = self.pipelines.keys().copied().collect::<Vec<_>>();
        ranks.sort_unstable();
        for rank in ranks {
            let (pipeline, cache_engine) = self.get_mut_pipeline(rank).ok_or_else(|| {
                candle_core::Error::msg(format!("missing pipeline for rank {rank}"))
            })?;
            let device = pipeline.device();
            let _ = device.as_cuda_device().unwrap().bind_to_thread();
            pipeline.warmup_capture(Some(&cache_engine.get_kv_cache()))?;
        }
        self.restored_prefix_sequences.write().clear();
        Ok(())
    }

    pub fn new(
        mut pipelines: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)>,
        scheduler_config: SchedulerConfig,
        cache_config: &CacheConfig,
        config: &Config,
        notify: Arc<Notify>,
        holding_time: usize,
        num_shards: usize,
        multi_process: bool,
        #[cfg(feature = "nccl")] daemon_manager: Option<DaemonManager>,
        prefill_chunk_size: Option<usize>,
    ) -> Result<Arc<RwLock<Self>>> {
        const MIN_MAMBA_SLOT_CAPACITY: usize = 16;
        let devices: Vec<_> = pipelines
            .values()
            .map(|(pipeline, _)| pipeline.device())
            .collect();
        let model_dtype = pipelines
            .values()
            .next()
            .map(|(pipeline, _)| pipeline.dtype)
            .unwrap_or(cache_config.dtype);
        let hybrid_mamba_estimate =
            crate::estimate_hybrid_mamba_cache(config, model_dtype, num_shards);
        let require_mamba_prefix_snapshots = hybrid_mamba_estimate.is_some();
        let mut mamba_slot_capacity = scheduler_config.max_num_seqs.max(MIN_MAMBA_SLOT_CAPACITY);
        let mut mamba_prefix_capacity = 0usize;

        if let Some(estimate) = hybrid_mamba_estimate {
            let stride_blocks = crate::mamba_snapshot_block_stride_blocks();
            info!(
                "Hybrid mamba snapshot capture stride: {} block(s) ({} tokens), configured by {}",
                stride_blocks,
                stride_blocks.saturating_mul(cache_config.block_size),
                crate::MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
            );
            let reports = crate::query_device_memory_for_devices(&devices)?;
            let min_free_bytes = reports
                .iter()
                .map(|report| report.free_bytes)
                .min()
                .unwrap_or(0);
            for (rank, report) in reports.iter().enumerate() {
                info!(
                    "Rank {} GPU memory after KV cache allocation: total {:.2} GB, free {:.2} GB, used {:.2} GB",
                    rank,
                    report.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                    report.free_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                    report.used_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                );
            }

            if cache_config.mamba_cache_budget_bytes > 0 {
                let planned_total_slots =
                    cache_config.mamba_cache_budget_bytes / estimate.slot_bytes;
                if planned_total_slots >= mamba_slot_capacity {
                    mamba_prefix_capacity = if scheduler_config.prefix_cache.enabled {
                        planned_total_slots.saturating_sub(mamba_slot_capacity)
                    } else {
                        0
                    };
                } else if planned_total_slots > 0 {
                    warn!(
                        "Hybrid mamba planned budget only fits {} total slot(s), below the target active minimum {}; using the largest safe active capacity instead.",
                        planned_total_slots,
                        mamba_slot_capacity
                    );
                    mamba_slot_capacity = planned_total_slots;
                    mamba_prefix_capacity = 0;
                } else {
                    warn!(
                        "Hybrid mamba planned budget is smaller than one slot; falling back to 1 active slot."
                    );
                    mamba_slot_capacity = 1;
                    mamba_prefix_capacity = 0;
                }

                let active_bytes = mamba_slot_capacity.saturating_mul(estimate.slot_bytes);
                let prefix_budget_bytes = cache_config
                    .mamba_cache_budget_bytes
                    .saturating_sub(active_bytes);

                info!(
                    "Hybrid mamba cache sizing: {} active slot(s), {} prefix snapshot slot(s), {:.2} MB/slot, {} GDN layer(s), planned mamba budget {:.2} GB, min free GPU memory after KV {:.2} GB",
                    mamba_slot_capacity,
                    mamba_prefix_capacity,
                    estimate.slot_bytes as f64 / 1024.0 / 1024.0,
                    estimate.num_gdn_layers,
                    cache_config.mamba_cache_budget_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                    min_free_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                );
                info!(
                    "Hybrid mamba memory split: active {:.2} GB, prefix snapshots {:.2} GB",
                    active_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                    prefix_budget_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                );
            } else {
                warn!(
                    "Hybrid mamba budget was not reserved in the combined cache budget; using fallback active slot capacity {}.",
                    mamba_slot_capacity
                );
            }
        }

        for (pipeline, _) in pipelines.values_mut() {
            pipeline.preallocate_mamba_cache(mamba_slot_capacity)?;
            pipeline.set_mamba_prefix_cache_capacity(mamba_prefix_capacity);
        }

        let num_threads: usize = pipelines.len();
        let engine = Arc::new(RwLock::new(Self {
            pipelines,
            scheduler: Scheduler::new(
                scheduler_config,
                cache_config,
                require_mamba_prefix_snapshots,
            ),
            seq_id: 0,
            cache_config: cache_config.clone(),
            config: config.clone(),
            group_id: 0,
            notify: notify.clone(),
            completion_records: HashMap::new(),
            sequence_groups: RwLock::new(VecDeque::new()),
            multi_process,
            num_shards,
            waiting_tasks: RwLock::new(Vec::<TaskData>::new()),
            #[cfg(feature = "nccl")]
            daemon_manager: RwLock::new(daemon_manager),
            sync_notifies: HashMap::new(),
            senders: HashMap::new(),
            prefill_chunk_size,
            restored_prefix_sequences: RwLock::new(HashSet::new()),
            exit_flag: Arc::new(AtomicBool::new(false)),
            last_decoding_throughput_log_ms: 0,
        }));
        let engine_clone = engine.clone();

        let mut ranks = Vec::<usize>::new();
        for rank in 0..num_threads {
            ranks.push(rank);
        }

        #[cfg(feature = "nccl")]
        let is_master_rank = DaemonManager::is_master_rank();
        #[cfg(not(feature = "nccl"))]
        let is_master_rank = true;
        #[cfg(all(feature = "cuda", feature = "graph"))]
        let (graph_capture_tx, graph_capture_rx) = std::sync::mpsc::sync_channel(1);

        let _ = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                #[cfg(all(feature = "cuda", feature = "graph"))]
                {
                    let graph_capture_result = {
                        let mut e = engine.write();
                        e.graph_capture_all_pipelines()
                    };
                    let _ = graph_capture_tx.send(
                        graph_capture_result
                            .as_ref()
                            .map(|_| ())
                            .map_err(|e| format!("{e:?}")),
                    );
                    if graph_capture_result.is_err() {
                        return;
                    }
                }
                loop {
                    if engine.read().exit_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    if is_master_rank {
                        notify.notified().await;
                        if engine.read().exit_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        let _ = tokio::time::sleep(tokio::time::Duration::from_millis(holding_time as u64)).await;
                    }
                    {
                        let mut e = engine.write();
                        if e.sync_waiting_task_to_group() {
                            let _ = tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                            continue;
                        }
                    }

                    let results = Self::generate_parallel(&engine, ranks.clone(), multi_process).await;

                    #[cfg(feature = "nccl")]
                    if multi_process && !is_master_rank {
                        continue;
                    }
                    let result = &results[0];
                    if results.is_empty() || result.is_empty() {
                        continue;
                    }

                    //chat completion statistics
                    let overall_usage = ChatCompletionUsageResponse {
                        request_id: "".to_string(),
                        created: 0,
                        completion_tokens: result.values()
                            .map(|(_, usage)| usage.completion_tokens)
                            .sum(),
                        prompt_tokens: result.values().map(|(_, usage)| usage.prompt_tokens).sum(),
                        total_tokens: result.values().map(|(_, usage)| usage.total_tokens).sum(),
                        prompt_time_costs: result
                            .values()
                            .map(|(_, usage)| usage.prompt_time_costs)
                            .max()
                            .unwrap_or(0),
                        completion_time_costs: result
                            .values()
                            .map(|(_, usage)| usage.completion_time_costs)
                            .max()
                            .unwrap_or(0),
                    };

                    let prompt_tps : f32 = result.values().map(|(_, usage)| {
                        //time costs in milliseconds
                        usage.prompt_tokens as f32  * 1000f32 / f32::max(usage.prompt_time_costs as f32, 1f32)
                    }).sum::<f32>() / result.len() as f32;

                    let decode_tps : f32 = result.values().map(|(_, usage)| {
                        //time costs in milliseconds
                        usage.completion_tokens as f32  * 1000f32 / f32::max(usage.completion_time_costs as f32, 1f32)
                    }).sum::<f32>() / result.len() as f32;

                    println!(
                        "\r\n [{} requests] Prefilling: {} prompt tokens processed (avg tps {:.02} tokens/s, throughput {:.02} tokens/s)",
                        result.len(),
                        overall_usage.prompt_tokens,
                        prompt_tps,
                        prompt_tps * result.len() as f32,
                    );
                    println!(
                        "\r\n [{} requests] Decoding: {} tokens processed (avg tps {:.02} tokens/s, throughput {:.02} tokens/s)",
                        result.len(),
                        overall_usage.completion_tokens,
                        decode_tps,
                        decode_tps * result.len() as f32,
                    );
                }
            });
        });

        #[cfg(all(feature = "cuda", feature = "graph"))]
        match graph_capture_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => candle_core::bail!("Unable to capture cuda graph {err}!"),
            Err(err) => candle_core::bail!(
                "Failed to receive cuda graph warmup result from engine worker thread: {}",
                err
            ),
        }

        Ok(engine_clone)
    }

    #[allow(unused_mut, unused_variables)]
    pub fn sync_waiting_task_to_group(&mut self) -> bool {
        let mut continue_loop = false;
        #[cfg(feature = "nccl")]
        let is_master_rank = DaemonManager::is_master_rank();
        #[cfg(not(feature = "nccl"))]
        let is_master_rank = true;

        #[cfg(feature = "nccl")]
        if self.multi_process && !is_master_rank {
            debug!("daemon process sync task!");
            let message = {
                let mut daemon_manager = self.daemon_manager.write();
                daemon_manager.as_mut().unwrap().receive_message()
            };
            match message {
                Ok(MessageType::Continue) | Ok(MessageType::Sample(_)) => {
                    debug!("A start/continue/sample message*****!");
                    continue_loop = true;
                }
                Ok(MessageType::Data(data)) => {
                    debug!("A data message*****!");
                    for task in data {
                        let seq_group = self.create_sequence_group(
                            task.seq_id,
                            task.group_id,
                            &task.prompt,
                            &task.request_id,
                            task.created,
                            &task.sampling_params,
                            task.use_logprobs,
                            task.is_embedding,
                            task.encoding_format,
                            task.embedding_type,
                            task.tools.clone(),
                            None,
                        );
                        tracing::debug!("Daemon process: add_sequence to group {}", task.group_id);
                        self.scheduler.add_sequence(seq_group);
                    }
                }
                Ok(MessageType::Abort(_)) | Ok(MessageType::Finish) | Ok(MessageType::Close) => {
                    warn!("A abort/finish or close message!");
                    continue_loop = true;
                }
                _ => {
                    warn!("Invalid message, perhaps the main process is exited!");
                    panic!("Exit process");
                }
            };
        }

        if is_master_rank {
            let (send_tasks, num_send_tasks) = {
                let waiting_tasks = self.waiting_tasks.write();
                let send_tasks = waiting_tasks.clone();
                let num_send_tasks = send_tasks.len();
                (send_tasks, num_send_tasks)
            };

            for task in &send_tasks {
                let sender: Option<Sender<ChatResponse>> = self
                    .senders
                    .get(&task.request_id)
                    .and_then(|opt_arc_sender| {
                        opt_arc_sender.as_ref().map(|arc| arc.as_ref().clone())
                    });
                let seq_group = self.create_sequence_group(
                    task.seq_id,
                    task.group_id,
                    &task.prompt,
                    &task.request_id,
                    task.created,
                    &task.sampling_params,
                    task.use_logprobs,
                    task.is_embedding,
                    task.encoding_format.clone(),
                    task.embedding_type.clone(),
                    task.tools.clone(),
                    sender,
                );
                tracing::debug!("Main process: add_sequence to group {}", task.group_id);
                self.scheduler.add_sequence(seq_group);
            }

            #[cfg(feature = "nccl")]
            if self.multi_process {
                let mut daemon_manager = self.daemon_manager.write();
                if num_send_tasks > 0 {
                    if self.num_shards > 1 {
                        warn!(
                            "Sending {} tasks to {} subprocesses",
                            num_send_tasks,
                            self.num_shards - 1
                        );
                    }
                    let mut ipc_tasks = send_tasks.clone();
                    for task in &mut ipc_tasks {
                        task.tools.clear();
                    }
                    let _ = daemon_manager
                        .as_mut()
                        .unwrap()
                        .send_message(&MessageType::Data(ipc_tasks));
                } else {
                    let _ = daemon_manager
                        .as_mut()
                        .unwrap()
                        .send_message(&MessageType::Continue);
                    continue_loop = true;
                }
            }

            {
                let mut waiting_tasks = self.waiting_tasks.write();
                waiting_tasks.clear();
            }
        }
        continue_loop
    }

    pub fn get_pipeline(&self, rank: usize) -> Option<&(Box<DefaultPipeline>, CacheEngine)> {
        self.pipelines.get(&rank)
    }

    pub fn get_mut_pipeline(
        &mut self,
        rank: usize,
    ) -> Option<&mut (Box<DefaultPipeline>, CacheEngine)> {
        self.pipelines.get_mut(&rank)
    }

    #[cfg(feature = "flashinfer")]
    fn flashinfer_kv_params_for_rank(&self, rank: usize) -> Result<Option<FlashInferKvParams>> {
        let (_, cache_engine) = self
            .get_pipeline(rank)
            .ok_or_else(|| candle_core::Error::msg(format!("missing pipeline for rank {rank}")))?;
        let kv_cache = cache_engine.get_kv_cache();
        if kv_cache.is_empty() {
            return Ok(None);
        }

        let (k_cache, _) = &kv_cache[0];
        if k_cache.dtype() == candle_core::DType::U8 {
            return Ok(None);
        }

        let (_, page_size, num_kv_heads, head_dim) = k_cache.dims4()?;
        Ok(Some(FlashInferKvParams {
            kv_dtype: k_cache.dtype(),
            out_dtype: self
                .get_pipeline(rank)
                .map(|(pipeline, _)| pipeline.dtype)
                .unwrap_or(k_cache.dtype()),
            page_size,
            num_kv_heads,
            head_dim,
            num_qo_heads: self.config.num_attention_heads / self.num_shards,
        }))
    }

    #[cfg(feature = "flashinfer")]
    fn ensure_flashinfer_decode_plan(
        &self,
        rank: usize,
        device: &Device,
        input_batch: usize,
        metadata: &mut InputMetadata,
    ) -> Result<()> {
        let Some(fm) = metadata.flashinfer_metadata.as_mut() else {
            return Ok(());
        };
        if fm.decode_plan_info.is_some() {
            return Ok(());
        }
        let Some(params) = self.flashinfer_kv_params_for_rank(rank)? else {
            return Ok(());
        };
        fm.decode_plan_info = Some(attention_rs::flashinfer::decode_plan(
            device,
            params.kv_dtype,
            params.out_dtype,
            &fm.indptr_host,
            fm.last_len_host.as_deref(),
            fm.kv_len_arr_host.as_deref(),
            input_batch,
            params.num_qo_heads,
            params.num_kv_heads,
            params.head_dim,
            params.page_size,
            fm.use_cuda_graph,
        )?);
        Ok(())
    }

    fn get_stream_response(
        &self,
        request_id: String,
        created: u64,
        content: Option<String>,
        tool_calls: Option<Vec<crate::tools::ToolCall>>,
        finish_reason: Option<String>,
        pipeline: &DefaultPipeline,
    ) -> ChatCompletionChunk {
        let mut choices = Vec::new();
        let choice = Choice {
            delta: ChoiceData {
                role: "assistant".to_string(),
                content,
                tool_calls,
            },
            finish_reason,
            index: 0,
        };
        choices.push(choice);

        ChatCompletionChunk {
            id: request_id,
            choices,
            created,
            model: pipeline.name().to_string(),
            object: "chat.completion.chunk",
            system_fingerprint: None,
        }
    }

    fn should_finish_tool_calls_for_token(
        &self,
        rank: usize,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
        token: u32,
    ) -> bool {
        if group.sampling_params.mcp_mode.is_none() {
            return false;
        }

        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        if pipeline.tool_call_end_token_ids.contains(&token) {
            return if pipeline.tool_call_start_token_ids.is_empty() {
                true
            } else {
                seq.deref()
                    .get_output_tokens()
                    .iter()
                    .any(|output| pipeline.tool_call_start_token_ids.contains(&output.token))
            };
        }

        if pipeline.json_end_token_id == Some(token) {
            let mut output_tokens: Vec<u32> = seq
                .deref()
                .get_output_tokens()
                .iter()
                .map(|output| output.token)
                .collect();
            output_tokens.push(token);
            if let Ok(decoded) = pipeline.tokenizer().decode(&output_tokens, true) {
                return pipeline.tool_call_regex.is_match(&decoded);
            }
        }

        false
    }

    fn collect_stream_emission_for_token(
        &self,
        rank: usize,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
        logprobs: &Logprobs,
        should_finish_tool_calls: bool,
    ) -> StreamEmission {
        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        let token_str = &logprobs.bytes;
        let should_parse_tools = group.sampling_params.mcp_mode.is_some();
        let mut content = None;
        let mut tool_calls = None;

        {
            let outer = seq.deref();
            let mut data = outer.deref_mut();

            if should_parse_tools {
                if data.stream_tool_parser.is_none() {
                    data.stream_tool_parser = Some(StreamToolParser::new_with_config(
                        &pipeline.tool_model_type,
                        pipeline.tool_parser_model_id.clone(),
                        pipeline.tool_config.clone(),
                        group.tools.clone(),
                        pipeline.enforce_parser.clone(),
                    ));
                }
                if let Some(parser) = data.stream_tool_parser.as_mut() {
                    match parser.process_token(logprobs.token, token_str) {
                        StreamResult::Content(text) | StreamResult::FlushBuffer(text) => {
                            if !text.is_empty() {
                                content = Some(text);
                            }
                        }
                        StreamResult::Buffering => {}
                        StreamResult::ToolCalls(calls) => {
                            data.pending_tool_calls.extend(calls);
                        }
                    }
                }

                if should_finish_tool_calls {
                    if let Some(parser) = data.stream_tool_parser.as_mut() {
                        if matches!(parser.state(), ParserState::Buffering) {
                            match parser.finalize_buffered_tool_calls() {
                                Some(BufferedFinalizeResult::ToolCalls(mut parsed)) => {
                                    data.pending_tool_calls.append(&mut parsed);
                                }
                                Some(BufferedFinalizeResult::FlushBuffer(buffer)) => {
                                    if !buffer.is_empty() {
                                        content = Some(buffer);
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                    let pending = std::mem::take(&mut data.pending_tool_calls);
                    if !pending.is_empty() {
                        tool_calls = Some(pending);
                        content = None;
                    }
                }
            } else if !token_str.is_empty() {
                content = Some(token_str.clone());
            }
        }

        StreamEmission {
            content,
            tool_calls,
        }
    }

    fn collect_stream_emission_on_finish(
        &self,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
    ) -> StreamEmission {
        let mut content = None;
        let mut tool_calls = None;

        {
            let outer = seq.deref();
            let mut data = outer.deref_mut();
            let should_parse_tools = group.sampling_params.mcp_mode.is_some();

            if should_parse_tools {
                if let Some(parser) = data.stream_tool_parser.as_mut() {
                    if matches!(parser.state(), ParserState::Buffering) {
                        match parser.finalize_buffered_tool_calls() {
                            Some(BufferedFinalizeResult::ToolCalls(mut parsed)) => {
                                data.pending_tool_calls.append(&mut parsed);
                            }
                            Some(BufferedFinalizeResult::FlushBuffer(buffer)) => {
                                if !buffer.is_empty() {
                                    content = Some(buffer);
                                }
                            }
                            None => {}
                        }
                    }
                }
            }

            let pending = std::mem::take(&mut data.pending_tool_calls);
            if !pending.is_empty() {
                tool_calls = Some(pending);
            }
        }

        StreamEmission {
            content,
            tool_calls,
        }
    }

    fn send_stream_emission(
        &self,
        rank: usize,
        sender: &Sender<ChatResponse>,
        group: &Arc<SequenceGroup>,
        seq: &Arc<Sequence>,
        emission: StreamEmission,
        finish_reason: Option<String>,
    ) {
        let should_log_free_blocks =
            finish_reason.is_none() && emission.tool_calls.is_none() && emission.content.is_some();
        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        let chunk = if let Some(tool_calls) = emission.tool_calls {
            self.get_stream_response(
                group.request_id.clone(),
                get_created_time_secs(),
                None,
                Some(
                    tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(idx, call)| call.with_index(idx))
                        .collect(),
                ),
                Some("tool_calls".to_string()),
                pipeline,
            )
        } else {
            self.get_stream_response(
                group.request_id.clone(),
                group.arrival_time,
                emission.content,
                None,
                finish_reason,
                pipeline,
            )
        };

        if matches!(
            chunk.choices.first(),
            Some(choice) if choice.delta.tool_calls.is_some()
        ) {
            tracing::info!("Sending final tool call chunk: {:?}", chunk);
        }

        let ret = sender.send(ChatResponse::Chunk(chunk));
        if ret.is_err() {
            warn!(
                "Send stream response error! (sequence id {})",
                seq.deref().get_id()
            );
            seq.deref_mut().set_finish_reason("abort".to_string());
        } else if should_log_free_blocks && seq.deref_mut().get_len() % 1000 == 0 {
            self.scheduler.print_free_blocks();
        }
    }

    #[cfg(feature = "nccl")]
    pub fn sync_abort_sequences(
        &self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        aborted_sequences: Vec<usize>,
    ) {
        if DaemonManager::is_master_rank() {
            if aborted_sequences.len() > 0 {
                warn!(
                    "Sending abort message ({} sequence(s)) to subprocesses!",
                    aborted_sequences.len()
                );
                {
                    warn!("engine.write write for aborted_sequences");
                    let mut daemon_manager = self.daemon_manager.write();
                    let _ = daemon_manager
                        .as_mut()
                        .unwrap()
                        .send_message(&MessageType::Abort(aborted_sequences));
                }
            } else {
                let mut daemon_manager = self.daemon_manager.write();
                let _ = daemon_manager
                    .as_mut()
                    .unwrap()
                    .send_message(&MessageType::Continue);
            }
        } else {
            let message = {
                let mut daemon_manager = self.daemon_manager.write();
                daemon_manager.as_mut().unwrap().receive_message()
            };
            match message {
                Ok(MessageType::Abort(ids)) => {
                    for group in scheduled.iter() {
                        let seq = Self::primary_sequence(group);
                        if ids.contains(&seq.deref().get_id()) {
                            seq.deref_mut().set_finish_reason("abort".to_string());
                            warn!("abort sequence ({}) in subprocess!", seq.deref().get_id());
                        }
                    }
                }
                Ok(MessageType::Finish) | Ok(MessageType::Close) => {
                    warn!("A abort/finish or close message!");
                    for group in scheduled.iter() {
                        let seq = Self::primary_sequence(group);
                        seq.deref_mut().set_finish_reason("abort".to_string());
                        warn!(
                            "abort/finish sequence ({}) in subprocess!",
                            seq.deref().get_id()
                        );
                    }
                }
                Ok(MessageType::Continue) | Ok(MessageType::Sample(_)) => {
                    info!("other message!");
                }
                Ok(MessageType::Data(_)) => {
                    warn!("data message found!");
                }
                _ => {
                    warn!("invalid message!");
                    panic!("Exit process")
                }
            };
        }
    }

    pub fn generate_once(
        engine: Arc<RwLock<Self>>,
        rank: usize,
        multi_process: bool,
    ) -> Result<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        let mut responses =
            HashMap::<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>::new();
        let mut prompt_finish_times = HashMap::<usize, SystemTime>::new();
        #[cfg(feature = "cuda")]
        {
            debug!("Start processing...");
            let e = engine.read();
            let (pipeline, _) = e.get_pipeline(rank).unwrap();
            let device = pipeline.device();
            let _ = device.as_cuda_device().unwrap().bind_to_thread();
        }
        loop {
            {
                if !multi_process {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                let e = engine.read();
                if !e.scheduler.has_unfinished_sequences() {
                    break;
                }
            }
            if rank == 0 {
                //only the first rank thread perform task scheduling
                let mut e = engine.write();
                let scheduler_outputs = e.scheduler.schedule();
                if !scheduler_outputs.ignored_seq_groups.is_empty() {
                    for group in scheduler_outputs.ignored_seq_groups.iter() {
                        if let Some(sender) = &group.sender {
                            let _ = sender.send(ChatResponse::ModelError(
                                candle_core::Error::msg(
                                    "Ignored sequence group: allocation impossible",
                                )
                                .to_string(),
                            ));
                        }
                    }
                }
                e.execute_scheduler_ops(&scheduler_outputs, 0).unwrap();
                let mut groups = e.sequence_groups.write();

                *groups = match Arc::try_unwrap(scheduler_outputs.scheduled) {
                    Ok(deq) => deq,
                    Err(arc_deq) => (*arc_deq).clone(),
                };
            };

            let mut scheduled: VecDeque<Arc<SequenceGroup>> = {
                let e = engine.read();
                let x = e.sequence_groups.read();
                x.clone()
            };
            if scheduled.is_empty() {
                continue; //data not ready
            }

            let is_embedding = scheduled[0].is_embedding;
            let is_prompt_request = Self::primary_sequence(&scheduled[0]).deref().is_prompt();

            if is_prompt_request {
                let mut e = engine.write();
                e.restore_mamba_prefix_states_for_prompt(&scheduled, rank)?;
            }
            //run partial models in parallel
            let (mut logits, is_prompt, model_name) = {
                let e = engine.read();
                let (pipeline, cache_engine) = e.get_pipeline(rank).unwrap();
                let device = pipeline.device();
                let model_name = pipeline.name().to_string();
                #[cfg_attr(not(feature = "flashinfer"), allow(unused_mut))]
                let mut prepared = if is_prompt_request {
                    e.prepare_prompt(&scheduled, device, rank)
                } else {
                    e.prepare_decode(&scheduled, device, rank)
                }?;
                #[cfg(feature = "flashinfer")]
                if !prepared.metadata.is_prefill {
                    let use_cuda_graph = prepared
                        .metadata
                        .flashinfer_metadata
                        .as_ref()
                        .map(|fm| fm.use_cuda_graph)
                        .unwrap_or(false);
                    if !use_cuda_graph {
                        e.ensure_flashinfer_decode_plan(
                            rank,
                            device,
                            prepared.tokens.dim(0)?,
                            &mut prepared.metadata,
                        )?;
                    }
                }
                let PreparedInputs {
                    tokens,
                    positions,
                    metadata,
                } = prepared;

                let x = if is_embedding {
                    pipeline.forward_embedding(
                        tokens,
                        &positions,
                        Some(&cache_engine.get_kv_cache()),
                        &metadata,
                    )?
                } else {
                    pipeline.forward(
                        tokens,
                        &positions,
                        Some(&cache_engine.get_kv_cache()),
                        &metadata,
                    )?
                };

                (x, metadata.is_prefill, model_name)
            };

            if is_embedding {
                if is_prompt {
                    let mut e = engine.write();
                    //Process embedding response
                    let mut start_idx = 0;
                    for group in &scheduled {
                        let seq = Self::primary_sequence(group);
                        let prompt_len = seq.deref().get_prompt_len();
                        let end_idx = start_idx + prompt_len;

                        //extract sequence embedding
                        let seq_embedding = logits.narrow(0, start_idx, prompt_len)?;

                        //Pooling
                        let pooled_embedding = match group.embedding_type {
                            crate::openai::requests::EmbeddingType::Last => {
                                seq_embedding.narrow(0, prompt_len - 1, 1)?.squeeze(0)?
                            }
                            crate::openai::requests::EmbeddingType::Mean => {
                                seq_embedding.mean(0)?
                            }
                        };
                        info!("Resulting embedding shape: {:?}", pooled_embedding.shape());

                        let vec_embedding = pooled_embedding
                            .to_dtype(candle_core::DType::F32)?
                            .to_vec1::<f32>()?;

                        let output = match group.encoding_format {
                            crate::openai::requests::EncodingFormat::Float => {
                                EmbeddingOutput::Vector(vec_embedding)
                            }
                            crate::openai::requests::EncodingFormat::Base64 => {
                                use base64::{engine::general_purpose::STANDARD, Engine as _};
                                let bytes = unsafe {
                                    std::slice::from_raw_parts(
                                        vec_embedding.as_ptr() as *const u8,
                                        vec_embedding.len() * 4,
                                    )
                                };
                                EmbeddingOutput::Base64(STANDARD.encode(bytes))
                            }
                        };

                        if let Some(sender) = &group.sender {
                            let response = EmbeddingResponse {
                                object: "list",
                                data: vec![EmbeddingData {
                                    object: "embedding",
                                    embedding: output,
                                    index: 0,
                                }],
                                model: model_name.clone(),
                                usage: EmbeddingUsage {
                                    prompt_tokens: prompt_len,
                                    total_tokens: prompt_len,
                                },
                            };
                            let _ = sender.send(ChatResponse::Embedding(response));
                        } else {
                            tracing::error!("No sender for embedding group!");
                        }
                        seq.deref_mut().set_finish_reason("stop".to_string());
                        start_idx = end_idx;
                    }

                    e.free_finished_sequence_groups_and_sync_mamba(rank);
                }
                continue;
            }

            if is_prompt {
                let mut e = engine.write();
                let prefill_chunk_size = e.prefill_chunk_size.unwrap_or(PREFILL_CHUNK_SIZE);

                if prefill_chunk_size > 0 {
                    e.capture_mamba_prefix_states_for_prefill_progress(
                        &scheduled,
                        prefill_chunk_size,
                        rank,
                    )?;
                    let (finished_indices, finished_groups) = e
                        .scheduler
                        .filter_prefill_finished(&scheduled, prefill_chunk_size);

                    if finished_indices.is_empty() {
                        continue;
                    }
                    scheduled = finished_groups;
                    let batch = finished_indices.len();
                    logits = logits.index_select(
                        &Tensor::from_vec(finished_indices, (batch,), logits.device())?,
                        0,
                    )?;
                }
            }

            #[cfg(feature = "nccl")]
            let do_sample = if rank == 0 && DaemonManager::is_master_rank() {
                true
            } else {
                false
            };
            #[cfg(not(feature = "nccl"))]
            let do_sample = rank == 0;

            let optional_results = if do_sample {
                let sample = {
                    //only the first rank thread perform sampling
                    let mut e = engine.write();
                    let default_pipeline = e.get_mut_pipeline(0usize).unwrap().0.as_mut();
                    default_pipeline.sample(&logits, &scheduled).unwrap()
                };

                #[cfg(feature = "nccl")]
                if multi_process {
                    let e = engine.read();
                    let mut daemon_manager = e.daemon_manager.write();
                    let mut logprobs: Vec<TaskSampleData> = Vec::new();
                    for (s, group) in sample.iter().zip(scheduled.iter()) {
                        let seq_id = Self::primary_sequence_id(group);
                        match s {
                            Either::Left(logprobs_data) => logprobs.push(TaskSampleData::Token {
                                seq_id,
                                logprobs: logprobs_data.clone(),
                            }),
                            Either::Right(reason) => logprobs.push(TaskSampleData::StopReason {
                                seq_id,
                                reason: reason.clone(),
                            }),
                        };
                    }
                    let _ = daemon_manager
                        .as_mut()
                        .unwrap()
                        .send_message(&MessageType::Sample(logprobs));
                }
                Some(sample)
            } else {
                #[cfg(feature = "nccl")]
                if multi_process && !DaemonManager::is_master_rank() {
                    let _ = logits.to_device(&Device::Cpu).unwrap(); //sync
                    let message = {
                        let e = engine.read();
                        let mut daemon_manager = e.daemon_manager.write();
                        daemon_manager.as_mut().unwrap().receive_message()
                    };
                    match message {
                        Ok(MessageType::Sample(data)) => {
                            let mut by_seq_id = HashMap::<usize, TokenOrFinishReason>::new();
                            for s in data {
                                match s {
                                    TaskSampleData::Token { seq_id, logprobs } => {
                                        by_seq_id
                                            .insert(seq_id, TokenOrFinishReason::Left(logprobs));
                                    }
                                    TaskSampleData::StopReason { seq_id, reason } => {
                                        by_seq_id
                                            .insert(seq_id, TokenOrFinishReason::Right(reason));
                                    }
                                }
                            }
                            let mut ordered =
                                Vec::<TokenOrFinishReason>::with_capacity(scheduled.len());
                            for group in &scheduled {
                                let seq_id = Self::primary_sequence_id(group);
                                if let Some(sample) = by_seq_id.remove(&seq_id) {
                                    ordered.push(sample);
                                } else {
                                    candle_core::bail!(
                                        "Missing sampled token for seq_id {} on daemon rank {}",
                                        seq_id,
                                        rank
                                    );
                                }
                            }
                            if !by_seq_id.is_empty() {
                                warn!(
                                    "Received {} extra sampled entries on daemon rank {}",
                                    by_seq_id.len(),
                                    rank
                                );
                            }
                            debug!("generate_once: received sample");
                            Some(ordered)
                        }
                        _ => {
                            info!("generate_once: received empty sample");
                            break;
                        }
                    }
                } else {
                    None
                }
                #[cfg(not(feature = "nccl"))]
                None
            };

            {
                let e = engine.read();
                let mut cur_group = e.sequence_groups.write();
                cur_group.clear();
            }

            if optional_results.is_none() {
                continue;
            }

            //only the first rank thread perform stream response
            let results = optional_results.unwrap();
            if results.len() != scheduled.len() {
                candle_core::bail!(
                    "Sample result and scheduled group length mismatch on rank {}: {} vs {}",
                    rank,
                    results.len(),
                    scheduled.len()
                );
            }
            for (result_, group) in zip(results, &scheduled) {
                match result_ {
                    Either::Left(logprobs) => {
                        let seq = Self::primary_sequence(group);
                        let should_finish_tool_calls = {
                            let e = engine.read();
                            e.should_finish_tool_calls_for_token(rank, group, &seq, logprobs.token)
                        };
                        if seq.deref().is_prompt() {
                            let e = engine.read();
                            e.scheduler.print_free_blocks();
                            let prompt_finish_time = SystemTime::now();
                            prompt_finish_times.insert(*group.get_id(), prompt_finish_time);

                            #[cfg(feature = "nccl")]
                            let do_log = DaemonManager::is_master_rank();
                            #[cfg(not(feature = "nccl"))]
                            let do_log = true;
                            if do_log {
                                let prompt_time_costs = prompt_finish_time
                                    .duration_since(group.created_time)
                                    .unwrap()
                                    .as_millis();
                                if prompt_time_costs > 0 {
                                    warn!(
                                        "Prefilling {} tokens finished in {} seconds ({} tokens/s) ({})",
                                        seq.deref().get_prompt_len(),
                                        prompt_time_costs / 1000,
                                        seq.deref().get_prompt_len() * 1000
                                            / prompt_time_costs as usize,
                                        group.request_id,
                                    );
                                }
                            }
                        }
                        if let Some(sender) = &group.sender {
                            let e = engine.read();
                            let emission = e.collect_stream_emission_for_token(
                                rank,
                                group,
                                &seq,
                                &logprobs,
                                should_finish_tool_calls,
                            );
                            if emission.tool_calls.is_some() || emission.content.is_some() {
                                e.send_stream_emission(rank, sender, group, &seq, emission, None);
                            }
                        };
                        seq.deref_mut().add_token(logprobs);
                        if should_finish_tool_calls {
                            seq.deref_mut().set_finish_reason("tool_calls".to_string());
                        }
                    }
                    Either::Right(finish_reason) => {
                        let seq = Self::primary_sequence(group);
                        if let Some(sender) = &group.sender {
                            let e = engine.read();
                            let emission = e.collect_stream_emission_on_finish(group, &seq);
                            let stream_finish_reason = if finish_reason == "tool_calls" {
                                "stop".to_string()
                            } else {
                                finish_reason.clone()
                            };
                            e.send_stream_emission(
                                rank,
                                sender,
                                group,
                                &seq,
                                emission,
                                Some(stream_finish_reason),
                            );
                        };
                        seq.deref_mut().set_finish_reason(finish_reason);
                    }
                }
            }

            {
                let mut e = engine.write();
                e.capture_mamba_prefix_states_for_decode_progress(&scheduled, rank)?;
                e.free_finished_sequence_groups_and_sync_mamba(rank);
                if !is_prompt {
                    e.may_log_decoding_throughput(&scheduled, &prompt_finish_times, rank);
                }
            }

            let mut aborted_sequences: Vec<usize> = Vec::new();
            for group in scheduled.iter() {
                if group.is_finished() && !responses.contains_key(&group.request_id) {
                    let end_time = SystemTime::now();
                    let prompt_finish_time = prompt_finish_times
                        .get(group.get_id())
                        .cloned()
                        .unwrap_or_else(|| {
                            warn!(
                                "Missing prompt_finish_time for finished request {} (group id {}), using created_time fallback",
                                group.request_id,
                                group.get_id()
                            );
                            group.created_time
                        });
                    let completion_time_costs = end_time
                        .duration_since(prompt_finish_time)
                        .unwrap()
                        .as_millis();
                    let seq = Self::primary_sequence(group);
                    let decoded_tokens = seq.deref().get_len() - seq.deref().get_prompt_len();
                    #[cfg(feature = "nccl")]
                    let do_log = DaemonManager::is_master_rank();
                    #[cfg(not(feature = "nccl"))]
                    let do_log = true;
                    if do_log {
                        warn!(
                            "Decoding {} tokens finished in {} seconds ({})",
                            decoded_tokens,
                            completion_time_costs / 1000,
                            group.request_id,
                        );
                    }
                    // Create choices from the group
                    let mut seqs = group.get_seqs().values().collect::<Vec<_>>();
                    seqs.sort_by(|seq_a, seq_b| {
                        let logprob_cmp = seq_b
                            .deref_mut()
                            .get_cumulative_logprob()
                            .partial_cmp(&seq_a.deref_mut().get_cumulative_logprob())
                            .unwrap_or(std::cmp::Ordering::Equal);
                        if logprob_cmp == std::cmp::Ordering::Equal {
                            seq_a.deref().get_id().cmp(&seq_b.deref().get_id())
                        } else {
                            logprob_cmp
                        }
                    });
                    let top_n = seqs.get(0..group.sampling_params.n).unwrap();

                    let mut choices = Vec::new();

                    let do_sync_response = {
                        #[cfg(feature = "nccl")]
                        if multi_process && !DaemonManager::is_master_rank() {
                            false
                        } else {
                            group.sender.is_none()
                        }
                        #[cfg(not(feature = "nccl"))]
                        group.sender.is_none()
                    };

                    if do_sync_response {
                        let e = engine.read();
                        for (index, seq) in top_n.iter().enumerate() {
                            let outputs = seq.deref_mut().get_output_tokens();
                            let pipeline = e.get_pipeline(0usize).unwrap().0.as_ref();
                            let should_parse_tools = group.sampling_params.mcp_mode.is_some();
                            let mut finish_reason = seq.deref_mut().get_finish_reason().clone();

                            let (content, tool_calls) = if should_parse_tools {
                                let mut parser = StreamToolParser::new_with_config(
                                    &pipeline.tool_model_type,
                                    pipeline.tool_parser_model_id.clone(),
                                    pipeline.tool_config.clone(),
                                    group.tools.clone(),
                                    pipeline.enforce_parser.clone(),
                                );
                                let mut pending_tool_calls = Vec::new();
                                let mut accumulated = String::new();

                                for output in &outputs {
                                    match parser.process_token(output.token, &output.bytes) {
                                        StreamResult::Content(text) => {
                                            accumulated.push_str(&text);
                                        }
                                        StreamResult::FlushBuffer(text) => {
                                            accumulated.push_str(&text);
                                        }
                                        StreamResult::Buffering => {}
                                        StreamResult::ToolCalls(mut calls) => {
                                            pending_tool_calls.append(&mut calls);
                                        }
                                    }
                                }

                                match parser.state() {
                                    ParserState::Buffering => {
                                        match parser.finalize_buffered_tool_calls() {
                                            Some(BufferedFinalizeResult::ToolCalls(mut parsed)) => {
                                                pending_tool_calls.append(&mut parsed);
                                            }
                                            Some(BufferedFinalizeResult::FlushBuffer(buffer)) => {
                                                if !buffer.is_empty() {
                                                    accumulated.push_str(&buffer);
                                                }
                                            }
                                            None => {}
                                        }
                                    }
                                    ParserState::Normal => {}
                                }

                                if pending_tool_calls.is_empty() {
                                    let content = if accumulated.is_empty() {
                                        None
                                    } else {
                                        Some(accumulated)
                                    };
                                    (content, None)
                                } else {
                                    finish_reason = "tool_calls".to_string();
                                    (None, Some(pending_tool_calls))
                                }
                            } else {
                                let data = outputs
                                    .iter()
                                    .map(|x| x.token.try_into().unwrap())
                                    .collect::<Vec<_>>();
                                let data = pipeline
                                    .tokenizer()
                                    .decode(&data, group.sampling_params.skip_special_tokens)
                                    .unwrap();
                                (Some(data), None)
                            };

                            if tool_calls.is_none() && finish_reason == "tool_calls" {
                                finish_reason = "stop".to_string();
                            }

                            let choice = ChatChoice {
                                message: ChatChoiceData {
                                    role: "assistant".to_string(),
                                    content,
                                    tool_calls,
                                },
                                finish_reason: Some(finish_reason),
                                index,
                                logprobs: if group.use_logprobs {
                                    Some(WrapperLogprobs { content: outputs })
                                } else {
                                    None
                                },
                            };
                            choices.push(choice);
                        }
                    }

                    let completion_tokens = top_n
                        .iter()
                        .map(|seq| seq.deref().get_len() - seq.deref().get_prompt_len())
                        .sum();
                    let prompt_tokens = top_n.first().unwrap().deref().get_prompt_len();

                    let prompt_time_costs = prompt_finish_time
                        .duration_since(group.created_time)
                        .unwrap()
                        .as_millis();

                    let usage = ChatCompletionUsageResponse {
                        request_id: group.request_id.clone(),
                        created: group.arrival_time,
                        completion_tokens,
                        prompt_tokens,
                        total_tokens: completion_tokens + prompt_tokens,
                        prompt_time_costs: prompt_time_costs as usize,
                        completion_time_costs: completion_time_costs as usize,
                    };

                    responses.insert(group.request_id.clone(), (choices, usage));

                    if do_sync_response {
                        //sync response notification
                        for request_id in responses.keys() {
                            let mut e = engine.write();
                            e.completion_records
                                .insert(request_id.to_string(), responses[request_id].clone());
                            let notify = e.sync_notifies.get(request_id);
                            if let Some(Some(notify)) = notify {
                                notify.notify_one();
                            }
                        }
                    }
                    if let Some(sender) = &group.sender {
                        let seq = Self::primary_sequence(group);
                        if seq.deref().get_finish_reason() != "abort" {
                            debug!(
                                "Sending completion message to client! (sequence id {})",
                                seq.deref().get_id()
                            );
                            let e = engine.read();
                            e.scheduler.print_free_blocks();
                            let _ = sender.send(ChatResponse::Done);
                        } else {
                            aborted_sequences.push(seq.deref().get_id());
                        }
                    };
                }
            }

            #[cfg(feature = "nccl")]
            if multi_process {
                let mut e = engine.write();
                e.sync_abort_sequences(&scheduled, aborted_sequences);
                e.free_finished_sequence_groups_and_sync_mamba(rank);
            };

            {
                let mut e = engine.write();
                e.sync_waiting_task_to_group();
            }
        }

        if rank == 0 {
            let mut e = engine.write();
            let default_pipeline = e.get_mut_pipeline(rank).unwrap().0.as_mut();
            default_pipeline.reset_decoder();
        }

        #[cfg(feature = "nccl")]
        if multi_process && DaemonManager::is_master_rank() {
            warn!("Sending finish message to subprocesses");
            let e = engine.read();
            e.scheduler.print_free_blocks();
            let mut daemon_manager = e.daemon_manager.write();
            let _ = daemon_manager
                .as_mut()
                .unwrap()
                .send_message(&MessageType::Finish);
        }

        debug!("generate_once: finished generation");
        Ok(responses)
    }
}

impl LLMEngine {
    fn execute_scheduler_ops(
        &mut self,
        scheduler_output: &SchedulerOutput,
        rank: usize,
    ) -> Result<()> {
        let cache_engine = Box::new(&mut self.get_mut_pipeline(rank).unwrap().1);
        if !scheduler_output.blocks_to_swap_in.is_empty() {
            cache_engine.swap_in(scheduler_output.blocks_to_swap_in.clone())?;
        }
        if !scheduler_output.blocks_to_swap_out.is_empty() {
            cache_engine.swap_out(scheduler_output.blocks_to_swap_out.clone())?;
        }
        if !scheduler_output.blocks_to_copy.is_empty() {
            cache_engine.copy(scheduler_output.blocks_to_copy.clone())?;
        }
        Ok(())
    }

    fn prepare_block_tables(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
    ) -> Result<Tensor> {
        let mut ordered_sequences = Vec::<Arc<Sequence>>::new();
        for group in groups {
            ordered_sequences.extend(Self::ordered_group_sequences(group));
        }

        let mut max_len = 0;
        for seq in &ordered_sequences {
            let len = self
                .scheduler
                .block_engine
                .block_tables
                .get(&seq.deref().get_id())
                .unwrap()
                .len();
            if len > max_len {
                max_len = len;
            }
        }
        let mut flat: Vec<u32> = Vec::with_capacity(ordered_sequences.len() * max_len);

        for seq in &ordered_sequences {
            let table = self
                .scheduler
                .block_engine
                .block_tables
                .get(&seq.deref().get_id())
                .unwrap();
            let table = table
                .iter()
                .map(|block| block.deref_mut().block_id as u32)
                .collect::<Vec<_>>();

            let bt = if let Some(sliding_window) = self.config.sliding_window {
                let sliding_window_blocks = sliding_window / self.cache_config.block_size;
                let slide_idx = if table.len() > sliding_window_blocks {
                    table.len() - sliding_window_blocks
                } else {
                    0
                };
                table.get(slide_idx..).unwrap().to_vec()
            } else {
                table
            };

            flat.extend_from_slice(bt.as_slice());
            flat.extend(std::iter::repeat(0).take(max_len - bt.len()));
        }

        Tensor::from_vec(flat, (ordered_sequences.len(), max_len), device)
    }

    //Revised based on https://github.com/guoqingbao/vllm.rs/blob/main/src/core/runner.rs#L392
    fn prepare_mamba_slot_mapping(
        &self,
        sequence_ids: &[usize],
        is_prefill: bool,
        rank: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        let slots = if is_prefill {
            pipeline.ensure_mamba_slots_for_sequences(sequence_ids)?
        } else {
            pipeline.get_mamba_slots_for_sequences(sequence_ids)?
        };

        if slots.is_empty() {
            return Ok(None);
        }

        let slots_i64 = slots.into_iter().map(|s| s as i64).collect::<Vec<_>>();
        let len = slots_i64.len();
        Ok(Some(Tensor::from_vec(slots_i64, (len,), device)?))
    }

    fn prepare_prompt(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
        rank: usize,
    ) -> Result<PreparedInputs> {
        let mut context_lens = Vec::new();
        let mut input_ids: Vec<u32> = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0];
        let mut cu_seqlens_k = vec![0];
        let mut sequence_ids = Vec::new();
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        let mut slot_mapping = Vec::new();
        let chunk_size = self.prefill_chunk_size.unwrap_or(PREFILL_CHUNK_SIZE);
        let mut max_context_len = 0;
        #[cfg(feature = "flashinfer")]
        let mut ordered_sequences = Vec::<Arc<Sequence>>::new();
        #[cfg(feature = "flashinfer")]
        let mut prefill_tokens = Vec::new();
        #[cfg(feature = "flashinfer")]
        let mut batch_indices_vec = Vec::<u32>::new();
        #[cfg(feature = "flashinfer")]
        let mut positions_vec = Vec::<u32>::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                #[cfg(feature = "flashinfer")]
                ordered_sequences.push(Arc::clone(&seq));
                let prompt_ids = seq.deref_mut().get_token_ids();
                sequence_ids.push(seq.deref().get_id());
                let seq_len = prompt_ids.len();
                if seq_len > max_context_len {
                    max_context_len = seq_len + self.cache_config.block_size;
                }
                let num_cached_tokens = seq.deref().get_num_cached_tokens();
                let num_tokens = if chunk_size > 0 {
                    std::cmp::min(chunk_size, seq_len - num_cached_tokens)
                } else {
                    seq_len - num_cached_tokens
                };
                #[cfg(feature = "flashinfer")]
                prefill_tokens.push(num_tokens);

                context_lens.push(seq_len as u32);

                let seqlen_q = num_tokens;
                let use_cached_kv = num_cached_tokens > 0
                    && ((cfg!(feature = "flashattn") || cfg!(feature = "flashinfer"))
                        || self.scheduler.prefix_cache_enabled());
                let seqlen_k = if use_cached_kv {
                    num_cached_tokens + num_tokens
                } else {
                    num_tokens
                };

                cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + seqlen_q as u32);
                cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + seqlen_k as u32);
                max_seqlen_q = std::cmp::max(max_seqlen_q, seqlen_q);
                max_seqlen_k = std::cmp::max(max_seqlen_k, seqlen_k);

                input_ids
                    .extend(prompt_ids[num_cached_tokens..num_cached_tokens + num_tokens].to_vec());
                positions.extend(
                    (num_cached_tokens as i64..(num_cached_tokens + num_tokens) as i64)
                        .collect::<Vec<_>>(),
                );
                #[cfg(feature = "flashinfer")]
                {
                    let batch_idx = (sequence_ids.len() - 1) as u32;
                    batch_indices_vec.extend(std::iter::repeat(batch_idx).take(num_tokens));
                    positions_vec.extend(
                        (num_cached_tokens as u32..(num_cached_tokens + num_tokens) as u32)
                            .collect::<Vec<_>>(),
                    );
                }
                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref().get_id());
                if table.is_none() {
                    slot_mapping.extend([_PAD_SLOT_ID].repeat(num_tokens));
                    continue;
                }
                let table = table
                    .unwrap()
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let start_idx = if let Some(sliding_window) = self.config.sliding_window {
                    if seq_len > sliding_window {
                        0.min(seq_len - sliding_window)
                    } else {
                        0
                    }
                } else {
                    0
                };

                for i in num_cached_tokens..num_cached_tokens + num_tokens {
                    if i < start_idx {
                        // Pad [0,start_idx) with _PAD_TOKEN_ID
                        slot_mapping.push(_PAD_SLOT_ID);
                        continue;
                    }

                    let block_number = if i / self.cache_config.block_size >= table.len() {
                        candle_core::bail!(
                            "Block table is too small (prompt)! i={} block_size={} table_len={}",
                            i,
                            self.cache_config.block_size,
                            table.len()
                        );
                    } else {
                        table.get(i / self.cache_config.block_size).unwrap()
                    };
                    let block_offset = i % self.cache_config.block_size;
                    let slot = block_number * self.cache_config.block_size + block_offset;
                    slot_mapping.push(slot as i64);
                }
            }
        }

        assert!(
            input_ids.len() > 0 && positions.len() > 0 && slot_mapping.len() > 0,
            "Invalid inputs!"
        );
        // Validate lengths
        if input_ids.len() != slot_mapping.len() {
            candle_core::bail!(
                "input_ids and slot_mapping must have same length: {}, {}",
                input_ids.len(),
                slot_mapping.len()
            );
        }
        if input_ids.len() != *cu_seqlens_q.last().unwrap() as usize {
            candle_core::bail!("input_ids length must match last cu_seqlens_q",);
        }
        // crate::log_info!("input_ids {:?}, positions {:?}, slot_mapping {:?}", input_ids, positions, slot_mapping);

        // Create tensors
        let length = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), device)?;
        let positions = Tensor::from_vec(positions, (length,), device)?;
        let q_len = cu_seqlens_q.len();
        let k_len = cu_seqlens_k.len();
        let s_len = slot_mapping.len();

        let slot_mapping = Tensor::from_vec(slot_mapping, (s_len,), device)?;

        let (context_lens, block_tables) = if cu_seqlens_k.last() > cu_seqlens_q.last() {
            let len = context_lens.len();
            let context_lens_t = Tensor::from_vec(context_lens, len, device)?;
            let block_tables_t = self.prepare_block_tables(groups, device)?;
            (Some(context_lens_t), Some(block_tables_t))
        } else {
            (None, None)
        };
        let cu_seqlens_q_vec = cu_seqlens_q.clone();
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (q_len,), device)?;
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, (k_len,), device)?;
        let mamba_slot_mapping =
            self.prepare_mamba_slot_mapping(&sequence_ids, true, rank, device)?;
        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if self.flashinfer_kv_params_for_rank(rank)?.is_some() {
            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for (seq, &num_tokens) in ordered_sequences.iter().zip(prefill_tokens.iter()) {
                let effective_len = seq.deref().get_num_cached_tokens() + num_tokens;
                let Some(table) = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref().get_id())
                else {
                    indptr.push(indices.len() as u32);
                    last_len.push(0);
                    continue;
                };
                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id as u32)
                    .collect::<Vec<_>>();
                let max_blocks = table.len();
                let num_blocks = if effective_len == 0 {
                    0
                } else {
                    (effective_len + self.cache_config.block_size - 1)
                        / self.cache_config.block_size
                };
                let num_blocks = std::cmp::min(num_blocks, max_blocks);
                indices.extend(table[..num_blocks].iter().copied());
                indptr.push(indices.len() as u32);
                let last = if effective_len == 0 {
                    0
                } else {
                    (effective_len - 1) % self.cache_config.block_size + 1
                };
                last_len.push(last as u32);
            }
            if let Some(limit) = self.cache_config.num_gpu_blocks {
                if let Some((pos, &bad_idx)) = indices
                    .iter()
                    .enumerate()
                    .find(|(_, idx)| **idx as usize >= limit)
                {
                    candle_core::bail!(
                        "flashinfer prefill block index out of range: indices[{}]={} >= num_gpu_blocks ({})",
                        pos,
                        bad_idx,
                        limit
                    );
                }
            }
            let indptr_host = indptr.clone();
            let last_len_host = last_len.clone();
            let mut kv_len_arr_host = Vec::with_capacity(last_len_host.len());
            for i in 0..last_len_host.len() {
                let num_pages = indptr_host[i + 1] - indptr_host[i];
                if num_pages == 0 {
                    kv_len_arr_host.push(0);
                } else {
                    let full = (num_pages - 1) * self.cache_config.block_size as u32;
                    kv_len_arr_host.push(full + last_len_host[i]);
                }
            }
            Some(FlashInferMetadata {
                indptr: Tensor::from_vec(indptr, (indptr_host.len(),), device)?,
                indptr_host,
                indices: Tensor::from_vec(indices.clone(), (indices.len(),), device)?,
                last_len: Tensor::from_vec(last_len, (last_len_host.len(),), device)?,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                cu_seqlens_q_host: Some(cu_seqlens_q_vec.clone()),
                total_num_rows: Some(*cu_seqlens_q_vec.last().unwrap()),
                batch_indices: Some(Tensor::from_vec(batch_indices_vec, (length,), device)?),
                positions: Some(Tensor::from_vec(positions_vec, (length,), device)?),
                use_cuda_graph: false,
                decode_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;

        let input_metadata = InputMetadata {
            is_prefill: true,
            sequence_ids: Some(sequence_ids),
            mamba_slot_mapping,
            slot_mapping,
            block_tables,
            context_lens,
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            max_seqlen_q,
            max_seqlen_k,
            max_context_len,
            disable_flash_attn: None,
            seqlens: Some(cu_seqlens_q_vec[1..].to_vec()),
            flashinfer_metadata,
        };

        Ok(PreparedInputs {
            tokens: input_ids,
            positions: positions,
            metadata: input_metadata,
        })
    }

    //Revised based on https://github.com/guoqingbao/vllm.rs/blob/main/src/core/runner.rs#L498
    fn prepare_decode(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
        rank: usize,
    ) -> Result<PreparedInputs> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut sequence_ids = Vec::new();
        let mut context_lens = Vec::new();
        let mut block_tables = Vec::new();
        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                sequence_ids.push(seq.deref().get_id());
                let last_token_id = seq.deref_mut().get_last_token_id();
                input_ids.push(last_token_id);
                let position = seq.deref_mut().get_len() - 1;
                positions.push(position as i64);

                let context_len = if let Some(sliding_window) = self.config.sliding_window {
                    seq.deref_mut().get_len().min(sliding_window)
                } else {
                    seq.deref_mut().get_len()
                };
                context_lens.push(context_len as u32);

                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref().get_id())
                    .unwrap();
                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let block_number = if position / self.cache_config.block_size >= table.len() {
                    candle_core::bail!("Block table is too small (completion)! start_pos={} block_size={} table_len={}", position, self.cache_config.block_size, table.len());
                } else {
                    table.get(position / self.cache_config.block_size).unwrap()
                };
                let block_offset = position % self.cache_config.block_size;
                let slot = block_number * self.cache_config.block_size + block_offset;
                let slot: i64 = slot.try_into().unwrap();
                slot_mapping.push(slot);

                if let Some(sliding_window) = self.config.sliding_window {
                    let sliding_window_blocks = sliding_window / self.cache_config.block_size;
                    let slide_idx = if table.len() > sliding_window_blocks {
                        table.len() - sliding_window_blocks
                    } else {
                        0
                    };
                    block_tables.push(table.get(slide_idx..).unwrap().to_vec());
                } else {
                    block_tables.push(table);
                }
            }
        }

        let length = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (length,), device)?;
        let positions = Tensor::from_vec(positions, (length,), device)?;
        let slot_mapping = Tensor::from_vec(slot_mapping, (length,), device)?;

        let max_context_len = context_lens.clone().into_iter().max().unwrap();
        let context_lens = Tensor::from_vec(context_lens, (length,), device)?;

        let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
        let block_tables = super::_make_tensor_with_pad(
            block_tables
                .iter()
                .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            max_block_table_len,
            0,
            device,
        )?;
        let block_tables = block_tables.reshape(((), max_block_table_len))?;
        let mamba_slot_mapping =
            self.prepare_mamba_slot_mapping(&sequence_ids, false, rank, device)?;
        #[cfg(feature = "flashinfer")]
        let flashinfer_metadata = if self.flashinfer_kv_params_for_rank(rank)?.is_some() {
            #[cfg(all(feature = "cuda", feature = "graph"))]
            let use_cuda_graph = {
                let (pipeline, _) = self.get_pipeline(rank).ok_or_else(|| {
                    candle_core::Error::msg(format!("missing pipeline for rank {rank}"))
                })?;
                pipeline.capturer.is_exact_captured(length)
            };
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            let use_cuda_graph = false;

            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for group in groups {
                for seq in Self::ordered_group_sequences(group) {
                    let table = self
                        .scheduler
                        .block_engine
                        .block_tables
                        .get(&seq.deref().get_id())
                        .unwrap()
                        .iter()
                        .map(|block| block.deref_mut().block_id as u32)
                        .collect::<Vec<_>>();
                    indices.extend(table);
                    indptr.push(indices.len() as u32);
                    let len = seq.deref().get_len();
                    let last = if len == 0 {
                        0
                    } else {
                        (len - 1) % self.cache_config.block_size + 1
                    };
                    last_len.push(last as u32);
                }
            }
            if let Some(limit) = self.cache_config.num_gpu_blocks {
                if let Some((pos, &bad_idx)) = indices
                    .iter()
                    .enumerate()
                    .find(|(_, idx)| **idx as usize >= limit)
                {
                    candle_core::bail!(
                        "flashinfer decode block index out of range: indices[{}]={} >= num_gpu_blocks ({})",
                        pos,
                        bad_idx,
                        limit
                    );
                }
            }
            let indptr_host = indptr.clone();
            let last_len_host = last_len.clone();
            let mut kv_len_arr_host = Vec::with_capacity(last_len_host.len());
            for i in 0..last_len_host.len() {
                let num_pages = indptr_host[i + 1] - indptr_host[i];
                if num_pages == 0 {
                    kv_len_arr_host.push(0);
                } else {
                    let full = (num_pages - 1) * self.cache_config.block_size as u32;
                    kv_len_arr_host.push(full + last_len_host[i]);
                }
            }
            Some(FlashInferMetadata {
                indptr: Tensor::from_vec(indptr, (length + 1,), device)?,
                indptr_host,
                indices: Tensor::from_vec(indices.clone(), (indices.len(),), device)?,
                last_len: Tensor::from_vec(last_len, (length,), device)?,
                last_len_host: Some(last_len_host),
                kv_len_arr_host: Some(kv_len_arr_host),
                cu_seqlens_q_host: None,
                total_num_rows: None,
                batch_indices: None,
                positions: None,
                use_cuda_graph,
                decode_plan_info: None,
            })
        } else {
            None
        };
        #[cfg(not(feature = "flashinfer"))]
        let flashinfer_metadata = None;
        let input_metadata = InputMetadata {
            is_prefill: false,
            sequence_ids: Some(sequence_ids),
            mamba_slot_mapping,
            slot_mapping,
            block_tables: Some(block_tables),
            context_lens: Some(context_lens),
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            max_context_len: max_context_len as usize,
            disable_flash_attn: None,
            seqlens: None,
            flashinfer_metadata,
        };

        Ok(PreparedInputs {
            tokens: input_ids,
            positions: positions,
            metadata: input_metadata,
        })
    }

    pub fn create_sequence_group(
        &mut self,
        seq_id: usize,
        group_id: usize,
        prompt: &Vec<u32>,
        request_id: &str,
        created: SystemTime,
        sampling_params: &SamplingParams,
        use_logprobs: bool,
        is_embedding: bool,
        encoding_format: crate::openai::requests::EncodingFormat,
        embedding_type: crate::openai::requests::EmbeddingType,
        tools: Vec<crate::tools::Tool>,
        sender: Option<Sender<ChatResponse>>,
    ) -> SequenceGroup {
        let seq = Arc::new(Sequence(std::sync::RwLock::new(_Sequence::new(
            prompt,
            seq_id,
            self.cache_config.block_size,
        ))));
        SequenceGroup::new(
            &[seq],
            get_created_time_secs(),
            group_id,
            request_id.to_owned(),
            created,
            sampling_params.clone(),
            use_logprobs,
            is_embedding,
            encoding_format,
            embedding_type,
            tools,
            sender,
        )
    }

    pub fn add_request(
        &mut self,
        prompt: Vec<u32>,
        request_id: String,
        created: SystemTime,
        sampling_params: SamplingParams,
        use_logprobs: bool,
        is_embedding: bool,
        encoding_format: crate::openai::requests::EncodingFormat,
        embedding_type: crate::openai::requests::EmbeddingType,
        tools: Vec<crate::tools::Tool>,
        sender: Option<Arc<Sender<ChatResponse>>>,
        sync_notify: Option<Arc<Notify>>,
    ) {
        let prompt_len = prompt.len();
        let sync_notify = sync_notify.clone();
        if let Some(sync) = sync_notify {
            self.sync_notifies.insert(request_id.clone(), Some(sync));
        }

        let sender_clone = sender.clone();
        if let Some(sender) = sender_clone {
            self.senders.insert(request_id.clone(), Some(sender));
        }

        #[cfg(feature = "nccl")]
        let do_log = DaemonManager::is_master_rank();
        #[cfg(not(feature = "nccl"))]
        let do_log = true;
        if do_log {
            warn!(
                "New Request with length {} tokens ({}).",
                prompt_len,
                request_id.clone(),
            );
        }

        let task = TaskData {
            seq_id: self.seq_id,
            group_id: self.group_id,
            prompt,
            request_id,
            created,
            sampling_params,
            use_logprobs,
            is_embedding,
            encoding_format,
            embedding_type,
            tools,
        };
        let mut waiting_tasks = self.waiting_tasks.write();
        waiting_tasks.push(task);
        self.seq_id += 1;
        self.group_id += 1;
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        self.scheduler.get_available_kv_tokens()
    }
}
