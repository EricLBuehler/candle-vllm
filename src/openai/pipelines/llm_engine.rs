use super::DefaultPipeline;
#[path = "inputs.rs"]
mod inputs;
#[cfg(feature = "nccl")]
#[path = "multiprocess.rs"]
mod multiprocess;
#[path = "streaming.rs"]
mod streaming;
#[path = "threaded.rs"]
mod threaded;

#[cfg(feature = "nccl")]
use crate::openai::communicator::DaemonManager;
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
        multimodal::compute_image_slice,
        multimodal::ImageData,
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
use candle_core::{Result, Tensor};
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
    collections::{HashMap, VecDeque},
    iter::zip,
    sync::Arc,
};
use tokio::sync::Notify;
#[allow(unused_imports)]
use tracing::{debug, info, warn};
#[allow(dead_code)]
pub struct PreparedInputs {
    tokens: Tensor,
    positions: Tensor,
    metadata: InputMetadata,
}

pub struct StreamEmission {
    content: Option<String>,
    tool_calls: Option<Vec<crate::tools::ToolCall>>,
}

pub struct BatchExecution {
    logits: Tensor,
    is_prompt: bool,
    is_embedding: bool,
    model_name: String,
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
    pub exit_flag: Arc<AtomicBool>,
    last_decoding_throughput_log_ms: usize,
}

impl LLMEngine {
    #[cfg(feature = "nccl")]
    pub(crate) fn planned_prompt_cache_statuses(
        &mut self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        rank: usize,
    ) -> Result<Vec<(usize, usize, bool)>> {
        if scheduled.is_empty() || !Self::primary_sequence(&scheduled[0]).deref().is_prompt() {
            return Ok(Vec::new());
        }

        let restore_plans = self.scheduler.prepare_prompt_mamba_restores(scheduled);
        let restore_by_seq = restore_plans
            .into_iter()
            .map(|plan| (plan.seq_id, plan))
            .collect::<HashMap<_, _>>();
        let (pipeline, _) = self.get_pipeline(rank).unwrap();
        let mut statuses = Vec::new();
        for group in scheduled {
            for seq in Self::ordered_group_sequences(group) {
                let seq_id = seq.deref().get_id();
                let cached_tokens = seq.deref().get_num_cached_tokens();
                let available = if cached_tokens == 0 {
                    true
                } else if pipeline.has_mamba_slot_for_sequence(seq_id) {
                    true
                } else if let Some(plan) = restore_by_seq.get(&seq_id) {
                    pipeline.has_mamba_prefix_state(plan.hash)?
                } else {
                    false
                };
                statuses.push((seq_id, cached_tokens, available));
            }
        }
        Ok(statuses)
    }

    #[cfg(feature = "nccl")]
    pub(crate) fn apply_prompt_mamba_fallbacks(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        fallback_seq_ids: &[usize],
    ) -> Result<()> {
        if fallback_seq_ids.is_empty() {
            return Ok(());
        }

        let fallback_ids = fallback_seq_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let restore_by_seq = self
            .scheduler
            .prepare_prompt_mamba_restores(groups)
            .into_iter()
            .map(|plan| (plan.seq_id, plan.clone()))
            .collect::<HashMap<_, _>>();

        for group in groups {
            for seq in Self::ordered_group_sequences(group) {
                let seq_id = seq.deref().get_id();
                if !fallback_ids.contains(&seq_id) {
                    continue;
                }
                if let Some(restore) = restore_by_seq.get(&seq_id) {
                    self.scheduler.handle_missing_mamba_snapshot(restore);
                }
            }
        }

        Ok(())
    }

    fn effective_mamba_prefix_capacity(
        prefix_cache_enabled: bool,
        mamba_slot_capacity: usize,
    ) -> usize {
        if !prefix_cache_enabled || mamba_slot_capacity == 0 {
            return 0;
        }
        // Keep a larger snapshot pool than active slots so prompt/chunk-prefill
        // boundaries survive decode-time snapshot churn when prefix cache is hot.
        mamba_slot_capacity.saturating_mul(2)
    }

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

    fn capture_mamba_prefix_states_for_prefill_progress(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        chunk_size: usize,
        rank: usize,
    ) -> Result<()> {
        let captures = self
            .scheduler
            .collect_prefill_mamba_captures(groups, chunk_size);
        if captures.is_empty() {
            return Ok(());
        }

        let mut captured = Vec::new();
        {
            let pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            for capture in captures {
                match pipeline.capture_mamba_prefix_state(capture.seq_id, capture.hash, true) {
                    Ok(true) => captured.push(capture),
                    Ok(false) => {}
                    Err(e) => {
                        tracing::warn!(
                            "Failed to capture prefill mamba prefix state for seq {} hash {}: {}",
                            capture.seq_id,
                            capture.hash,
                            e
                        );
                    }
                }
            }
        }
        self.scheduler.record_mamba_prefix_captures(captured);
        Ok(())
    }

    fn capture_mamba_prefix_states_for_decode_progress(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        rank: usize,
    ) -> Result<()> {
        let captures = self.scheduler.collect_decode_mamba_captures(groups);
        if captures.is_empty() {
            return Ok(());
        }

        let mut captured = Vec::new();
        {
            let pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            for capture in captures {
                match pipeline.capture_mamba_prefix_state(capture.seq_id, capture.hash, false) {
                    Ok(true) => captured.push(capture),
                    Ok(false) => {}
                    Err(e) => {
                        tracing::warn!(
                            "Failed to capture decode mamba prefix state for seq {} hash {}: {}",
                            capture.seq_id,
                            capture.hash,
                            e
                        );
                    }
                }
            }
        }
        self.scheduler.record_mamba_prefix_captures(captured);
        Ok(())
    }

    fn restore_mamba_prefix_states_for_prompt(
        &mut self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        rank: usize,
    ) -> Result<()> {
        let mut restore_plans = self.scheduler.prepare_prompt_mamba_restores(groups);
        if restore_plans.is_empty() {
            return Ok(());
        }

        // Chunked-prefill continuations keep their live mamba slot and state.
        // Skip snapshot restore for those sequences and only restore requests
        // that need their prefix state materialized into a fresh slot.
        let resident_seq_ids = {
            let (pipeline, _) = self.get_pipeline(rank).unwrap();
            restore_plans
                .iter()
                .filter_map(|plan| {
                    pipeline
                        .has_mamba_slot_for_sequence(plan.seq_id)
                        .then_some(plan.seq_id)
                })
                .collect::<std::collections::HashSet<_>>()
        };
        for seq_id in &resident_seq_ids {
            self.scheduler.mark_mamba_restored(*seq_id);
        }
        restore_plans.retain(|plan| !resident_seq_ids.contains(&plan.seq_id));
        if restore_plans.is_empty() {
            return Ok(());
        }

        {
            let (pipeline, _) = self.get_pipeline(rank).unwrap();
            let restore_candidates = restore_plans
                .iter()
                .map(|plan| plan.seq_id)
                .collect::<Vec<_>>();
            pipeline.ensure_mamba_slots_for_sequences(&restore_candidates)?;
        }

        for restore in restore_plans {
            let has_snapshot = {
                let (pipeline, _) = self.get_pipeline(rank).unwrap();
                pipeline.has_mamba_prefix_state(restore.hash)?
            };
            if !has_snapshot {
                self.scheduler.handle_missing_mamba_snapshot(&restore);
                continue;
            }

            let restored = {
                let (pipeline, _) = self.get_pipeline(rank).unwrap();
                pipeline.restore_mamba_prefix_state(restore.seq_id, restore.hash)?
            };
            if restored {
                self.scheduler.mark_mamba_restored(restore.seq_id);
                tracing::info!(
                    "Restored mamba prefix state on rank {} for seq {}",
                    rank,
                    restore.seq_id,
                );
            } else {
                self.scheduler.handle_failed_mamba_restore(&restore);
            }
        }

        Ok(())
    }

    fn free_finished_sequence_groups_and_sync_mamba(&mut self, rank: usize) {
        let sync = self
            .scheduler
            .free_finished_sequence_groups_and_collect_mamba();

        let mut captured = Vec::new();
        {
            let pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            for capture in sync.captures {
                match pipeline.capture_mamba_prefix_state(capture.seq_id, capture.hash, true) {
                    Ok(true) => captured.push(capture),
                    Ok(false) => {}
                    Err(e) => {
                        tracing::warn!(
                            "Failed to capture mamba prefix state for seq {} hash {}: {}",
                            capture.seq_id,
                            capture.hash,
                            e
                        );
                    }
                }
            }
        }
        self.scheduler.record_mamba_prefix_captures(captured);
        {
            let pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            for seq_id in sync.released_ids {
                pipeline.release_sequence_state(seq_id);
            }
        }
    }

    fn validate_scheduled_prompt_mamba_prefix_states(
        &mut self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        rank: usize,
    ) -> Result<usize> {
        if scheduled.is_empty() || !Self::primary_sequence(&scheduled[0]).deref().is_prompt() {
            return Ok(0);
        }

        let restore_plans = self.scheduler.prepare_prompt_mamba_restores(scheduled);
        if restore_plans.is_empty() {
            return Ok(0);
        }

        let mut downgraded = 0usize;
        for restore in restore_plans {
            let has_snapshot = {
                let (pipeline, _) = self.get_pipeline(rank).unwrap();
                if pipeline.has_mamba_slot_for_sequence(restore.seq_id) {
                    true
                } else {
                    pipeline.has_mamba_prefix_state(restore.hash)?
                }
            };
            if !has_snapshot {
                self.scheduler.handle_missing_mamba_snapshot(&restore);
                downgraded += 1;
            }
        }

        Ok(downgraded)
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
        self.scheduler.reset_mamba_state();
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
        let mut mamba_prefix_capacity = Self::effective_mamba_prefix_capacity(
            scheduler_config.prefix_cache.enabled,
            mamba_slot_capacity,
        );

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
                    let prefix_budget_slots =
                        planned_total_slots.saturating_sub(mamba_slot_capacity);
                    if prefix_budget_slots == 0 && mamba_prefix_capacity > 0 {
                        info!(
                            "Hybrid mamba planned budget leaves 0 explicit snapshot slot(s); using auto prefix-state capacity {}.",
                            mamba_prefix_capacity
                        );
                    } else if mamba_prefix_capacity > prefix_budget_slots {
                        warn!(
                            "Capping hybrid mamba prefix-state cache from {} to {} entries by planned budget.",
                            mamba_prefix_capacity,
                            prefix_budget_slots
                        );
                        mamba_prefix_capacity = prefix_budget_slots;
                    }
                } else if planned_total_slots > 0 {
                    warn!(
                        "Hybrid mamba planned budget only fits {} total slot(s), below the target active minimum {}; using the largest safe active capacity instead.",
                        planned_total_slots,
                        mamba_slot_capacity
                    );
                    mamba_slot_capacity = planned_total_slots;
                    mamba_prefix_capacity = Self::effective_mamba_prefix_capacity(
                        scheduler_config.prefix_cache.enabled,
                        mamba_slot_capacity,
                    );
                } else {
                    warn!(
                        "Hybrid mamba planned budget is smaller than one slot; falling back to 1 active slot."
                    );
                    mamba_slot_capacity = 1;
                    mamba_prefix_capacity = Self::effective_mamba_prefix_capacity(
                        scheduler_config.prefix_cache.enabled,
                        mamba_slot_capacity,
                    );
                }

                let active_bytes = mamba_slot_capacity.saturating_mul(estimate.slot_bytes);
                let prefix_budget_bytes = cache_config
                    .mamba_cache_budget_bytes
                    .saturating_sub(active_bytes);
                let actual_prefix_bytes = mamba_prefix_capacity.saturating_mul(estimate.slot_bytes);

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
                    actual_prefix_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                );
                info!(
                    "Hybrid mamba planned prefix snapshot budget: {:.2} GB",
                    prefix_budget_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                );
            } else {
                warn!(
                    "Hybrid mamba budget was not reserved in the combined cache budget; using fallback active slot capacity {} and auto prefix-state capacity {}.",
                    mamba_slot_capacity,
                    mamba_prefix_capacity
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
                        let should_continue = if multi_process {
                            #[cfg(feature = "nccl")]
                            {
                                Self::sync_multiprocess_waiting_tasks_before_cycle(&engine)
                            }
                            #[cfg(not(feature = "nccl"))]
                            {
                                false
                            }
                        } else {
                            Self::sync_threaded_waiting_tasks_before_cycle(&engine)
                        };
                        if should_continue {
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

    fn move_waiting_tasks_to_scheduler(&mut self) -> Vec<TaskData> {
        let send_tasks = {
            let waiting_tasks = self.waiting_tasks.write();
            waiting_tasks.clone()
        };

        for task in &send_tasks {
            let sender: Option<Sender<ChatResponse>> = self
                .senders
                .get(&task.request_id)
                .and_then(|opt_arc_sender| opt_arc_sender.as_ref().map(|arc| arc.as_ref().clone()));
            let seq_group = self.create_sequence_group(
                task.seq_id,
                task.group_id,
                &task.prompt,
                task.images.clone(),
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

        self.waiting_tasks.write().clear();
        send_tasks
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
        device: &candle_core::Device,
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

    pub fn generate_once(
        engine: Arc<RwLock<Self>>,
        rank: usize,
        multi_process: bool,
    ) -> Result<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        if multi_process {
            #[cfg(feature = "nccl")]
            {
                return Self::generate_once_multiprocess(engine, rank);
            }
        }
        Self::generate_once_threaded(engine, rank)
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

    #[allow(unused)]
    fn bind_rank_to_thread(&self, rank: usize) {
        #[cfg(feature = "cuda")]
        {
            debug!("Start processing...");
            let (pipeline, _) = self.get_pipeline(rank).unwrap();
            let device = pipeline.device();
            let _ = device.as_cuda_device().unwrap().bind_to_thread();
        }
    }

    fn has_unfinished_sequences(&self) -> bool {
        self.scheduler.has_unfinished_sequences()
    }

    fn schedule_current_batch(&mut self, rank: usize) -> Result<()> {
        let scheduler_outputs = self.scheduler.schedule();
        if !scheduler_outputs.ignored_seq_groups.is_empty() {
            for group in scheduler_outputs.ignored_seq_groups.iter() {
                if let Some(sender) = &group.sender {
                    let _ = sender.send(ChatResponse::ModelError(
                        candle_core::Error::msg("Ignored sequence group: allocation impossible")
                            .to_string(),
                    ));
                }
            }
        }
        let is_multiprocess = self.multi_process;
        if !is_multiprocess {
            let downgraded = self.validate_scheduled_prompt_mamba_prefix_states(
                &scheduler_outputs.scheduled,
                rank,
            )?;
            if downgraded > 0 {
                warn!(
                    "Prefill fallback: {} sequence(s) downgraded to full prefill due to missing mamba snapshots.",
                    downgraded
                );
            }
        }
        self.execute_scheduler_ops(&scheduler_outputs, rank)?;
        let mut groups = self.sequence_groups.write();
        *groups = match Arc::try_unwrap(scheduler_outputs.scheduled) {
            Ok(deq) => deq,
            Err(arc_deq) => (*arc_deq).clone(),
        };
        Ok(())
    }

    fn current_scheduled_groups(&self) -> VecDeque<Arc<SequenceGroup>> {
        self.sequence_groups.read().clone()
    }

    fn clear_current_scheduled_groups(&self) {
        self.sequence_groups.write().clear();
    }

    fn execute_scheduled_batch(
        &mut self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        rank: usize,
    ) -> Result<BatchExecution> {
        let is_embedding = scheduled[0].is_embedding;
        let is_prompt_request = Self::primary_sequence(&scheduled[0]).deref().is_prompt();

        if is_prompt_request {
            self.restore_mamba_prefix_states_for_prompt(scheduled, rank)?;
        }

        let (pipeline, cache_engine) = self.get_pipeline(rank).unwrap();
        let device = pipeline.device();
        let model_name = pipeline.name().to_string();
        #[cfg_attr(not(feature = "flashinfer"), allow(unused_mut))]
        let mut prepared = if is_prompt_request {
            self.prepare_prompt(scheduled, device, rank)
        } else {
            self.prepare_decode(scheduled, device, rank)
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
                self.ensure_flashinfer_decode_plan(
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

        let images: Option<ImageData> = if is_prompt_request {
            let seq = Self::primary_sequence(&scheduled[0]);
            let seq_guard = seq.deref();
            let seq_images = seq_guard.get_images();
            let seq_token_ids = seq_guard.get_token_ids();
            let seq_num_cached_tokens = seq_guard.get_num_cached_tokens();
            drop(seq_guard);
            if let Some(images) = seq_images {
                if scheduled.len() > 1 {
                    candle_core::bail!(
                        "multimodal prefill does not support batching multiple sequence groups"
                    );
                }
                if images.image_idx == -1 {
                    None
                } else {
                    compute_image_slice(&seq_token_ids, seq_num_cached_tokens, &images).map(
                        |(image_idx, token_offset)| {
                            let mut images = images.clone();
                            images.image_idx = image_idx;
                            images.image_token_offset = token_offset;
                            images
                        },
                    )
                }
            } else {
                None
            }
        } else {
            None
        };

        let logits = if is_embedding {
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
                images.as_ref(),
            )?
        };

        Ok(BatchExecution {
            logits,
            is_prompt: metadata.is_prefill,
            is_embedding,
            model_name,
        })
    }

    fn process_embedding_batch(
        &mut self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        batch: &BatchExecution,
        rank: usize,
    ) -> Result<bool> {
        if !batch.is_embedding {
            return Ok(false);
        }
        if !batch.is_prompt {
            return Ok(true);
        }

        let mut start_idx = 0;
        for group in scheduled {
            let seq = Self::primary_sequence(group);
            let prompt_len = seq.deref().get_prompt_len();
            let end_idx = start_idx + prompt_len;
            let seq_embedding = batch.logits.narrow(0, start_idx, prompt_len)?;

            let pooled_embedding = match group.embedding_type {
                crate::openai::requests::EmbeddingType::Last => {
                    seq_embedding.narrow(0, prompt_len - 1, 1)?.squeeze(0)?
                }
                crate::openai::requests::EmbeddingType::Mean => seq_embedding.mean(0)?,
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
                    model: batch.model_name.clone(),
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

        self.free_finished_sequence_groups_and_sync_mamba(rank);
        Ok(true)
    }

    fn process_prefill_progress(
        &mut self,
        scheduled: &mut VecDeque<Arc<SequenceGroup>>,
        batch: &mut BatchExecution,
        rank: usize,
    ) -> Result<bool> {
        if !batch.is_prompt {
            return Ok(false);
        }

        let prefill_chunk_size = self.prefill_chunk_size.unwrap_or(PREFILL_CHUNK_SIZE);
        if prefill_chunk_size == 0 {
            return Ok(false);
        }

        self.capture_mamba_prefix_states_for_prefill_progress(scheduled, prefill_chunk_size, rank)?;
        let (finished_indices, finished_groups) = self
            .scheduler
            .filter_prefill_finished(scheduled, prefill_chunk_size);

        if finished_indices.is_empty() {
            return Ok(true);
        }

        *scheduled = finished_groups;
        let batch_size = finished_indices.len();
        batch.logits = batch.logits.index_select(
            &Tensor::from_vec(finished_indices, (batch_size,), batch.logits.device())?,
            0,
        )?;
        Ok(false)
    }

    fn apply_sample_results(
        &mut self,
        rank: usize,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        results: Vec<TokenOrFinishReason>,
        prompt_finish_times: &mut HashMap<usize, SystemTime>,
    ) -> Result<()> {
        if results.len() != scheduled.len() {
            candle_core::bail!(
                "Sample result and scheduled group length mismatch on rank {}: {} vs {}",
                rank,
                results.len(),
                scheduled.len()
            );
        }

        for (result_, group) in zip(results, scheduled) {
            match result_ {
                Either::Left(logprobs) => {
                    let seq = Self::primary_sequence(group);
                    if seq.deref().is_prompt() {
                        self.scheduler.print_free_blocks();
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
                                    seq.deref().get_prompt_len() * 1000 / prompt_time_costs as usize,
                                    group.request_id,
                                );
                            }
                        }
                    }
                    if let Some(sender) = &group.sender {
                        let emission =
                            self.collect_stream_emission_for_token(rank, group, &seq, &logprobs);
                        if emission.tool_calls.is_some() || emission.content.is_some() {
                            self.send_stream_emission(rank, sender, group, &seq, emission, None);
                        }
                    }
                    seq.deref_mut().add_token(logprobs);
                }
                Either::Right(finish_reason) => {
                    let seq = Self::primary_sequence(group);
                    self.apply_pending_finish_logprobs(rank, group, &seq);
                    if let Some(sender) = &group.sender {
                        let emission = self.collect_stream_emission_on_finish(group, &seq);
                        let stream_finish_reason = if finish_reason == "tool_calls" {
                            "stop".to_string()
                        } else {
                            finish_reason.clone()
                        };
                        self.send_stream_emission(
                            rank,
                            sender,
                            group,
                            &seq,
                            emission,
                            Some(stream_finish_reason),
                        );
                    }
                    seq.deref_mut().set_finish_reason(finish_reason);
                }
            }
        }
        Ok(())
    }

    fn finalize_post_sampling(
        &mut self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        rank: usize,
        prompt_finish_times: &HashMap<usize, SystemTime>,
        is_prompt: bool,
    ) -> Result<()> {
        self.capture_mamba_prefix_states_for_decode_progress(scheduled, rank)?;
        self.free_finished_sequence_groups_and_sync_mamba(rank);
        if !is_prompt {
            self.may_log_decoding_throughput(scheduled, prompt_finish_times, rank);
        }
        Ok(())
    }

    fn collect_finished_responses(
        &mut self,
        scheduled: &VecDeque<Arc<SequenceGroup>>,
        responses: &mut HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>,
        prompt_finish_times: &HashMap<usize, SystemTime>,
        allow_sync_response: bool,
    ) -> Vec<usize> {
        let mut aborted_sequences = Vec::new();
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
                let do_sync_response = allow_sync_response && group.sender.is_none();

                if do_sync_response {
                    let pipeline = self.get_pipeline(0usize).unwrap().0.as_ref();
                    for (index, seq) in top_n.iter().enumerate() {
                        let outputs = seq.deref_mut().get_output_tokens();
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
                                    StreamResult::Content(text)
                                    | StreamResult::FlushBuffer(text) => {
                                        accumulated.push_str(&text);
                                    }
                                    StreamResult::Buffering => {}
                                    StreamResult::ToolCalls(mut calls) => {
                                        pending_tool_calls.append(&mut calls);
                                    }
                                }
                            }

                            let mut buffered_finish_content = String::new();
                            if matches!(parser.state(), ParserState::Buffering) {
                                match parser.finalize_buffered_tool_calls() {
                                    Some(BufferedFinalizeResult::ToolCalls(mut parsed)) => {
                                        pending_tool_calls.append(&mut parsed);
                                    }
                                    Some(BufferedFinalizeResult::FlushBuffer(buffer)) => {
                                        if !buffer.is_empty() {
                                            buffered_finish_content.push_str(&buffer);
                                        }
                                    }
                                    None => {}
                                }
                            }

                            if pending_tool_calls.is_empty() {
                                let mut reparsed = parser.reparse_accumulated_output();
                                if !reparsed.is_empty() {
                                    warn!(
                                        "Recovered {} tool call(s) from full-output fallback parse",
                                        reparsed.len()
                                    );
                                    pending_tool_calls.append(&mut reparsed);
                                }
                            }

                            if pending_tool_calls.is_empty() && !buffered_finish_content.is_empty()
                            {
                                accumulated.push_str(&buffered_finish_content);
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

                        choices.push(ChatChoice {
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
                        });
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

                responses.insert(group.request_id.clone(), (choices, usage.clone()));

                if do_sync_response {
                    if let Some((choices, usage)) = responses.get(&group.request_id).cloned() {
                        self.completion_records
                            .insert(group.request_id.clone(), (choices, usage));
                    }
                    if let Some(Some(notify)) = self.sync_notifies.get(&group.request_id) {
                        notify.notify_one();
                    }
                }

                if let Some(sender) = &group.sender {
                    if seq.deref().get_finish_reason() != "abort" {
                        debug!(
                            "Sending completion message to client! (sequence id {})",
                            seq.deref().get_id()
                        );
                        self.scheduler.print_free_blocks();
                        let _ = sender.send(ChatResponse::Done);
                    } else {
                        aborted_sequences.push(seq.deref().get_id());
                    }
                }
            }
        }
        aborted_sequences
    }

    fn reset_decoder_for_rank(&mut self, rank: usize) {
        if rank == 0 {
            let default_pipeline = self.get_mut_pipeline(rank).unwrap().0.as_mut();
            default_pipeline.reset_decoder();
        }
    }

    pub fn create_sequence_group(
        &mut self,
        seq_id: usize,
        group_id: usize,
        prompt: &Vec<u32>,
        images: Option<ImageData>,
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
            images,
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
        images: Option<ImageData>,
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
            images,
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
