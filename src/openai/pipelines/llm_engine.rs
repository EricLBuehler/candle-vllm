use super::DefaultPipeline;
#[cfg(feature = "nccl")]
use crate::openai::communicator::{DaemonManager, MessageType, TaskSampleData};
#[cfg(feature = "nccl")]
use crate::openai::pipelines::TokenOrFinishReason;
use crate::openai::streaming::ChatResponse;
use crate::openai::TaskData;
use crate::scheduler::Scheduler;
use crate::{
    openai::{
        models::Config,
        responses::{
            ChatChoice, ChatChoiceData, ChatCompletionChunk, ChatCompletionUsageResponse, Choice,
            ChoiceData, WrapperLogprobs,
        },
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
use candle_core::{Device, Result, Tensor};
use either::Either;
use flume::Sender;
use parking_lot::RwLock;
#[cfg(feature = "nccl")]
use rayon::iter::IntoParallelRefIterator;
#[cfg(feature = "nccl")]
use rayon::iter::ParallelIterator;
use std::time::SystemTime;
use std::{
    collections::{HashMap, VecDeque},
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

const _PAD_SLOT_ID: i64 = -1;
const PREFILL_CHUNK_SIZE: usize = 8192;

#[allow(unused)]
pub struct LLMEngine {
    pipelines: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)>,
    pub scheduler: Scheduler,
    seq_id: usize,
    cache_config: CacheConfig,
    config: Config,
    group_id: usize,
    pub notify: Arc<Notify>,
    sync_notifies: HashMap<String, Option<Arc<Notify>>>,
    senders: HashMap<String, Option<Arc<Sender<ChatResponse>>>>,
    pub completion_records: HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>,
    sequence_groups: RwLock<VecDeque<Arc<SequenceGroup>>>,
    multi_process: bool,
    num_shards: usize,
    waiting_tasks: RwLock<Vec<TaskData>>,
    #[cfg(feature = "nccl")]
    pub daemon_manager: RwLock<Option<DaemonManager>>,
    prefill_chunk_size: Option<usize>,
}

impl LLMEngine {
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
    pub fn graph_capture(engine: &Arc<RwLock<LLMEngine>>) -> Result<()> {
        let mut e = engine.write();
        let (ref mut pipeline, cache_engine) = e.get_mut_pipeline(0usize).unwrap();
        let device = pipeline.device();
        let _ = device.as_cuda_device().unwrap().bind_to_thread();
        let x = pipeline.warmup_capture(Some(&cache_engine.get_kv_cache()));
        x
    }

    pub fn new(
        pipelines: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)>,
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
        let num_threads: usize = pipelines.len();
        let engine = Arc::new(RwLock::new(Self {
            pipelines,
            scheduler: Scheduler::new(scheduler_config, cache_config),
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

        let _ = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                loop {
                    if is_master_rank {
                        notify.notified().await;
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
                    let _ = daemon_manager
                        .as_mut()
                        .unwrap()
                        .send_message(&MessageType::Data(send_tasks));
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

    fn get_stream_response(
        &self,
        request_id: String,
        created: u64,
        content: Option<String>,
        finish_reason: Option<String>,
        pipeline: &DefaultPipeline,
    ) -> ChatCompletionChunk {
        let mut choices = Vec::new();
        let choice = Choice {
            delta: ChoiceData {
                role: pipeline.get_past_conversation().get_roles().0.clone(),
                content,
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
                        let seq = group.get_seqs().values().nth(0).unwrap();
                        if ids.contains(&seq.deref().get_id()) {
                            seq.deref_mut().set_finish_reason("abort".to_string());
                            warn!("abort sequence ({}) in subprocess!", seq.deref().get_id());
                        }
                    }
                }
                Ok(MessageType::Finish) | Ok(MessageType::Close) => {
                    warn!("A abort/finish or close message!");
                    for group in scheduled.iter() {
                        let seq = group.get_seqs().values().nth(0).unwrap();
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
        #[cfg(feature = "nccl")]
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
                    todo!();
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

            let seqs = scheduled[0].get_seqs();
            //run partial models in parallel
            let (mut logits, is_prompt) = {
                let e = engine.read();
                let (pipeline, cache_engine) = e.get_pipeline(rank).unwrap();
                let device = pipeline.device();
                let PreparedInputs {
                    tokens,
                    positions,
                    metadata,
                } = if seqs.values().nth(0).unwrap().deref().is_prompt() {
                    e.prepare_prompt(&scheduled, device)
                } else {
                    e.prepare_decode(&scheduled, device)
                }?;

                let x = pipeline.forward(
                    tokens,
                    &positions,
                    Some(&cache_engine.get_kv_cache()),
                    &metadata,
                )?;

                (x, metadata.is_prefill)
            };

            if is_prompt {
                let mut e = engine.write();
                let prefill_chunk_size = e.prefill_chunk_size.unwrap_or(PREFILL_CHUNK_SIZE);
                if prefill_chunk_size > 0 {
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
                    for s in &sample {
                        match s {
                            Either::Left(logprob) => {
                                logprobs.push(TaskSampleData::Token(logprob.clone()))
                            }
                            Either::Right(s) => {
                                logprobs.push(TaskSampleData::StopReason(s.clone()))
                            }
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
                    let mut logprobs: Vec<TokenOrFinishReason> = Vec::new();
                    match message {
                        Ok(MessageType::Sample(data)) => {
                            for s in data {
                                match s {
                                    TaskSampleData::Token(t) => {
                                        logprobs.push(TokenOrFinishReason::Left(t))
                                    }
                                    TaskSampleData::StopReason(s) => {
                                        logprobs.push(TokenOrFinishReason::Right(s))
                                    }
                                }
                            }
                            debug!("generate_once: received sample");
                            Some(logprobs)
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
            for (result_, group) in zip(results, &scheduled) {
                match result_ {
                    Either::Left(logprobs) => {
                        let seq = group.get_seqs().values().nth(0).unwrap();
                        if seq.deref().is_prompt() {
                            let e = engine.read();
                            e.scheduler.print_free_blocks();
                            prompt_finish_times.insert(*group.get_id(), SystemTime::now());
                        }
                        if let Some(sender) = &group.sender {
                            let e = engine.read();
                            let (pipeline, _) = e.get_pipeline(rank).unwrap();
                            let chunk = e.get_stream_response(
                                group.request_id.clone(),
                                group.arrival_time,
                                Some(logprobs.bytes.clone()),
                                None,
                                pipeline,
                            );
                            let ret = sender.send(ChatResponse::Chunk(chunk));
                            if ret.is_err() {
                                warn!(
                                    "Send stream response error! (sequence id {})",
                                    seq.deref().get_id()
                                );
                                seq.deref_mut().set_finish_reason("abort".to_string());
                            }
                            if seq.deref_mut().get_len() % 1000 == 0 {
                                e.scheduler.print_free_blocks();
                            }
                        };
                        seq.deref_mut().add_token(logprobs);
                    }
                    Either::Right(finish_reason) => {
                        let seq = group.get_seqs().values().nth(0).unwrap();
                        if let Some(sender) = &group.sender {
                            let e = engine.read();
                            let (pipeline, _) = e.get_pipeline(rank).unwrap();
                            let chunk = e.get_stream_response(
                                group.request_id.clone(),
                                group.arrival_time,
                                None,
                                Some(finish_reason.clone()),
                                pipeline,
                            );
                            let ret = sender.send(ChatResponse::Chunk(chunk));
                            if ret.is_err() {
                                warn!("Send stream finish response error!");
                            }
                        };
                        seq.deref_mut().set_finish_reason(finish_reason)
                    }
                }
            }

            {
                let mut e = engine.write();
                e.scheduler.free_finished_sequence_groups();
            }

            let mut aborted_sequences: Vec<usize> = Vec::new();
            for group in scheduled.iter() {
                if group.is_finished() && !responses.contains_key(&group.request_id) {
                    let end_time = SystemTime::now();
                    let prompt_finish_time = prompt_finish_times[group.get_id()];
                    let completion_time_costs = end_time
                        .duration_since(prompt_finish_time)
                        .unwrap()
                        .as_millis();
                    let seq = group.get_seqs().values().nth(0).unwrap();
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
                        seq_b
                            .deref_mut()
                            .get_cumulative_logprob()
                            .partial_cmp(&seq_a.deref_mut().get_cumulative_logprob())
                            .unwrap()
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
                            let data = outputs
                                .iter()
                                .map(|x| x.token.try_into().unwrap())
                                .collect::<Vec<_>>();
                            let pipeline = e.get_pipeline(0usize).unwrap().0.as_ref();
                            let data = pipeline.tokenizer().decode(&data, false).unwrap();
                            let choice = ChatChoice {
                                message: ChatChoiceData {
                                    role: pipeline.get_past_conversation().get_roles().0.clone(),
                                    content: Some(data),
                                },
                                finish_reason: Some(seq.deref_mut().get_finish_reason().clone()),
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
                        let seq = group.get_seqs().values().nth(0).unwrap();
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
                e.scheduler.free_finished_sequence_groups();
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
        let mut max_len = 0;
        for group in groups {
            for seq in group.get_seqs().values() {
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
        }
        let mut flat: Vec<u32> = Vec::with_capacity(groups.len() * max_len);

        for group in groups {
            for seq in group.get_seqs().values() {
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
        }

        Tensor::from_vec(flat, (groups.len(), max_len), device)
    }

    //Revised based on https://github.com/guoqingbao/vllm.rs/blob/main/src/core/runner.rs#L392
    fn prepare_prompt(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
    ) -> Result<PreparedInputs> {
        let mut context_lens = Vec::new();
        let mut input_ids: Vec<u32> = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0];
        let mut cu_seqlens_k = vec![0];
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        let mut slot_mapping = Vec::new();
        let chunk_size = self.prefill_chunk_size.unwrap_or(PREFILL_CHUNK_SIZE);
        let mut max_context_len = 0;
        for group in groups {
            for seq in group.get_seqs().values() {
                let prompt_ids = seq.deref_mut().get_token_ids();
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

                context_lens.push(seq_len as u32);

                let seqlen_q = num_tokens;
                let seqlen_k = if num_cached_tokens > 0 && cfg!(feature = "flash-decoding") {
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
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (q_len,), device)?;
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, (k_len,), device)?;

        let input_metadata = InputMetadata {
            is_prefill: true,
            slot_mapping,
            block_tables,
            context_lens,
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            max_seqlen_q,
            max_seqlen_k,
            max_context_len,
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
    ) -> Result<PreparedInputs> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut context_lens = Vec::new();
        let mut block_tables = Vec::new();
        for group in groups {
            for seq in group.get_seqs().values() {
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
        let input_metadata = InputMetadata {
            is_prefill: false,
            slot_mapping,
            block_tables: Some(block_tables),
            context_lens: Some(context_lens),
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            max_context_len: max_context_len as usize,
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
