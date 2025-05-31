use super::{DefaultPipeline, _make_tensor_with_pad};
#[cfg(feature = "nccl")]
use crate::openai::communicator::{DaemonManager, MessageType, TaskData, TaskSampleData};
#[cfg(feature = "nccl")]
use crate::openai::pipelines::TokenOrFinishReason;
use crate::openai::streaming::ChatResponse;
use crate::scheduler::Scheduler;
use crate::{
    openai::{
        models::Config,
        responses::{
            APIError, ChatChoice, ChatChoiceData, ChatCompletionChunk, ChatCompletionUsageResponse,
            Choice, ChoiceData, WrapperLogprobs,
        },
        sampling_params::SamplingParams,
        utils::get_created_time_secs,
    },
    paged_attention::input_metadata::InputMetadata,
    scheduler::{
        cache_engine::{CacheConfig, CacheEngine},
        sequence::{Sequence, SequenceGroup, _Sequence},
        SchedulerConfig, SchedulerOutput,
    },
    try_api,
};
use candle_core::{Device, Tensor};
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
use tokenizers::Encoding;
use tokio::sync::Notify;
use tracing::warn;
#[cfg(feature = "nccl")]
use tracing::{debug, info};
#[allow(dead_code)]
struct PreparedInputs {
    tokens: Tensor,
    positions: Vec<Vec<usize>>,
    metadata: InputMetadata,
}

const _PAD_SLOT_ID: i64 = -1;

pub struct LLMEngine {
    pipelines: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)>,
    scheduler: Scheduler,
    seq_id: usize,
    cache_config: CacheConfig,
    config: Config,
    group_id: usize,
    pub notify: Arc<Notify>,
    sync_notifies: HashMap<String, Option<Arc<Notify>>>,
    pub completion_records: HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>,
    sequence_groups: RwLock<VecDeque<Arc<SequenceGroup>>>,
    multi_process: bool,
    #[cfg(feature = "nccl")]
    cur_tasks: RwLock<Vec<TaskData>>,
    #[cfg(feature = "nccl")]
    pub daemon_manager: RwLock<Option<DaemonManager>>,
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
    ) -> Result<Arc<RwLock<Self>>, APIError> {
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
            #[cfg(feature = "nccl")]
            cur_tasks: RwLock::new(Vec::<TaskData>::new()),
            #[cfg(feature = "nccl")]
            daemon_manager: RwLock::new(daemon_manager),
            sync_notifies: HashMap::new(),
        }));
        let engine_clone = engine.clone();

        let mut ranks = Vec::<usize>::new();
        for rank in 0..num_threads {
            ranks.push(rank);
        }

        let _ = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                loop {
                    #[cfg(feature = "nccl")]
                    if multi_process {
                        if !DaemonManager::is_master_rank() {
                            info!("daemon process wait_task!");
                            let message = {
                                let e = engine.read();
                                let mut daemon_manager = e.daemon_manager.write();
                                daemon_manager.as_mut().unwrap().receive_message()
                            };
                            match message {
                                Ok(MessageType::Start) => {
                                    info!("A start message*****!");
                                }
                                Ok(MessageType::Continue) => {
                                    info!("A continue message before start!");
                                }
                                Ok(MessageType::Data(data)) => {
                                    let mut e = engine.write();
                                    for task in data {
                                        warn!("Add request {} to task list!", task.request_id);
                                        e.add_request(task.prompt, task.request_id, task.created, task.sampling_params, task.use_logprobs, None, None.into());
                                    }
                                    continue;
                                }
                                Ok(MessageType::Sample(_)) => {
                                    info!("A sample message before start!");
                                    continue;
                                }
                                Ok(MessageType::Abort(_)) | Ok(MessageType::Finish) | Ok(MessageType::Close) => {
                                    info!("A abort/finish or close message before start!");
                                    continue;
                                }
                                _=> {
                                    warn!("Invalid message, perhaps the main process is exited!");
                                    panic!("Exit process");
                                }
                            };

                        } else {
                            notify.notified().await;
                            let _ = tokio::time::sleep(tokio::time::Duration::from_millis(holding_time as u64)).await;
                            {
                                let e = engine.read();
                                let mut daemon_manager = e.daemon_manager.write();
                                let mut cur_tasks = e.cur_tasks.write();
                                let send_tasks = cur_tasks.clone();
                                if send_tasks.len() < 1 {
                                    continue
                                }
                                warn!("Sending {} tasks to {} subprocesses", send_tasks.len(), num_shards - 1);
                                let _ = daemon_manager.as_mut().unwrap().send_message(&MessageType::Data(send_tasks));
                                cur_tasks.clear();
                                let _ = daemon_manager.as_mut().unwrap().send_message(&MessageType::Start);
                            }
                        }
                    }
                    if !multi_process {
                        notify.notified().await; // Blocking call to wait for notification
                        let _ = tokio::time::sleep(tokio::time::Duration::from_millis(holding_time as u64)).await;
                    }
                    let results = Self::generate_parallel(&engine, ranks.clone(), multi_process).await;

                    #[cfg(feature = "nccl")]
                    if multi_process && !DaemonManager::is_master_rank() {
                        continue;
                    }
                    let result = &results[0];
                    if results.len() == 0 || result.len() == 0 {
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

                    println!(
                        "\r\n [{} requests] Prefilling: {} prompt tokens processed in {} seconds",
                        result.len(),
                        overall_usage.prompt_tokens,
                        overall_usage.prompt_time_costs / 1000
                    );
                    println!(
                        "\r\n [{} requests] Decoding: {} tokens processed in {} seconds ({:.02} tokens/s)",
                        result.len(),
                        overall_usage.completion_tokens,
                        overall_usage.completion_time_costs / 1000,
                        overall_usage.completion_tokens as f32 * 1000.0
                            / if overall_usage.completion_time_costs > 0 {
                                overall_usage.completion_time_costs as f32
                            } else {
                                1f32
                            }
                    );
                }
            });
        });

        Ok(engine_clone)
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
    pub fn sync_process(engine: &Arc<RwLock<Self>>, msg_send: MessageType) -> Option<Vec<usize>> {
        let mut e = engine.write();
        if !DaemonManager::is_master_rank() {
            debug!("waiting sync message!");
            let message = {
                let mut daemon_manager = e.daemon_manager.write();
                daemon_manager.as_mut().unwrap().receive_message()
            };
            match message {
                Ok(MessageType::Abort(ids)) => {
                    return Some(ids);
                }
                Ok(MessageType::Finish) | Ok(MessageType::Close) => {
                    warn!("A abort/finish or close message!");
                    return Some(Vec::<usize>::new());
                }
                Ok(MessageType::Continue) => {
                    debug!("continue message!");
                }
                Ok(MessageType::Data(data)) => {
                    for task in data {
                        warn!(
                            "Add request {} to task list (continuous batching)!",
                            task.request_id
                        );
                        e.add_request(
                            task.prompt,
                            task.request_id,
                            task.created,
                            task.sampling_params,
                            task.use_logprobs,
                            None,
                            None.into(),
                        );
                    }
                }
                Ok(MessageType::Start) | Ok(MessageType::Sample(_)) => {
                    info!("other message!");
                }
                _ => {
                    warn!("invalid message!");
                    panic!("Exit process")
                }
            };
        } else {
            debug!("sending sync message!");
            let _ = {
                let mut daemon_manager = e.daemon_manager.write();
                daemon_manager.as_mut().unwrap().send_message(&msg_send)
            };
        }
        return None;
    }

    pub fn generate_once(
        engine: Arc<RwLock<Self>>,
        rank: usize,
        multi_process: bool,
    ) -> Result<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>, APIError> {
        let mut responses =
            HashMap::<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>::new();
        let mut prompt_finish_times = HashMap::<usize, SystemTime>::new();
        #[cfg(feature = "nccl")]
        {
            warn!("Start processing...");
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
                    warn!("generate_once: no unfinished_sequences, break");
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

            let scheduled: VecDeque<Arc<SequenceGroup>> = {
                let e = engine.read();
                let x = e.sequence_groups.read();
                x.clone()
            };
            if scheduled.len() == 0 {
                continue; //data not ready
            }

            let seqs = scheduled[0].get_seqs();
            //run partial models in parallel
            let logits = {
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

                x
            };

            #[cfg(feature = "nccl")]
            let do_sample = if rank == 0 && DaemonManager::is_master_rank() {
                true
            } else {
                false
            };
            #[cfg(not(feature = "nccl"))]
            let do_sample = if rank == 0 { true } else { false };

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
                                &pipeline,
                            );
                            let ret = sender.send(ChatResponse::Chunk(chunk));
                            if ret.is_err() {
                                println!(
                                    "Send stream response error! (sequence id {})",
                                    seq.deref().get_id()
                                );
                                seq.deref_mut().set_finish_reason("abort".to_string());
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
                                &pipeline,
                            );
                            let ret = sender.send(ChatResponse::Chunk(chunk));
                            if ret.is_err() {
                                println!("Send stream finish response error!");
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
                        println!(
                            "Request {} decoding {} tokens finished in {} seconds",
                            group.request_id,
                            decoded_tokens,
                            completion_time_costs / 1000
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
                            if notify.is_some() {
                                notify.unwrap().as_ref().unwrap().notify_one();
                            }
                        }
                    }
                    if let Some(sender) = &group.sender {
                        let seq = group.get_seqs().values().nth(0).unwrap();
                        if seq.deref().get_finish_reason() != "abort" {
                            warn!(
                                "Sending completion message to client! (sequence id {})",
                                seq.deref().get_id()
                            );
                            let _ = sender.send(ChatResponse::Done);
                        } else {
                            aborted_sequences.push(seq.deref().get_id());
                        }
                    };
                }
            }

            #[cfg(feature = "nccl")]
            if multi_process {
                if DaemonManager::is_master_rank() {
                    let e = engine.write();
                    let mut daemon_manager = e.daemon_manager.write();
                    if aborted_sequences.len() > 0 {
                        warn!(
                            "Sending abort message ({} sequence(s)) to subprocesses!",
                            aborted_sequences.len()
                        );
                        let _ = daemon_manager
                            .as_mut()
                            .unwrap()
                            .send_message(&MessageType::Abort(aborted_sequences));
                    } else {
                        let mut cur_tasks = e.cur_tasks.write();
                        let send_tasks = cur_tasks.clone();
                        if send_tasks.len() > 0 {
                            warn!(
                                "Sending {} tasks to subprocesses (continuous batching)",
                                send_tasks.len()
                            );
                            let _ = daemon_manager
                                .as_mut()
                                .unwrap()
                                .send_message(&MessageType::Data(send_tasks));
                            cur_tasks.clear();
                        } else {
                            let _ = daemon_manager
                                .as_mut()
                                .unwrap()
                                .send_message(&MessageType::Continue);
                        }
                    }
                } else {
                    let has_unfinished_sequences = {
                        let e = engine.read();
                        e.scheduler.has_unfinished_sequences()
                    };
                    if has_unfinished_sequences {
                        match Self::sync_process(&engine, MessageType::Continue) {
                            Some(ids) => {
                                for group in scheduled.iter() {
                                    let seq = group.get_seqs().values().nth(0).unwrap();
                                    if ids.is_empty() || ids.contains(&seq.deref().get_id()) {
                                        seq.deref_mut().set_finish_reason("abort".to_string());
                                        warn!(
                                            "abort sequence ({}) in subprocess!",
                                            seq.deref().get_id()
                                        );
                                    }
                                }
                            }
                            _ => {
                                debug!("sync continue message -> continue!");
                            }
                        }
                    }
                }
                {
                    let mut e = engine.write();
                    e.scheduler.free_finished_sequence_groups();
                }
            };
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
            let mut daemon_manager = e.daemon_manager.write();
            let _ = daemon_manager
                .as_mut()
                .unwrap()
                .send_message(&MessageType::Finish);
        }

        warn!("generate_once: finished generation");
        Ok(responses)
    }
}

impl LLMEngine {
    fn execute_scheduler_ops(
        &mut self,
        scheduler_output: &SchedulerOutput,
        rank: usize,
    ) -> Result<(), APIError> {
        let cache_engine = Box::new(&mut self.get_mut_pipeline(rank).unwrap().1);
        if !scheduler_output.blocks_to_swap_in.is_empty() {
            try_api!(cache_engine.swap_in(scheduler_output.blocks_to_swap_in.clone()));
        }
        if !scheduler_output.blocks_to_swap_out.is_empty() {
            try_api!(cache_engine.swap_out(scheduler_output.blocks_to_swap_out.clone()));
        }
        if !scheduler_output.blocks_to_copy.is_empty() {
            try_api!(cache_engine.copy(scheduler_output.blocks_to_copy.clone()));
        }
        Ok(())
    }

    fn prepare_prompt(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
    ) -> Result<PreparedInputs, APIError> {
        let mut prompt_lens = Vec::new();
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut slot_mappings = Vec::new();
        for group in groups {
            for seq in group.get_seqs().values() {
                let prompt_ids = seq.deref_mut().get_token_ids();

                let prompt_len = prompt_ids.len();
                prompt_lens.push(prompt_len);

                input_tokens.push(prompt_ids);
                input_positions.push((0..prompt_len).collect::<Vec<_>>());
                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref_mut().get_id());
                if table.is_none() {
                    // Will be None during profiling.
                    slot_mappings.push([_PAD_SLOT_ID].repeat(prompt_len));
                    continue;
                }
                let table = table
                    .unwrap()
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let start_idx = if let Some(sliding_window) = self.config.sliding_window {
                    if prompt_len > sliding_window {
                        0.min(prompt_len - sliding_window)
                    } else {
                        0
                    }
                } else {
                    0
                };

                let mut slot_mapping = Vec::new();
                for i in 0..prompt_len {
                    if i < start_idx {
                        // Pad [0,start_idx) with _PAD_TOKEN_ID
                        slot_mapping.push(_PAD_SLOT_ID);
                    }

                    let block_number = if i / self.cache_config.block_size >= table.len() {
                        panic!(
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
                    slot_mapping.push(slot.try_into().unwrap());
                }
                slot_mappings.push(slot_mapping);
            }
        }

        let max_prompt_len = prompt_lens.iter().max().unwrap();
        let input_tokens = _make_tensor_with_pad(
            input_tokens
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            *max_prompt_len,
            0,
            device,
        )?;
        let slot_mapping =
            _make_tensor_with_pad(slot_mappings, *max_prompt_len, _PAD_SLOT_ID, device)?;

        Ok(PreparedInputs {
            tokens: input_tokens,
            positions: input_positions,
            metadata: InputMetadata {
                prompt_lens,
                slot_mapping,
                max_context_len: None,
                context_lens: None,
                block_tables: None,
                is_prompt: true,
                kv_cache_dtype: "auto".to_string(), // TODO(EricLBuehler): specialize for models
            },
        })
    }

    fn prepare_decode(
        &self,
        groups: &VecDeque<Arc<SequenceGroup>>,
        device: &Device,
    ) -> Result<PreparedInputs, APIError> {
        let mut input_tokens = Vec::new();
        let mut input_positions = Vec::new();
        let mut context_lens = Vec::new();
        let mut slot_mappings = Vec::new();
        let mut block_tables = Vec::new();
        for group in groups {
            for seq in group.get_seqs().values() {
                let last_token_id = seq.deref_mut().get_last_token_id();
                input_tokens.push(vec![last_token_id]);

                let position = seq.deref_mut().get_len() - 1;
                input_positions.push(vec![position]);

                let context_len = if let Some(sliding_window) = self.config.sliding_window {
                    seq.deref_mut().get_len().min(sliding_window)
                } else {
                    seq.deref_mut().get_len()
                };
                context_lens.push(context_len);

                let table = self
                    .scheduler
                    .block_engine
                    .block_tables
                    .get(&seq.deref_mut().get_id())
                    .unwrap();
                let table = table
                    .iter()
                    .map(|block| block.deref_mut().block_id)
                    .collect::<Vec<_>>();

                let block_number = if position / self.cache_config.block_size >= table.len() {
                    panic!("Block table is too small (completion)! start_pos={} block_size={} table_len={}", position, self.cache_config.block_size, table.len());
                } else {
                    table.get(position / self.cache_config.block_size).unwrap()
                };
                let block_offset = position % self.cache_config.block_size;
                let slot = block_number * self.cache_config.block_size + block_offset;
                let slot = slot.try_into().unwrap();
                slot_mappings.push(vec![slot]);

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

        let input_tokens = _make_tensor_with_pad(
            input_tokens
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            1,
            0,
            device,
        )?;
        let slot_mapping = _make_tensor_with_pad(slot_mappings, 1, _PAD_SLOT_ID, device)?;

        let max_context_len = context_lens.iter().max().unwrap();
        let context_lens = try_api!(Tensor::from_vec(
            context_lens.iter().map(|x| *x as u32).collect::<Vec<_>>(),
            (context_lens.len(),),
            device,
        ));

        let max_block_table_len = block_tables.iter().map(|x| x.len()).max().unwrap();
        let block_tables = _make_tensor_with_pad(
            block_tables
                .iter()
                .map(|x| x.iter().map(|x| *x as u32).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            max_block_table_len,
            0,
            device,
        )?;
        let block_tables = try_api!(block_tables.reshape(((), max_block_table_len)));
        Ok(PreparedInputs {
            tokens: input_tokens,
            positions: input_positions,
            metadata: InputMetadata {
                prompt_lens: vec![],
                slot_mapping,
                max_context_len: Some(*max_context_len),
                context_lens: Some(context_lens),
                block_tables: Some(block_tables),
                is_prompt: false,
                kv_cache_dtype: "auto".to_string(), // TODO(EricLBuehler): specialize for models
            },
        })
    }

    pub fn add_request(
        &mut self,
        prompt: Encoding,
        request_id: String,
        created: SystemTime,
        sampling_params: SamplingParams,
        use_logprobs: bool,
        sender: Option<Sender<ChatResponse>>,
        sync_notify: Option<Arc<Notify>>,
    ) {
        let prompt_len = prompt.get_ids().len();
        let seq = Arc::new(Sequence(std::sync::RwLock::new(_Sequence::new(
            prompt
                .get_ids()
                .to_vec()
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>(),
            self.seq_id,
            self.cache_config.block_size,
        ))));
        self.seq_id += 1;
        let seq_group = SequenceGroup::new(
            &[seq],
            get_created_time_secs(),
            self.group_id,
            request_id.clone(),
            created,
            sampling_params.clone(),
            use_logprobs,
            sender,
        );
        self.group_id += 1;

        self.scheduler.add_sequence(seq_group);
        let sync_notify = sync_notify.clone();
        if sync_notify.is_some() {
            self.sync_notifies
                .insert(request_id.clone(), Some(sync_notify.unwrap()));
        }
        #[cfg(feature = "nccl")]
        let do_log = DaemonManager::is_master_rank();
        #[cfg(not(feature = "nccl"))]
        let do_log = true;
        if do_log {
            println!(
                "Request {} with length {} added to sequence group.",
                request_id.clone(),
                prompt_len
            );
        }

        #[cfg(feature = "nccl")]
        if self.multi_process && !DaemonManager::is_daemon() {
            info!("add task to list");
            let task = TaskData {
                prompt,
                request_id,
                created,
                sampling_params,
                use_logprobs,
            };
            let mut cur_tasks = self.cur_tasks.write();
            cur_tasks.push(task);
        }
    }
}
