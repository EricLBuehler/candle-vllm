use std::{collections::HashMap, sync::Arc, time::SystemTime};

use candle_core::{Device, Result};
use parking_lot::RwLock;

use crate::openai::communicator::{MessageType, TaskSampleData};

use super::{
    ChatChoice, ChatCompletionUsageResponse, DaemonManager, LLMEngine, TaskData,
    TokenOrFinishReason,
};

enum SampleOutcome {
    Results(Vec<TokenOrFinishReason>),
    Continue,
    Break,
}

struct MultiprocessRunner {
    engine: Arc<RwLock<LLMEngine>>,
    rank: usize,
    responses: HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>,
    prompt_finish_times: HashMap<usize, SystemTime>,
}

impl MultiprocessRunner {
    fn new(engine: Arc<RwLock<LLMEngine>>, rank: usize) -> Self {
        Self {
            engine,
            rank,
            responses: HashMap::new(),
            prompt_finish_times: HashMap::new(),
        }
    }

    fn is_master_rank() -> bool {
        DaemonManager::is_master_rank()
    }

    fn sync_waiting_tasks_before_cycle(engine: &Arc<RwLock<LLMEngine>>) -> bool {
        if !Self::is_master_rank() {
            tracing::debug!("daemon process sync task!");
            let message = {
                let e = engine.read();
                let mut daemon_manager = e.daemon_manager.write();
                daemon_manager.as_mut().unwrap().receive_message()
            };
            return match message {
                Ok(MessageType::Continue) | Ok(MessageType::Sample(_)) => {
                    tracing::debug!("A start/continue/sample message*****!");
                    true
                }
                Ok(MessageType::Data(data)) => {
                    Self::add_remote_tasks(engine, data);
                    false
                }
                Ok(MessageType::Abort(_)) | Ok(MessageType::Finish) | Ok(MessageType::Close) => {
                    tracing::warn!("A abort/finish or close message!");
                    true
                }
                _ => {
                    tracing::warn!("Invalid message, perhaps the main process is exited!");
                    panic!("Exit process");
                }
            };
        }

        let send_tasks = {
            let mut e = engine.write();
            e.move_waiting_tasks_to_scheduler()
        };
        let num_send_tasks = send_tasks.len();
        let e = engine.read();
        let mut daemon_manager = e.daemon_manager.write();
        if num_send_tasks > 0 {
            if e.num_shards > 1 {
                tracing::warn!(
                    "Sending {} tasks to {} subprocesses",
                    num_send_tasks,
                    e.num_shards - 1
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
            false
        } else {
            let _ = daemon_manager
                .as_mut()
                .unwrap()
                .send_message(&MessageType::Continue);
            true
        }
    }

    fn add_remote_tasks(engine: &Arc<RwLock<LLMEngine>>, data: Vec<TaskData>) {
        let mut e = engine.write();
        for task in data {
            let seq_group = e.create_sequence_group(
                task.seq_id,
                task.group_id,
                &task.prompt,
                task.images.clone(),
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
            e.scheduler.add_sequence(seq_group);
        }
    }

    fn sync_waiting_tasks_during_run(&self) {
        if self.rank == 0 {
            let _ = Self::sync_waiting_tasks_before_cycle(&self.engine);
        }
    }

    fn sample_results(
        &self,
        logits: &candle_core::Tensor,
        scheduled: &std::collections::VecDeque<Arc<crate::scheduler::sequence::SequenceGroup>>,
    ) -> Result<SampleOutcome> {
        if self.rank != 0 {
            return Ok(SampleOutcome::Continue);
        }

        if self.rank == 0 && Self::is_master_rank() {
            let sample = {
                let mut e = self.engine.write();
                let default_pipeline = e.get_mut_pipeline(0usize).unwrap().0.as_mut();
                default_pipeline.sample(logits, scheduled).unwrap()
            };

            let e = self.engine.read();
            let mut daemon_manager = e.daemon_manager.write();
            let mut logprobs: Vec<TaskSampleData> = Vec::new();
            for (s, group) in sample.iter().zip(scheduled.iter()) {
                let seq_id = LLMEngine::primary_sequence_id(group);
                match s {
                    either::Either::Left(logprobs_data) => logprobs.push(TaskSampleData::Token {
                        seq_id,
                        logprobs: logprobs_data.clone(),
                    }),
                    either::Either::Right(reason) => logprobs.push(TaskSampleData::StopReason {
                        seq_id,
                        reason: reason.clone(),
                    }),
                };
            }
            let _ = daemon_manager
                .as_mut()
                .unwrap()
                .send_message(&MessageType::Sample(logprobs));
            return Ok(SampleOutcome::Results(sample));
        }

        if !Self::is_master_rank() {
            let _ = logits.to_device(&Device::Cpu).unwrap();
            let message = {
                let e = self.engine.read();
                let mut daemon_manager = e.daemon_manager.write();
                daemon_manager.as_mut().unwrap().receive_message()
            };
            return match message {
                Ok(MessageType::Sample(data)) => {
                    let mut by_seq_id = HashMap::<usize, TokenOrFinishReason>::new();
                    for s in data {
                        match s {
                            TaskSampleData::Token { seq_id, logprobs } => {
                                by_seq_id.insert(seq_id, TokenOrFinishReason::Left(logprobs));
                            }
                            TaskSampleData::StopReason { seq_id, reason } => {
                                by_seq_id.insert(seq_id, TokenOrFinishReason::Right(reason));
                            }
                        }
                    }
                    let mut ordered = Vec::<TokenOrFinishReason>::with_capacity(scheduled.len());
                    for group in scheduled {
                        let seq_id = LLMEngine::primary_sequence_id(group);
                        if let Some(sample) = by_seq_id.remove(&seq_id) {
                            ordered.push(sample);
                        } else {
                            candle_core::bail!(
                                "Missing sampled token for seq_id {} on daemon rank {}",
                                seq_id,
                                self.rank
                            );
                        }
                    }
                    if !by_seq_id.is_empty() {
                        tracing::warn!(
                            "Received {} extra sampled entries on daemon rank {}",
                            by_seq_id.len(),
                            self.rank
                        );
                    }
                    tracing::debug!("generate_once: received sample");
                    Ok(SampleOutcome::Results(ordered))
                }
                Ok(MessageType::Continue) => Ok(SampleOutcome::Continue),
                _ => {
                    tracing::info!("generate_once: received empty sample");
                    Ok(SampleOutcome::Break)
                }
            };
        }

        Ok(SampleOutcome::Continue)
    }

    fn sync_abort_sequences(
        &self,
        scheduled: &std::collections::VecDeque<Arc<crate::scheduler::sequence::SequenceGroup>>,
        aborted_sequences: Vec<usize>,
    ) {
        if self.rank != 0 {
            return;
        }
        let e = self.engine.read();
        if Self::is_master_rank() {
            if !aborted_sequences.is_empty() {
                tracing::warn!(
                    "Sending abort message ({} sequence(s)) to subprocesses!",
                    aborted_sequences.len()
                );
                let mut daemon_manager = e.daemon_manager.write();
                let _ = daemon_manager
                    .as_mut()
                    .unwrap()
                    .send_message(&MessageType::Abort(aborted_sequences));
            } else {
                let mut daemon_manager = e.daemon_manager.write();
                let _ = daemon_manager
                    .as_mut()
                    .unwrap()
                    .send_message(&MessageType::Continue);
            }
            return;
        }

        let message = {
            let mut daemon_manager = e.daemon_manager.write();
            daemon_manager.as_mut().unwrap().receive_message()
        };
        match message {
            Ok(MessageType::Abort(ids)) => {
                for group in scheduled.iter() {
                    let seq = LLMEngine::primary_sequence(group);
                    if ids.contains(&seq.deref().get_id()) {
                        seq.deref_mut().set_finish_reason("abort".to_string());
                        tracing::warn!("abort sequence ({}) in subprocess!", seq.deref().get_id());
                    }
                }
            }
            Ok(MessageType::Finish) | Ok(MessageType::Close) => {
                tracing::warn!("A abort/finish or close message!");
                for group in scheduled.iter() {
                    let seq = LLMEngine::primary_sequence(group);
                    seq.deref_mut().set_finish_reason("abort".to_string());
                    tracing::warn!(
                        "abort/finish sequence ({}) in subprocess!",
                        seq.deref().get_id()
                    );
                }
            }
            Ok(MessageType::Continue) | Ok(MessageType::Sample(_)) => {
                tracing::info!("other message!");
            }
            Ok(MessageType::Data(_)) => {
                tracing::warn!("data message found!");
            }
            _ => {
                tracing::warn!("invalid message!");
                panic!("Exit process");
            }
        }
    }

    fn send_finish_to_workers(&self) {
        if self.rank != 0 || !Self::is_master_rank() {
            return;
        }
        tracing::warn!("Sending finish message to subprocesses");
        let e = self.engine.read();
        e.scheduler.print_free_blocks();
        let mut daemon_manager = e.daemon_manager.write();
        let _ = daemon_manager
            .as_mut()
            .unwrap()
            .send_message(&MessageType::Finish);
    }

    fn run(mut self) -> Result<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        {
            let e = self.engine.read();
            e.bind_rank_to_thread(self.rank);
        }

        loop {
            self.sync_waiting_tasks_during_run();

            if !self.engine.read().has_unfinished_sequences() {
                break;
            }

            if self.rank == 0 {
                self.engine.write().schedule_current_batch(0)?;
            }

            let mut scheduled = self.engine.read().current_scheduled_groups();
            if scheduled.is_empty() {
                continue;
            }

            let mut batch = self
                .engine
                .write()
                .execute_scheduled_batch(&scheduled, self.rank)?;

            if self.rank != 0 {
                self.engine.read().clear_current_scheduled_groups();
                continue;
            }

            {
                let (embedding_done, prefill_continues) = {
                    let mut e = self.engine.write();
                    let embedding_done =
                        e.process_embedding_batch(&scheduled, &batch, self.rank)?;
                    let prefill_continues = if embedding_done {
                        false
                    } else {
                        e.process_prefill_progress(&mut scheduled, &mut batch, self.rank)?
                    };
                    (embedding_done, prefill_continues)
                };
                if embedding_done || prefill_continues {
                    continue;
                }
            }

            let sample_outcome = self.sample_results(&batch.logits, &scheduled)?;
            self.engine.read().clear_current_scheduled_groups();

            let results = match sample_outcome {
                SampleOutcome::Results(results) => results,
                SampleOutcome::Continue => continue,
                SampleOutcome::Break => break,
            };

            {
                let mut e = self.engine.write();
                e.apply_sample_results(
                    self.rank,
                    &scheduled,
                    results,
                    &mut self.prompt_finish_times,
                )?;
                e.finalize_post_sampling(
                    &scheduled,
                    self.rank,
                    &self.prompt_finish_times,
                    batch.is_prompt,
                )?;
                let aborted_sequences = e.collect_finished_responses(
                    &scheduled,
                    &mut self.responses,
                    &self.prompt_finish_times,
                    Self::is_master_rank(),
                );
                drop(e);
                self.sync_abort_sequences(&scheduled, aborted_sequences);
            }

            self.engine
                .write()
                .free_finished_sequence_groups_and_sync_mamba(self.rank);
        }

        self.engine.write().reset_decoder_for_rank(self.rank);
        self.send_finish_to_workers();
        tracing::debug!("generate_once: finished multiprocess generation");
        Ok(self.responses)
    }
}

impl LLMEngine {
    pub fn sync_multiprocess_waiting_tasks_before_cycle(engine: &Arc<RwLock<Self>>) -> bool {
        MultiprocessRunner::sync_waiting_tasks_before_cycle(engine)
    }

    pub fn generate_once_multiprocess(
        engine: Arc<RwLock<Self>>,
        rank: usize,
    ) -> Result<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        MultiprocessRunner::new(engine, rank).run()
    }
}
