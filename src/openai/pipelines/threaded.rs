use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};

use parking_lot::RwLock;

use super::{ChatChoice, ChatCompletionUsageResponse, LLMEngine};

struct ThreadedRunner {
    engine: Arc<RwLock<LLMEngine>>,
    rank: usize,
    responses: HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>,
    prompt_finish_times: HashMap<usize, SystemTime>,
}

impl ThreadedRunner {
    fn new(engine: Arc<RwLock<LLMEngine>>, rank: usize) -> Self {
        Self {
            engine,
            rank,
            responses: HashMap::new(),
            prompt_finish_times: HashMap::new(),
        }
    }

    fn sync_waiting_tasks(engine: &Arc<RwLock<LLMEngine>>) {
        let mut e = engine.write();
        let _ = e.move_waiting_tasks_to_scheduler();
    }

    fn run(
        mut self,
    ) -> candle_core::Result<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        {
            let e = self.engine.read();
            e.bind_rank_to_thread(self.rank);
        }

        loop {
            std::thread::sleep(Duration::from_millis(1));

            if self.rank == 0 {
                Self::sync_waiting_tasks(&self.engine);
            }

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

            if batch.is_prompt && !batch.is_embedding {
                let aborted_sequences = LLMEngine::disconnected_stream_sequence_ids(&scheduled);
                if !aborted_sequences.is_empty() {
                    let kept_indices = {
                        let mut e = self.engine.write();
                        e.abort_sequences_and_prune_scheduled(&mut scheduled, &aborted_sequences)
                    };
                    if scheduled.is_empty() {
                        self.engine.read().clear_current_scheduled_groups();
                        continue;
                    }
                    batch.logits = LLMEngine::select_logits_rows(&batch.logits, &kept_indices)?;
                }
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

            let results = {
                let mut e = self.engine.write();
                let default_pipeline = e.get_mut_pipeline(0usize).unwrap().0.as_mut();
                default_pipeline.sample(&batch.logits, &scheduled).unwrap()
            };

            self.engine.read().clear_current_scheduled_groups();

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
            let aborted = e.collect_finished_responses(
                &scheduled,
                &mut self.responses,
                &self.prompt_finish_times,
                true,
            );
            debug_assert!(aborted.is_empty());
        }

        self.engine.write().reset_decoder_for_rank(self.rank);
        tracing::debug!("generate_once: finished threaded generation");
        Ok(self.responses)
    }
}

impl LLMEngine {
    pub fn sync_threaded_waiting_tasks_before_cycle(engine: &Arc<RwLock<Self>>) -> bool {
        ThreadedRunner::sync_waiting_tasks(engine);
        false
    }

    pub fn generate_once_threaded(
        engine: Arc<RwLock<Self>>,
        rank: usize,
    ) -> candle_core::Result<HashMap<String, (Vec<ChatChoice>, ChatCompletionUsageResponse)>> {
        ThreadedRunner::new(engine, rank).run()
    }
}
