use std::rc::Rc;

use tokenizers::Encoding;

use crate::{
    openai::utils::get_created_time_secs,
    scheduler::{
        cache_engine::CacheConfig,
        scheduler::{Scheduler, SchedulerConfig},
        sequence::{Sequence, SequenceGroup},
    },
};

use super::ModulePipeline;

pub struct LLMEngine<'a> {
    pipeline: Box<dyn ModulePipeline<'a>>,
    scheduler: Scheduler,
    seq_id: usize,
    cache_config: CacheConfig,
    group_id: usize,
}

impl<'a> LLMEngine<'a> {
    pub fn new(
        pipeline: Box<dyn ModulePipeline<'a>>,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> Self {
        Self {
            pipeline,
            scheduler: Scheduler::new(scheduler_config, &cache_config),
            seq_id: 0,
            cache_config,
            group_id: 0,
        }
    }

    pub fn add_request(&mut self, prompt: Encoding) {
        let seq = Rc::new(Sequence::new(
            prompt
                .get_ids()
                .to_vec()
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<_>>(),
            self.seq_id,
            self.cache_config.block_size,
        ));
        self.seq_id += 1;
        let seq_group = SequenceGroup::new(&vec![seq], get_created_time_secs(), self.group_id);
        self.group_id += 1;

        self.scheduler.add_sequence(seq_group);
    }
}
