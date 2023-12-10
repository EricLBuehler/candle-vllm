use std::{collections::HashMap, iter, path::PathBuf, sync::Arc};

use crate::{
    openai::{
        conversation::{
            default_conversation::{
                DefaultConversation, DefaultConversationSeparators, SeparatorStyle,
            },
            Conversation,
        },
        models::{
            llama::{Cache, Config, Llama, LlamaConfig},
            ConfigLike,
        },
        requests::StopTokens,
        responses::{
            APIError, ChatChoice, ChatChoiceData, ChatCompletionUsageResponse,
            StreamingChatCompletionResponse, StreamingChoice, StreamingChoiceData,
        },
        sampling_params::{EarlyStoppingCondition, SamplingParams},
        streaming::SenderError,
        utils::get_created_time_secs,
        PipelineConfig, TokenizerWrapper,
    },
    paged_attention::{
        cache_engine::{CacheConfig, CacheEngine},
        input_metadata::InputMetadata,
        scheduler::{Scheduler, SchedulerConfig},
        sequence::{SequenceData, SequenceGroupMetadata},
    },
};
use actix_web::web::Bytes;
use candle_core::{DType, Device, Tensor};
use candle_lora_transformers::varbuilder_utils::from_mmaped_safetensors;
use candle_sampling::logits_processor::{LogitsProcessor, SamplingMethod};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

use super::{read_env_var, ModelLoader, ModelPaths, ModulePipeline};

const EOS_TOKEN: &str = "</s>";
const SAMPLING_SEED: u64 = 299792458;

#[derive(Debug, Clone)]
pub struct LlamaSpecificConfig {
    repeat_last_n: usize,
}

impl LlamaSpecificConfig {
    pub fn new(repeat_last_n: usize) -> Self {
        Self { repeat_last_n }
    }
}

/// top-p, multinomial, and argmax sampling are implemented. Beam search is not implemented.
pub struct LlamaPipeline {
    llama: Llama,
    args: LlamaSpecificConfig,
    tokenizer: Tokenizer,
    conversation: DefaultConversation,
    name: String,
    input_metadata: InputMetadata,
    scheduler: Scheduler,
    block_size: Option<usize>,
}

pub struct LlamaLoader {
    config: LlamaSpecificConfig,
    name: String,
}

pub struct LlamaModelPaths<P> {
    tokenizer_filename: P,
    config_filename: P,
    filenames: Vec<P>,
}

impl ModelPaths for LlamaModelPaths<PathBuf> {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config_filename
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        &self.tokenizer_filename
    }
    fn get_weight_filenames(&self) -> &Vec<PathBuf> {
        &self.filenames
    }
}

impl LlamaLoader {
    pub fn new(config: LlamaSpecificConfig, name: String) -> Self {
        Self { config, name }
    }
}

impl<'a> ModelLoader<'a> for LlamaLoader {
    fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<String>,
    ) -> Result<Box<dyn ModelPaths>, APIError> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(read_env_var(hf_token.unwrap())?))
            .build()
            .map_err(APIError::from)?;
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api.get("tokenizer.json").map_err(APIError::from)?;

        let config_filename = api.get("config.json").map_err(APIError::from)?;

        let mut filenames = vec![];
        for rfilename in api
            .info()
            .map_err(APIError::from)?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| x.ends_with(".safetensors"))
        {
            let filename = api.get(&rfilename).map_err(APIError::from)?;
            filenames.push(filename);
        }

        Ok(Box::new(LlamaModelPaths {
            tokenizer_filename,
            config_filename,
            filenames,
        }))
    }

    fn load_model(
        &self,
        paths: Box<dyn ModelPaths>,
        dtype: DType,
        device: Device,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> Result<(Box<dyn ModulePipeline<'a>>, PipelineConfig), APIError> {
        let args = self.config.clone();

        let config: LlamaConfig = serde_json::from_slice(
            &std::fs::read(paths.get_config_filename()).map_err(APIError::from)?,
        )
        .map_err(APIError::from)?;
        let config = config.into_config();

        println!("Loading {} model.", self.name);

        let vb = from_mmaped_safetensors(paths.get_weight_filenames(), dtype, &device, false)
            .map_err(APIError::from)?;

        let cache = Cache::new(dtype, &config, &device).map_err(APIError::from)?;

        let llama = Llama::load(vb, &cache, &config).map_err(APIError::from)?;

        let tokenizer = Tokenizer::from_file(paths.get_tokenizer_filename())
            .map_err(|x| APIError::new(x.to_string()))?;

        println!("Done loading.");

        //max is https://huggingface.co/docs/transformers/model_doc/llama2#transformers.LlamaConfig.max_position_embeddings
        let pipeline_config = PipelineConfig {
            max_model_len: 4096,
        };

        //reference: https://huggingface.co/blog/codellama#conversational-instructions,
        //reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
        Ok((
            Box::new(LlamaPipeline {
                llama,
                args,
                tokenizer,
                conversation: DefaultConversation::new(
                    "llama-2".to_string(),
                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n".to_string(),
                    Vec::default(),
                    0,
                    SeparatorStyle::Llama2,
                    "".to_string(),
                    Vec::default(),
                    ("[INST]".to_string(), "[/INST]".to_string()),
                    DefaultConversationSeparators {
                        sep: " ".to_string(),
                        sep2: Some(" </s></s>".to_string()),
                    },
                ),
                name: self.name.clone(),
                input_metadata: InputMetadata::new(todo!(), None, None, None, todo!()),
                scheduler: Scheduler::new(scheduler_config, cache_config)?,
                block_size: None,
            }),
            pipeline_config,
        ))
    }
}

impl LlamaPipeline {
    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &mut self,
        input_tokens: Tensor,
        _input_positions: Tensor,
        kv_cache: Option<Arc<Vec<(Tensor, Tensor)>>>,
        input_metadata: InputMetadata,
    ) -> Result<(Option<ChatChoice>, ChatCompletionUsageResponse), APIError> {
        let mut index_pos = 0;
        let mut index = 0;
        let mut result = "".to_string();
        let mut tokens_generated = 0;
        let finish_reason;

        loop {
            let context_size = if index > 0 { 1 } else { tokens.len() };

            let logits =
                self.llama
                    .forward(&input_tokens, index_pos, kv_cache, &mut self.input_metadata)?;

            tokens.push(next_token);

            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                if stop_tokens.contains(&text) {
                    finish_reason = "stop".to_string();
                    break;
                }
                result.push_str(&text);
            }

            if &Some(next_token) == eos_token_id {
                finish_reason = "stop".to_string();
                break;
            }
            if tokens_generated >= sampling.max_tokens {
                finish_reason = "length".to_string();
                break;
            }

            index += 1;
        }

        Ok((
            Some(ChatChoice {
                message: ChatChoiceData {
                    content: Some(result),
                    role: self.conversation.get_roles().1.clone(),
                },
                finish_reason: Some(finish_reason),
                index: gen_index,
            }),
            ChatCompletionUsageResponse {
                completion_tokens: tokens_generated,
                prompt_tokens: tokens.len(),
                total_tokens: tokens_generated + tokens.len(),
            },
        ))
    }
}

impl<'s> ModulePipeline<'s> for LlamaPipeline {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<Arc<Vec<(Tensor, Tensor)>>>,
        input_metadata: InputMetadata,
    ) -> Result<(Option<Vec<ChatChoice>>, ChatCompletionUsageResponse), APIError> {
        todo!()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String> {
        &self.tokenizer
    }

    fn get_conversation(&mut self) -> &mut dyn Conversation {
        &mut self.conversation
    }

    fn get_scheduler_config(&self) -> &SchedulerConfig {
        self.scheduler.get_config()
    }

    fn get_model_config(&self) -> Box<dyn ConfigLike> {
        Box::new(self.llama.get_config().clone())
    }

    fn _get_block_size(&self) -> &Option<usize> {
        &self.block_size
    }

    fn set_block_size(&mut self, size: usize) {
        self.block_size = Some(size)
    }
}

unsafe impl Send for LlamaPipeline {}
unsafe impl Sync for LlamaPipeline {}
