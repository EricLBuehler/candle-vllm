use std::{iter::zip, path::PathBuf, sync::Arc};

use crate::{
    openai::{
        conversation::{
            default_conversation::{
                DefaultConversation, DefaultConversationSeparators, SeparatorStyle,
            },
            Conversation,
        },
        models::{
            llama::{Llama, LlamaConfig},
            ConfigLike,
        },
        requests::StopTokens,
        responses::APIError,
        sampling_params::SamplingParams,
        PipelineConfig, TokenizerWrapper,
    },
    paged_attention::input_metadata::InputMetadata,
    scheduler::sequence::Sequence,
    try_api,
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_lora_transformers::varbuilder_utils::from_mmaped_safetensors;
use either::Either::{Left, Right};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

use super::{get_token, ModelLoader, ModelPaths, ModulePipeline, TokenOrFinishReason};

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
        hf_token_path: Option<String>,
    ) -> Result<Box<dyn ModelPaths>, APIError> {
        let api = try_api!(ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(get_token(hf_token, hf_token_path)?))
            .build());
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = try_api!(api.get("tokenizer.json"));

        let config_filename = try_api!(api.get("config.json"));

        let mut filenames = vec![];
        for rfilename in try_api!(api.info())
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| x.ends_with(".safetensors"))
        {
            let filename = try_api!(api.get(&rfilename));
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
    ) -> Result<(Box<dyn ModulePipeline<'a>>, PipelineConfig), APIError> {
        let args = self.config.clone();

        let config: LlamaConfig = try_api!(serde_json::from_slice(&try_api!(std::fs::read(
            paths.get_config_filename()
        )),));
        let config = config.into_config();

        println!("Loading {} model.", self.name);

        let vb = try_api!(from_mmaped_safetensors(
            paths.get_weight_filenames(),
            dtype,
            &device,
            false
        ));

        let llama = try_api!(Llama::load(vb, &config, dtype, &device));

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
            }),
            pipeline_config,
        ))
    }
}

impl<'s> ModulePipeline<'s> for LlamaPipeline {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        mut input_metadata: InputMetadata,
    ) -> Result<Tensor, APIError> {
        self.llama.forward(
            &input_tokens,
            &input_positions,
            kv_cache,
            &mut input_metadata,
        )
    }

    fn sample(
        &mut self,
        logits: Tensor,
        sampling_params: &SamplingParams,
        seqs: &[(&usize, &Arc<Sequence>)],
    ) -> Result<Vec<TokenOrFinishReason>, APIError> {
        let eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN);

        let mut logits_processor = sampling_params.get_logits_processor(
            SAMPLING_SEED,
            &self.tokenizer,
            sampling_params.logprobs.unwrap_or(1),
        );
        let stop_tokens = match sampling_params.stop.clone() {
            Some(stop) => match stop {
                StopTokens::Multi(multi) => multi,
                StopTokens::Single(single) => vec![single],
            },

            None => vec![],
        };

        let n_seqs = logits.dims()[0];

        let mut result = Vec::new();
        for (seq_n, (_, seq)) in zip(0..n_seqs, seqs) {
            let logits = try_api!(logits.i((seq_n, try_api!(logits.dim(1)) - 1)));

            let tokens = seq
                .deref_mut()
                .get_token_ids()
                .iter()
                .map(|x| *x as u32)
                .collect::<Vec<_>>();
            let tokens_generated = seq.deref_mut().get_len() - seq.deref_mut().get_prompt_len();

            let logits = if sampling_params.repetition_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.args.repeat_last_n);
                try_api!(candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    sampling_params.repetition_penalty,
                    &tokens[start_at..],
                ))
            };

            let next_token = try_api!(logits_processor.sample(&logits));
            if let Some(text) = self.tokenizer.id_to_token(next_token.token as u32) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                if stop_tokens.contains(&text) {
                    result.push(Right("stop".to_string()));
                    continue;
                }
            }

            if Some(next_token.token) == eos_token_id.map(|x| x as usize) {
                result.push(Right("stop".to_string()));
                continue;
            }
            if tokens_generated >= sampling_params.max_tokens {
                result.push(Right("length".to_string()));
                continue;
            }
            result.push(Left(next_token));
        }

        Ok(result)
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

    fn get_model_config(&self) -> Box<dyn ConfigLike> {
        Box::new(self.llama.get_config().clone())
    }

    fn get_dtype(&self) -> DType {
        todo!()
    }
}

unsafe impl Send for LlamaPipeline {}
unsafe impl Sync for LlamaPipeline {}
