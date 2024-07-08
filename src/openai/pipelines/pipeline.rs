use super::{get_token, ModelLoader, ModelPaths, ModulePipeline, TokenOrFinishReason};
use crate::openai::sampling_params::{Logprobs, TopLogprob};
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
            phi3::{Phi, PhiConfig},
            qwen2::{Qwen2, QwenConfig},
            Config,
        },
        responses::APIError,
        sampling_params::SamplingParams,
        PipelineConfig,
    },
    paged_attention::input_metadata::InputMetadata,
    scheduler::sequence::Sequence,
    try_api,
};
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use either::Either::{Left, Right};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::{iter::zip, path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;
const EOS_TOKEN: &str = "</s>";
const SAMPLING_SEED: u64 = 299792458;
const MIN_GEN_TOKENS: usize = 128;
const MAX_GEN_TOKENS: usize = 4096;

#[derive(Debug, Clone)]
pub struct SpecificConfig {
    repeat_last_n: usize,
}

impl SpecificConfig {
    pub fn new(repeat_last_n: usize) -> Self {
        Self { repeat_last_n }
    }
}

enum LLMModel {
    LLAMA(Llama),
    Phi3(Phi),
    Qwen2(Qwen2),
}
/// top-p, multinomial, and argmax sampling are implemented. Beam search is not implemented.
pub struct DefaultPipeline {
    model: LLMModel,
    args: SpecificConfig,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    conversation: DefaultConversation,
    name: String,
    dtype: DType,
    device: Device,
    cur_idx: usize,
    config: Config,
    stop_token_id: u32,
}

pub struct DefaultLoader {
    config: SpecificConfig,
    name: String,
}

pub struct DefaultModelPaths<P> {
    pub tokenizer_filename: P,
    pub config_filename: P,
    pub filenames: Vec<P>,
}

impl ModelPaths for DefaultModelPaths<PathBuf> {
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

impl DefaultLoader {
    pub fn new(config: SpecificConfig, name: String) -> Self {
        Self { config, name }
    }
}

impl<'a> ModelLoader<'a> for DefaultLoader {
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

        Ok(Box::new(DefaultModelPaths {
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

        let config = match self.name.as_str() {
            "llama" => {
                let config: LlamaConfig = try_api!(serde_json::from_slice(&try_api!(
                    std::fs::read(paths.get_config_filename())
                ),));
                config.into_config(false)
            }
            "phi3" => {
                let config: PhiConfig = try_api!(serde_json::from_slice(&try_api!(std::fs::read(
                    paths.get_config_filename()
                )),));
                config.into_config(false)
            }
            "qwen2" => {
                let config: QwenConfig = try_api!(serde_json::from_slice(&try_api!(
                    std::fs::read(paths.get_config_filename())
                ),));
                config.into_config(false)
            }
            _ => panic!(""),
        };

        println!("Model {:?}", config);

        println!("Loading {} model.", self.name);

        let vb = match unsafe {
            VarBuilder::from_mmaped_safetensors(&paths.get_weight_filenames(), dtype, &device)
        } {
            Ok(vb_) => vb_,
            _ => panic!("Load model weights failed!"),
        };

        let (model, sep_style) = match self.name.as_str() {
            "llama" => (
                LLMModel::LLAMA(try_api!(Llama::load(vb, &config, dtype, &device))),
                SeparatorStyle::Llama,
            ),
            "phi3" => (
                LLMModel::Phi3(try_api!(Phi::new(vb, &config, dtype, &device))),
                SeparatorStyle::Phi,
            ),
            "qwen2" => (
                LLMModel::Qwen2(try_api!(Qwen2::new(vb, &config, dtype, &device))),
                SeparatorStyle::Qwen2,
            ),
            _ => panic!(""),
        };

        let tokenizer_ = Tokenizer::from_file(paths.get_tokenizer_filename())
            .map_err(|x| APIError::new(x.to_string()))?;

        let tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer_);

        println!("Done loading.");

        //max and min number of tokens generated per request
        let mut default_max_tokens = config.max_seq_len / 10;
        if default_max_tokens < MIN_GEN_TOKENS {
            default_max_tokens = MIN_GEN_TOKENS;
        } else if default_max_tokens > MAX_GEN_TOKENS {
            default_max_tokens = MAX_GEN_TOKENS;
        }

        let pipeline_config = PipelineConfig {
            max_model_len: config.max_seq_len,
            default_max_tokens,
        };

        println!("{:?}", pipeline_config);

        let eos_token = match tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => tokenizer.tokenizer().token_to_id(EOS_TOKEN).unwrap(),
        };

        let eos_token = match tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => tokenizer.tokenizer().token_to_id(EOS_TOKEN).unwrap(),
        };

        //reference: https://huggingface.co/blog/codellama#conversational-instructions,
        //reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
        Ok((
            Box::new(DefaultPipeline {
                model,
                args,
                tokenizer,
                logits_processor: LogitsProcessor::new(SAMPLING_SEED, None, None),
                conversation: DefaultConversation::new(
                    self.name.to_string(),
                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n [/INST]".to_string(),
                    Vec::default(),
                    0,
                    sep_style,
                    "".to_string(),
                    vec![eos_token as usize],
                    ("user".to_string(), "assistant".to_string()),
                    DefaultConversationSeparators {
                        sep: " ".to_string(),
                        sep2: Some(" </s></s>".to_string()),
                    },
                ),
                name: self.name.clone(),
                dtype,
                device: device.clone(),
                cur_idx: 0,
                config: config,
                stop_token_id: eos_token,
            }),
            pipeline_config,
        ))
    }
}

impl<'s> ModulePipeline<'s> for DefaultPipeline {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        _input_positions: Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        mut input_metadata: InputMetadata,
    ) -> Result<Tensor, APIError> {
        let length = input_tokens.shape().dims()[0];
        if length > 1 {
            self.cur_idx = 0;
        }
        let ret = match &mut self.model {
            LLMModel::LLAMA(llama) => llama
                .forward(
                    &input_tokens
                        .reshape((1, input_tokens.shape().dims()[0]))
                        .unwrap(),
                    self.cur_idx,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::Phi3(phi) => phi
                .forward(
                    &input_tokens
                        .reshape((1, input_tokens.shape().dims()[0]))
                        .unwrap(),
                    self.cur_idx,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::Qwen2(qwen2) => qwen2
                .forward(
                    &input_tokens
                        .reshape((1, input_tokens.shape().dims()[0]))
                        .unwrap(),
                    self.cur_idx,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
        };

        self.cur_idx += length;
        return ret;
    }

    fn sample(
        &mut self,
        logits: Tensor,
        sampling_params: &SamplingParams,
        seqs: &[(&usize, &Arc<Sequence>)],
    ) -> Result<Vec<TokenOrFinishReason>, APIError> {
        let eos_token_id = self
            .config
            .eos_token_id
            .or_else(|| Some(self.stop_token_id));

        let n_seqs = logits.dims()[0];

        let mut result = Vec::new();
        for (_, (_, seq)) in zip(0..n_seqs, seqs) {
            let logits = logits.squeeze(0).unwrap();
            let sq = seq.deref_mut();
            let tokens = sq
                .get_token_ids()
                .iter()
                .map(|x| *x as u32)
                .collect::<Vec<_>>();
            let tokens_generated = sq.get_len() - sq.get_prompt_len();

            if tokens_generated > sampling_params.max_tokens {
                result.push(Right("length".to_string()));
                break;
            }

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

            let next_token = try_api!(self.logits_processor.sample(&logits));
            //the first token (after prompt may be eos token)
            if Some(next_token) == eos_token_id && tokens_generated > 1 {
                result.push(Right("stop".to_string()));
                break;
            }
            let text = self.tokenizer.next_token(next_token).unwrap();
            let text = match text {
                Some(t) => t,
                _ => "".to_string(),
            };

            let logprob = Logprobs {
                token: next_token as usize,
                logprob: 0.0,
                top_logprobs: Vec::<TopLogprob>::new(),
                bytes: text,
            };
            result.push(Left(logprob));
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn tokenizer(&self) -> &TokenOutputStream {
        &self.tokenizer
    }

    fn get_conversation(&mut self, with_history: bool) -> &mut dyn Conversation {
        if !with_history {
            self.conversation.clear_message();
        }
        &mut self.conversation
    }

    fn get_model_config(&self) -> Config {
        match &self.model {
            LLMModel::LLAMA(llama) => llama.get_config().clone(),
            LLMModel::Phi3(phi) => phi.get_config().clone(),
            LLMModel::Qwen2(qwen2) => qwen2.get_config().clone(),
        }
    }

    fn get_dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

unsafe impl Send for DefaultPipeline {}
unsafe impl Sync for DefaultPipeline {}
