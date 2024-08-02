use super::{get_token, ModelLoader, ModelPaths, ModulePipeline, TokenOrFinishReason};
use crate::openai::logits_processor::{LogitsProcessor, Sampling};
use crate::openai::models::TokenID;
use crate::openai::sampling_params::{Logprobs, TopLogprob};
use crate::scheduler::sequence::SequenceGroup;
use crate::{
    openai::{
        conversation::{
            default_conversation::{
                DefaultConversation, DefaultConversationSeparators, SeparatorStyle,
            },
            Conversation,
        },
        models::{
            gemma::{Gemma, GemmaConfig},
            llama::{Llama, LlamaConfig},
            mistral::{Mistral, MistralConfig},
            phi2::{Phi2, Phi2Config},
            phi3::{Phi, PhiConfig},
            qwen2::{Qwen2, QwenConfig},
            stable_lm::{StableLM, StableLMConfig},
            yi::{Yi, YiConfig},
            Config,
        },
        responses::APIError,
        PipelineConfig,
    },
    paged_attention::input_metadata::InputMetadata,
    try_api,
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use either::Either;
use either::Either::{Left, Right};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::{path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;
const EOS_TOKEN: &str = "</s>";
const SAMPLING_SEED: u64 = 299792458;
const MIN_GEN_TOKENS: usize = 128;
const MAX_GEN_TOKENS: usize = 4096;

#[derive(Debug, Clone)]
pub struct SpecificConfig {
    repeat_last_n: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    penalty: Option<f32>,
    max_gen_tokens: Option<usize>,
}

impl SpecificConfig {
    pub fn new(
        repeat_last_n: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        penalty: Option<f32>,
        max_gen_tokens: Option<usize>,
    ) -> Self {
        Self {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
        }
    }
}

enum LLMModel {
    LLAMA(Llama),
    Phi2(Phi2),
    Phi3(Phi),
    Qwen2(Qwen2),
    Gemma(Gemma),
    Mistral(Mistral),
    Yi(Yi),
    StableLM(StableLM),
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
    config: Config,
    stop_token_ids: Vec<u32>,
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

impl ModelLoader for DefaultLoader {
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
    ) -> Result<(Box<dyn ModulePipeline>, PipelineConfig), APIError> {
        let specific_args = self.config.clone();

        let config = match self.name.as_str() {
            "llama" | "llama3" => {
                let config: LlamaConfig = try_api!(serde_json::from_slice(&try_api!(
                    std::fs::read(paths.get_config_filename())
                ),));
                config.into_config(false, dtype)
            }
            "phi2" => {
                let config: Phi2Config = try_api!(serde_json::from_slice(&try_api!(
                    std::fs::read(paths.get_config_filename())
                ),));
                //Phi2 use F32 type for kvcache
                config.into_config(false, DType::F32)
            }
            "phi3" => {
                let config: PhiConfig = try_api!(serde_json::from_slice(&try_api!(std::fs::read(
                    paths.get_config_filename()
                )),));
                config.into_config(false, dtype)
            }
            "qwen2" => {
                let config: QwenConfig = try_api!(serde_json::from_slice(&try_api!(
                    std::fs::read(paths.get_config_filename())
                ),));
                config.into_config(false, dtype)
            }
            "gemma" => {
                let config: GemmaConfig = try_api!(serde_json::from_slice(&try_api!(
                    std::fs::read(paths.get_config_filename())
                ),));
                config.into_config(false, dtype)
            }
            "mistral" => {
                let config: MistralConfig = try_api!(serde_json::from_slice(&try_api!(
                    std::fs::read(paths.get_config_filename())
                ),));
                config.into_config(false, dtype)
            }
            "yi" => {
                let config: YiConfig = try_api!(serde_json::from_slice(&try_api!(std::fs::read(
                    paths.get_config_filename()
                )),));
                config.into_config(false, dtype)
            }
            "stablelm" => {
                let config: StableLMConfig = try_api!(serde_json::from_slice(&try_api!(
                    std::fs::read(paths.get_config_filename())
                ),));
                config.into_config(false, dtype)
            }
            _ => panic!("Model not supported!"),
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
            "llama3" => (
                LLMModel::LLAMA(try_api!(Llama::load(vb, &config, dtype, &device))),
                SeparatorStyle::Llama3,
            ),
            "phi2" => (
                LLMModel::Phi2(try_api!(Phi2::new(vb, &config, dtype, &device))),
                SeparatorStyle::Phi,
            ),
            "phi3" => (
                LLMModel::Phi3(try_api!(Phi::new(vb, &config, dtype, &device))),
                SeparatorStyle::Phi,
            ),
            "qwen2" => (
                LLMModel::Qwen2(try_api!(Qwen2::new(vb, &config, dtype, &device))),
                SeparatorStyle::Qwen2,
            ),
            "gemma" => (
                LLMModel::Gemma(try_api!(Gemma::new(vb, &config, dtype, &device))),
                SeparatorStyle::Gemma,
            ),
            "mistral" => (
                LLMModel::Mistral(try_api!(Mistral::new(vb, &config, dtype, &device))),
                SeparatorStyle::Mistral,
            ),
            "yi" => (
                LLMModel::Yi(try_api!(Yi::new(vb, &config, dtype, &device))),
                SeparatorStyle::Yi,
            ),
            "stablelm" => (
                LLMModel::StableLM(try_api!(StableLM::new(vb, &config, dtype, &device))),
                SeparatorStyle::StableLM,
            ),
            _ => panic!("Model not supported!"),
        };

        let tokenizer_ = Tokenizer::from_file(paths.get_tokenizer_filename())
            .map_err(|x| APIError::new(x.to_string()))?;

        let tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer_);

        println!("Done loading.");

        //max and min number of tokens generated per request
        let mut default_max_tokens = specific_args
            .max_gen_tokens
            .unwrap_or(config.max_seq_len / 5);
        if default_max_tokens < MIN_GEN_TOKENS {
            default_max_tokens = MIN_GEN_TOKENS;
        } else if default_max_tokens > MAX_GEN_TOKENS {
            default_max_tokens = MAX_GEN_TOKENS;
        }

        let pipeline_config = PipelineConfig {
            max_model_len: config.max_seq_len,
            default_max_tokens,
            penalty: specific_args.penalty.unwrap_or(1.),
            repeat_last_n: specific_args.repeat_last_n.unwrap_or(64),
            temperature: specific_args.temperature.unwrap_or(0.7),
        };

        println!("{:?}", pipeline_config);

        let mut stop_token_ids = Vec::<u32>::new();

        match &config.eos_token_id {
            //eos_token defined in the config
            TokenID(Either::Left(eos_token)) => {
                if let Some(tk) = eos_token {
                    stop_token_ids.push(*tk);
                }
            }
            TokenID(Either::Right(eos_token_list)) => {
                if let Some(tks) = eos_token_list {
                    stop_token_ids.extend(tks)
                }
            }
        }

        if stop_token_ids.len() == 0 {
            //if no eos_token defined in the config, use default
            let eos_token = match tokenizer.get_token("<|endoftext|>") {
                Some(token) => token,
                _ => tokenizer.tokenizer().token_to_id(EOS_TOKEN).unwrap_or(0),
            };
            stop_token_ids.push(eos_token);
        }

        //custom stop tokens
        if let Some(custom_stop) = &config.custom_stop_tokens {
            for stop in custom_stop {
                match tokenizer.get_token(&stop) {
                    Some(token) => stop_token_ids.push(token),
                    None => {}
                };
            }
        }

        println!("{:?}", specific_args);

        let logits_processor = {
            let temperature = pipeline_config.temperature as f64;
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (specific_args.top_k, specific_args.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(SAMPLING_SEED, sampling)
        };

        Ok((
            Box::new(DefaultPipeline {
                model,
                args: specific_args,
                tokenizer,
                logits_processor: logits_processor,
                conversation: DefaultConversation::new(
                    self.name.to_string(),
                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n [/INST]".to_string(),
                    Vec::default(),
                    0,
                    sep_style,
                    "".to_string(),
                    stop_token_ids.clone(),
                    ("user".to_string(), "assistant".to_string()),
                    DefaultConversationSeparators {
                        sep: " ".to_string(),
                        sep2: Some(" </s></s>".to_string()),
                    },
                ),
                name: self.name.clone(),
                dtype,
                device: device.clone(),
                config: config.clone(),
                stop_token_ids,
            }),
            pipeline_config,
        ))
    }
}

impl ModulePipeline for DefaultPipeline {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: &Vec<Vec<usize>>,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        mut input_metadata: InputMetadata,
    ) -> Result<Tensor, APIError> {
        let input_tokens = if input_tokens.shape().dims().len() < 2 {
            input_tokens
                .reshape((1, input_tokens.shape().dims()[0]))
                .unwrap()
        } else {
            input_tokens
        };

        let ret = match &mut self.model {
            LLMModel::LLAMA(llama) => llama
                .forward(
                    &input_tokens,
                    &input_positions,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::Phi2(phi) => phi
                .forward(
                    &input_tokens,
                    &input_positions,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::Phi3(phi) => phi
                .forward(
                    &input_tokens,
                    input_positions,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::Qwen2(qwen2) => qwen2
                .forward(
                    &input_tokens,
                    &input_positions,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::Gemma(gemma) => gemma
                .forward(
                    &input_tokens,
                    &input_positions,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::Mistral(mistral) => mistral
                .forward(
                    &input_tokens,
                    &input_positions,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::Yi(yi) => yi
                .forward(
                    &input_tokens,
                    &input_positions,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
            LLMModel::StableLM(stablelm) => stablelm
                .forward(
                    &input_tokens,
                    input_positions,
                    kv_cache,
                    &mut input_metadata,
                )
                .map_err(APIError::from),
        };

        return ret;
    }

    fn sample(
        &mut self,
        logits: Tensor,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Result<Vec<TokenOrFinishReason>, APIError> {
        use std::collections::HashMap;
        use std::sync::Mutex;
        let shared_result = Arc::new(Mutex::new(HashMap::<usize, TokenOrFinishReason>::new()));
        let shared_group_idx = Arc::new(Mutex::new(0));
        groups.par_iter().for_each(|group| {
            let mut group_idx = 0;
            {
                let mut groupidx = shared_group_idx.lock().unwrap();
                group_idx = *groupidx;
                *groupidx += 1;
            }

            let sampling_params = &group.sampling_params;
            for (_, seq) in group.get_seqs() {
                let logits = logits.i((group_idx, ..)).unwrap().contiguous();
                let logits = logits.unwrap().squeeze(0).unwrap();
                let sq = seq.deref_mut();
                let tokens = sq
                    .get_token_ids()
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>();
                let tokens_generated = sq.get_len() - sq.get_prompt_len();

                if tokens_generated > sampling_params.max_tokens {
                    let mut result = shared_result.lock().unwrap();
                    result.insert(group_idx, Right("length".to_string()));
                    break;
                }

                let logits = if sampling_params.repetition_penalty == 1.
                    || self.args.repeat_last_n.unwrap_or(64) >= tokens_generated
                {
                    logits
                } else {
                    let start_at = tokens
                        .len()
                        .saturating_sub(self.args.repeat_last_n.unwrap_or(64));
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        sampling_params.repetition_penalty,
                        &tokens[start_at..],
                    )
                    .unwrap_or(logits)
                };

                let next_token = self.logits_processor.sample(&logits).unwrap();
                let mut text = self
                    .tokenizer
                    .tokenizer()
                    .decode(&[next_token], false)
                    .unwrap_or(" ".to_string());
                let origin_text = self
                    .tokenizer
                    .tokenizer()
                    .id_to_token(next_token)
                    .unwrap_or("".to_string());
                //properly handle space token
                if origin_text.contains("▁") && origin_text.replace("▁", "") == text {
                    text = origin_text.replace("▁", " ");
                }
                if self.stop_token_ids.contains(&next_token) && tokens_generated > 1 {
                    let mut result = shared_result.lock().unwrap();
                    result.insert(group_idx, Right("stop".to_string()));
                    break;
                }
                {
                    let logprob = Logprobs {
                        token: next_token as usize,
                        logprob: 0.0,
                        top_logprobs: Vec::<TopLogprob>::new(),
                        bytes: text,
                    };
                    let mut result = shared_result.lock().unwrap();
                    result.insert(group_idx, Left(logprob));
                }
            }
        });

        let final_result = Arc::try_unwrap(shared_result)
            .expect("Arc should have only one reference left")
            .into_inner()
            .expect("Mutex should not be poisoned");

        let mut sorted_vec: Vec<_> = final_result.into_iter().collect();
        sorted_vec.sort_by_key(|&(key, _)| key);

        let result: Vec<TokenOrFinishReason> =
            sorted_vec.into_iter().map(|(_, value)| value).collect();

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
            LLMModel::Phi2(phi) => phi.get_config().clone(),
            LLMModel::Phi3(phi) => phi.get_config().clone(),
            LLMModel::Qwen2(qwen2) => qwen2.get_config().clone(),
            LLMModel::Gemma(gemma) => gemma.get_config().clone(),
            LLMModel::Mistral(mistral) => mistral.get_config().clone(),
            LLMModel::Yi(yi) => yi.get_config().clone(),
            LLMModel::StableLM(stablelm) => stablelm.get_config().clone(),
        }
    }

    fn get_dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn reset_decoder(&mut self) -> Option<String> {
        let ret = self.tokenizer.decode_rest().unwrap_or(None);
        self.tokenizer.clear();
        ret
    }
}

unsafe impl Send for DefaultPipeline {}
unsafe impl Sync for DefaultPipeline {}
