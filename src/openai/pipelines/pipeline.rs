use super::{get_token, TokenOrFinishReason};
use crate::openai::logits_processor::{LogitsProcessor, Sampling};
use crate::openai::models::TokenID;
use crate::openai::requests::StopTokens;
use crate::openai::sampling_params::{Logprobs, TopLogprob};
use crate::openai::{BosEosToken, TokenizerConfig};
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
            deepseek::{DeepSeek, DeepSeekConfig},
            gemma::{Gemma, GemmaConfig},
            llama::{Llama, LlamaConfig},
            mistral::{Mistral, MistralConfig},
            phi2::{Phi2, Phi2Config},
            phi3::{Phi, PhiConfig},
            quantized_llama::GGUFLLaMa,
            quantized_phi3::GGUFPhi3,
            quantized_qwen2::GGUFQWen2,
            qwen2::{Qwen2, QwenConfig},
            stable_lm::{StableLM, StableLMConfig},
            yi::{Yi, YiConfig},
            Config,
        },
        responses::APIError,
        PipelineConfig,
    },
    paged_attention::input_metadata::InputMetadata,
    try_api, SpecificConfig,
};
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use either::Either;
use either::Either::{Left, Right};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rayon::prelude::*;
use std::collections::VecDeque;
pub use std::rc::Rc;
use std::{path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;
const EOS_TOKEN: &str = "</s>";
const SAMPLING_SEED: u64 = 299792458;
const MIN_GEN_TOKENS: usize = 128;
const MAX_GEN_TOKENS: usize = 16 * 1024;
enum LLMModel {
    Llama(Llama),
    Phi2(Phi2),
    Phi3(Phi),
    Qwen2(Qwen2),
    Gemma(Gemma),
    Mistral(Mistral),
    Yi(Yi),
    StableLM(StableLM),
    DeepSeek(DeepSeek),
    LlamaGGUF(GGUFLLaMa),
    Phi3GGUF(GGUFPhi3),
    QWen2GGUF(GGUFQWen2),
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
    stop_token_ids: Vec<u32>,
    rank: usize,
}

pub struct DefaultLoader {
    config: SpecificConfig,
    name: String,
}

#[derive(Debug, Clone)]
pub struct DefaultModelPaths {
    pub tokenizer_filename: PathBuf,
    pub tokenizer_config_filename: PathBuf,
    pub config_filename: PathBuf,
    pub filenames: Vec<PathBuf>,
}

impl DefaultModelPaths {
    fn get_config_filename(&self) -> PathBuf {
        self.config_filename.clone()
    }
    fn get_tokenizer_filename(&self) -> PathBuf {
        self.tokenizer_filename.clone()
    }
    fn get_tokenizer_config_filename(&self) -> PathBuf {
        self.tokenizer_config_filename.clone()
    }
    fn get_weight_filenames(&self) -> Vec<PathBuf> {
        self.filenames.clone()
    }
}

impl DefaultLoader {
    pub fn new(config: SpecificConfig, name: String) -> Self {
        Self { config, name }
    }
}

impl DefaultLoader {
    pub fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<DefaultModelPaths, APIError> {
        let api = try_api!(ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(get_token(hf_token, hf_token_path)?))
            .build());
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = try_api!(api.get("tokenizer.json"));

        let config_filename = try_api!(api.get("config.json"));

        let tokenizer_config_filename = match api.get("tokenizer_config.json") {
            Ok(f) => f,
            _ => "".into(),
        };

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

        Ok(DefaultModelPaths {
            tokenizer_filename,
            tokenizer_config_filename,
            config_filename,
            filenames,
        })
    }

    pub async fn load_model(
        &self,
        paths: DefaultModelPaths,
        dtype: DType,
        quant: &Option<String>,
        device_ids: Vec<usize>,
    ) -> Result<(Vec<Box<DefaultPipeline>>, PipelineConfig), APIError> {
        let specific_args = self.config.clone();

        let (models, devices, config, sep_style) = if quant.is_some()
            && matches!(quant.as_ref().unwrap().as_str(), "ggml" | "gguf")
        {
            let device = crate::new_device(device_ids[0]).unwrap();
            let path = paths.get_weight_filenames()[0].clone();
            println!(
                "Loading quantized {} model from file {}",
                self.name,
                path.display()
            );
            let mut file = try_api!(std::fs::File::open(&path));
            let content =
                try_api!(gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path)));
            let s_cfg = specific_args.clone();
            let (model, config, sep_style) = match self.name.as_str() {
                "llama" => {
                    let model = try_api!(GGUFLLaMa::from_gguf(
                        content, &mut file, &device, dtype, s_cfg
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::LlamaGGUF(model), cfg, SeparatorStyle::Llama)
                }
                "llama3" => {
                    let model = try_api!(GGUFLLaMa::from_gguf(
                        content, &mut file, &device, dtype, s_cfg
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::LlamaGGUF(model), cfg, SeparatorStyle::Llama3)
                }
                "phi3" => {
                    let model = try_api!(GGUFPhi3::from_gguf(
                        content, &mut file, &device, dtype, s_cfg
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::Phi3GGUF(model), cfg, SeparatorStyle::Phi)
                }
                "qwen2" => {
                    let model = try_api!(GGUFQWen2::from_gguf(
                        content, &mut file, &device, dtype, s_cfg
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::QWen2GGUF(model), cfg, SeparatorStyle::Qwen2)
                }
                _ => panic!("Model not supported!"),
            };
            (vec![model], vec![device], config.to_owned(), sep_style)
        } else {
            let config = match self.name.as_str() {
                "llama" | "llama3" => {
                    let config: LlamaConfig = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    config.into_config(false, dtype, &specific_args)
                }
                "phi2" => {
                    let config: Phi2Config = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    //Phi2 use F32 type for kvcache
                    config.into_config(false, DType::F32, &specific_args)
                }
                "phi3" => {
                    let config: PhiConfig = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    config.into_config(false, dtype, &specific_args)
                }
                "qwen2" => {
                    let config: QwenConfig = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    config.into_config(false, dtype, &specific_args)
                }
                "gemma" => {
                    let config: GemmaConfig = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    config.into_config(false, dtype, &specific_args)
                }
                "mistral" => {
                    let config: MistralConfig = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    config.into_config(false, dtype, &specific_args)
                }
                "yi" => {
                    let config: YiConfig = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    config.into_config(false, dtype, &specific_args)
                }
                "stablelm" => {
                    let config: StableLMConfig = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    config.into_config(false, dtype, &specific_args)
                }
                "deepseek" => {
                    let config: DeepSeekConfig = try_api!(serde_json::from_slice(&try_api!(
                        std::fs::read(paths.get_config_filename())
                    ),));
                    config.into_config(false, dtype, &specific_args)
                }
                _ => panic!("Model not supported!"),
            };

            println!("Model {:?}", config);

            println!("Loading {} model.", self.name);
            use crate::openai::distributed::Comm;
            #[cfg(feature = "nccl")]
            let id = cudarc::nccl::safe::Id::new().unwrap();
            let results: Vec<_> = device_ids
                .par_iter()
                .enumerate()
                .map(|(rank, dev_id)| {
                    let num_devices = device_ids.len();
                    if num_devices > 1 {
                        println!(
                            "Loading partial model on device rank {} (ordinal {})",
                            rank, *dev_id
                        );
                    }

                    let paths: Vec<PathBuf> = paths.get_weight_filenames();
                    let device = crate::new_device(*dev_id).unwrap();
                    #[cfg(feature = "nccl")]
                    let _ = device.as_cuda_device().unwrap().bind_to_thread();

                    #[cfg(feature = "nccl")]
                    let comm = Rc::new(
                        Comm::from_rank(
                            device.as_cuda_device().unwrap().cuda_device(),
                            rank,
                            num_devices,
                            id,
                        )
                        .unwrap(),
                    );

                    #[cfg(not(feature = "nccl"))]
                    let comm = Rc::new(Comm::default());

                    let vb = unsafe {
                        candle_nn::var_builder::ShardedSafeTensors::var_builder(
                            &paths, dtype, &device,
                        )
                        .unwrap()
                    };

                    let (model, sep) = match self.name.as_str() {
                        "llama" => (
                            LLMModel::Llama(try_api!(Llama::load(
                                vb, &config, dtype, &device, comm
                            ))),
                            SeparatorStyle::Llama,
                        ),
                        "llama3" => (
                            LLMModel::Llama(try_api!(Llama::load(
                                vb, &config, dtype, &device, comm
                            ))),
                            SeparatorStyle::Llama3,
                        ),
                        "phi2" => (
                            LLMModel::Phi2(try_api!(Phi2::new(vb, &config, dtype, &device, comm))),
                            SeparatorStyle::Phi,
                        ),
                        "phi3" => (
                            LLMModel::Phi3(try_api!(Phi::new(vb, &config, dtype, &device, comm))),
                            SeparatorStyle::Phi,
                        ),
                        "qwen2" => (
                            LLMModel::Qwen2(try_api!(Qwen2::new(
                                vb, &config, dtype, &device, comm
                            ))),
                            SeparatorStyle::Qwen2,
                        ),
                        "gemma" => (
                            LLMModel::Gemma(try_api!(Gemma::new(
                                vb, &config, dtype, &device, comm
                            ))),
                            SeparatorStyle::Gemma,
                        ),
                        "mistral" => (
                            LLMModel::Mistral(try_api!(Mistral::new(
                                vb, &config, dtype, &device, comm
                            ))),
                            SeparatorStyle::Mistral,
                        ),
                        "yi" => (
                            LLMModel::Yi(try_api!(Yi::new(vb, &config, dtype, &device, comm))),
                            SeparatorStyle::Yi,
                        ),
                        "stablelm" => (
                            LLMModel::StableLM(try_api!(StableLM::new(
                                vb, &config, dtype, &device, comm
                            ))),
                            SeparatorStyle::StableLM,
                        ),
                        "deepseek" => (
                            LLMModel::DeepSeek(try_api!(DeepSeek::load(
                                vb, &config, dtype, &device, comm
                            ))),
                            SeparatorStyle::Llama3,
                        ),
                        _ => panic!("Model not supported!"),
                    };
                    Ok((model, device, sep))
                })
                .collect();

            // Separate devices and models from the results
            let mut devices = Vec::new();
            let mut models = Vec::new();
            let mut sep_style = Vec::new();

            for result in results {
                match result {
                    Ok((model, device, sep)) => {
                        devices.push(device);
                        models.push(model);
                        sep_style.push(sep)
                    }
                    Err(e) => {
                        return Err(e.into());
                    }
                }
            }

            (models, devices, config, sep_style[0].clone())
        };

        println!("Done loading.");

        //max and min number of tokens generated per request
        let default_max_tokens = specific_args
            .max_gen_tokens
            .unwrap_or(config.max_seq_len / 2)
            .clamp(MIN_GEN_TOKENS, MAX_GEN_TOKENS);

        let pipeline_config = PipelineConfig {
            max_model_len: config.max_seq_len,
            default_max_tokens,
            penalty: specific_args.penalty.unwrap_or(1.),
            repeat_last_n: specific_args.repeat_last_n.unwrap_or(64),
            temperature: specific_args.temperature.unwrap_or(0.7),
        };

        let tokenizer_cfg_file = paths.get_tokenizer_config_filename();
        let (chat_template, bos_token, eos_token): (
            Option<String>,
            Option<String>,
            Option<String>,
        ) = if tokenizer_cfg_file.display().to_string() != "" && Path::exists(&tokenizer_cfg_file) {
            let cfg_tokenizer: TokenizerConfig = try_api!(serde_json::from_slice(try_api!(
                &std::fs::read(tokenizer_cfg_file)
            )));
            let bos = match cfg_tokenizer.bos_token {
                BosEosToken(Either::Left(Some(id))) => Some(id),
                BosEosToken(Either::Right(Some(content))) => content.content.clone(),
                _ => None,
            };
            let eos = match cfg_tokenizer.eos_token {
                BosEosToken(Either::Left(Some(id))) => Some(id),
                BosEosToken(Either::Right(Some(content))) => content.content.clone(),
                _ => None,
            };
            (cfg_tokenizer.chat_template, bos, eos)
        } else {
            (None, None, None)
        };
        if chat_template.is_some() {
            println!("Chat Template {} \n", chat_template.as_ref().unwrap());
        } else {
            println!("Warning: Missing tokenizer_config.json \n Warning: Chat Template not found, use built-in template which may not correct!");
        }
        println!("{:?}", pipeline_config);
        println!("{:?}", specific_args);

        let pipelines = models
            .into_iter()
            .enumerate()
            .map(|(rank, model)| {
                let logits_processor = {
                    let temperature = f64::from(pipeline_config.temperature);
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
                let tokenizer_ = Tokenizer::from_file(paths.get_tokenizer_filename())
                    .map_err(|x| APIError::new(x.to_string()))
                    .unwrap();
                let tokenizer =
                    candle_examples::token_output_stream::TokenOutputStream::new(tokenizer_);

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
                //custom stop tokens
                if let Some(custom_stop) = &config.custom_stop_tokens {
                    for stop in custom_stop {
                        if let Some(token) = tokenizer.get_token(stop) {
                            stop_token_ids.push(token)
                        };
                    }
                }

                if stop_token_ids.is_empty() {
                    //if no eos_token defined in the config, use default
                    if let Some(token) = tokenizer.get_token("<|endoftext|>") {
                        stop_token_ids.push(token);
                    }
                    if let Some(token) = tokenizer.get_token("<|end|>") {
                        stop_token_ids.push(token);
                    } else if stop_token_ids.is_empty() {
                        let token = tokenizer.tokenizer().token_to_id(EOS_TOKEN).unwrap_or(0);
                        stop_token_ids.push(token);
                    }
                }
                Box::new(DefaultPipeline {
                    model,
                    args: specific_args.clone(),
                    tokenizer,
                    logits_processor,
                    conversation: DefaultConversation::new(
                        self.name.to_string(),
                        chat_template.clone(),
                        Vec::default(),
                        sep_style.clone(),
                        bos_token.clone(),
                        eos_token.clone(),
                        ("user".to_string(), "assistant".to_string()),
                        DefaultConversationSeparators {
                            sep: " ".to_string(),
                            sep2: Some(" </s></s>".to_string()),
                        },
                    ),
                    name: self.name.clone(),
                    dtype,
                    device: devices[rank].clone(),
                    stop_token_ids,
                    rank,
                })
            })
            .collect();

        Ok((pipelines, pipeline_config))
    }
}

impl DefaultPipeline {
    pub fn forward(
        &self,
        input_tokens: Tensor,
        input_positions: &[Vec<usize>],
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor, APIError> {
        let input_tokens = if input_tokens.shape().dims().len() < 2 {
            input_tokens
                .reshape((1, input_tokens.shape().dims()[0]))
                .unwrap()
        } else {
            input_tokens
        };

        match &self.model {
            LLMModel::Llama(llama) => llama
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::Phi2(phi) => phi
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::Phi3(phi) => phi
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::Qwen2(qwen2) => qwen2
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::Gemma(gemma) => gemma
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::Mistral(mistral) => mistral
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::Yi(yi) => yi
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::StableLM(stablelm) => stablelm
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::DeepSeek(deepseek) => deepseek
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::Phi3GGUF(phi3) => phi3
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::LlamaGGUF(llama) => llama
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
            LLMModel::QWen2GGUF(qwen2) => qwen2
                .forward(&input_tokens, input_positions, kv_cache, &input_metadata)
                .map_err(APIError::from),
        }
    }

    pub fn sample(
        &mut self,
        logits: &Tensor,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Result<Vec<TokenOrFinishReason>, APIError> {
        let (tokens_generated, custom_stop_tokens, panalties, reference_tokens): (
            Vec<i32>,
            Vec<Vec<String>>,
            Vec<f32>,
            Vec<Vec<u32>>,
        ) = groups
            .into_par_iter()
            .map(|group| {
                let sampling_params = &group.sampling_params;
                let seq = group.get_seqs().values().next().unwrap();
                let sq = seq.deref_mut();
                let tokens = sq
                    .get_token_ids()
                    .iter()
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>();
                let generated = sq.get_len() - sq.get_prompt_len();

                let custom_stop_token = match &sampling_params.stop {
                    Some(StopTokens::Multi(v)) => v.to_vec(),
                    Some(StopTokens::Single(v)) => {
                        vec![v.clone()]
                    }
                    _ => vec![],
                };

                let ref_tokens = if sampling_params.repetition_penalty != 1.
                    && self.args.repeat_last_n.unwrap_or(64) < generated
                {
                    let start_at = tokens
                        .len()
                        .saturating_sub(self.args.repeat_last_n.unwrap_or(64));
                    tokens[start_at..].to_vec()
                } else {
                    vec![]
                };
                (
                    if generated > sampling_params.max_tokens {
                        -1i32
                    } else {
                        generated as i32
                    },
                    custom_stop_token,
                    sampling_params.repetition_penalty,
                    ref_tokens,
                )
            })
            .collect::<Vec<(i32, Vec<String>, f32, Vec<u32>)>>()
            .into_iter()
            .fold(
                (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                |mut acc, (gen, stop, penalty, ref_t)| {
                    acc.0.push(gen);
                    acc.1.push(stop);
                    acc.2.push(penalty);
                    acc.3.push(ref_t);
                    acc
                },
            );

        let logits = if panalties.iter().any(|&v| v != 1.0 && v != 0.) {
            self.logits_processor
                .apply_batch_repeat_penalty(&logits, panalties, reference_tokens)
                .unwrap()
        } else {
            logits.to_owned()
        };

        let next_tokens = self.logits_processor.sample(&logits).unwrap();
        let result: Vec<TokenOrFinishReason> = next_tokens
            .into_par_iter()
            .enumerate()
            .map(|(i, next_token)| {
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
                let custom_stop_token_match = if custom_stop_tokens[i].len() > 0
                    && custom_stop_tokens[i].contains(&text.trim().to_string())
                {
                    true
                } else {
                    false
                };

                if tokens_generated[i] < 0 {
                    Right("length".to_string())
                } else if tokens_generated[i] > 0
                    && (custom_stop_token_match || self.stop_token_ids.contains(&next_token))
                {
                    Right("stop".to_string())
                } else {
                    Left(Logprobs {
                        token: next_token as usize,
                        logprob: 0.0,
                        top_logprobs: Vec::<TopLogprob>::new(),
                        bytes: text,
                    })
                }
            })
            .collect();
        Ok(result)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tokenizer(&self) -> &TokenOutputStream {
        &self.tokenizer
    }

    pub fn get_conversation(&mut self, with_history: bool) -> &mut dyn Conversation {
        if !with_history {
            self.conversation.clear_message();
        }
        &mut self.conversation
    }

    pub fn get_past_conversation(&self) -> &dyn Conversation {
        &self.conversation
    }

    pub fn get_model_config(&self) -> Config {
        match &self.model {
            LLMModel::Llama(llama) => llama.get_config().clone(),
            LLMModel::Phi2(phi) => phi.get_config().clone(),
            LLMModel::Phi3(phi) => phi.get_config().clone(),
            LLMModel::Qwen2(qwen2) => qwen2.get_config().clone(),
            LLMModel::Gemma(gemma) => gemma.get_config().clone(),
            LLMModel::Mistral(mistral) => mistral.get_config().clone(),
            LLMModel::Yi(yi) => yi.get_config().clone(),
            LLMModel::StableLM(stablelm) => stablelm.get_config().clone(),
            LLMModel::DeepSeek(deepseek) => deepseek.get_config().clone(),
            LLMModel::Phi3GGUF(phi3) => phi3.get_config().clone(),
            LLMModel::LlamaGGUF(llama) => llama.get_config().clone(),
            LLMModel::QWen2GGUF(qwen2) => qwen2.get_config().clone(),
        }
    }

    pub fn get_dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn reset_decoder(&mut self) -> Option<String> {
        let ret = self.tokenizer.decode_rest().unwrap_or(None);
        self.tokenizer.clear();
        ret
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

unsafe impl Send for DefaultPipeline {}
unsafe impl Sync for DefaultPipeline {}
