use super::{get_token, TokenOrFinishReason};
use crate::backend::progress::{progress_worker, ProgressReporter};
use crate::openai::logits_processor::LogitsProcessor;
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
            gemma3::{Gemma3, Gemma3Config},
            glm4::{GLMConfig, GLM4},
            llama::{Llama, LlamaConfig},
            mistral::{Mistral, MistralConfig},
            phi2::{Phi2, Phi2Config},
            phi3::{Phi, PhiConfig},
            quantized_glm4::GGUFGLM4,
            quantized_llama::GGUFLLaMa,
            quantized_phi3::GGUFPhi3,
            quantized_qwen::GGUFQWen,
            qwen::{Qwen, QwenConfig},
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
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use either::Either;
use either::Either::{Left, Right};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::Path;
pub use std::rc::Rc;
use std::sync::RwLock;
use std::{path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;
use tracing::{info, warn};
const EOS_TOKEN: &str = "</s>";
const SAMPLING_SEED: u64 = 299792458;
const MIN_GEN_TOKENS: usize = 128;
const MAX_GEN_TOKENS: usize = 16 * 1024;
enum LLMModel {
    Llama(Llama),
    Phi2(Phi2),
    Phi3(Phi),
    Qwen(Qwen),
    Gemma(Gemma),
    Gemma3(Gemma3),
    Mistral(Mistral),
    Yi(Yi),
    StableLM(StableLM),
    GLM4(GLM4),
    DeepSeek(DeepSeek),
    LlamaGGUF(GGUFLLaMa),
    Phi3GGUF(GGUFPhi3),
    QWenGGUF(GGUFQWen),
    GLM4GGUF(GGUFGLM4),
}
/// top-p, multinomial, and argmax sampling are implemented. Beam search is not implemented.
pub struct DefaultPipeline {
    model: LLMModel,
    args: SpecificConfig,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    conversation: DefaultConversation,
    name: String,
    dtype: DType,
    device: Device,
    stop_token_ids: Vec<u32>,
    rank: usize,
    pub stream_decoders: RwLock<super::StreamDecoderMap>,
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
        weight_file: Option<String>,
        quant: Option<String>,
        revision: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<DefaultModelPaths, APIError> {
        if quant.is_some() && quant.as_ref().unwrap() == "gguf" && weight_file.is_some() {
            info!(
                "Downloading GGUF file {} from repo {}",
                weight_file.as_ref().unwrap(),
                model_id,
            );
            return self.download_gguf_model(model_id, None, weight_file.clone().unwrap());
        };
        let api = try_api!(ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(get_token(hf_token, hf_token_path)?))
            .build());
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            revision.clone(),
        ));

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

    pub fn download_gguf_model(
        &self,
        model_id: String,
        revision: Option<String>,
        filename: String,
    ) -> Result<DefaultModelPaths, APIError> {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let revision = revision.unwrap_or("main".to_string());
        let mut filenames = vec![];
        let filename = try_api!(api
            .repo(hf_hub::Repo::with_revision(
                model_id,
                hf_hub::RepoType::Model,
                revision.to_string(),
            ))
            .get(filename.as_str()));
        filenames.push(filename);

        Ok(DefaultModelPaths {
            tokenizer_filename: "".into(),
            tokenizer_config_filename: "".into(),
            config_filename: "".into(),
            filenames,
        })
    }

    //support loading in both multithreaded and multiprocess mode
    pub async fn load_model(
        &self,
        paths: DefaultModelPaths,
        dtype: DType,
        quant: &Option<String>,
        device_ids: Vec<usize>, //pass only 1 device_id in multiprocess mode, otherwise, multiple device_ids in multithread mode
        #[cfg(feature = "nccl")] comm_id: Option<crate::openai::distributed::Id>, //must pass comm id in multiprocess mode
        local_rank: Option<usize>, //must pass current rank in multiprocess mode
        local_world_size: Option<usize>, //must pass the number of local devices used in multiprocess mode
        #[cfg(feature = "nccl")] global_rank: Option<usize>, //must pass current global rank in multi-node mode
        #[cfg(feature = "nccl")] global_world_size: Option<usize>, //must pass total number of devices used in multi-node mode
    ) -> Result<(Vec<Box<DefaultPipeline>>, PipelineConfig), APIError> {
        let specific_args = self.config.clone();
        let reporter = Arc::new(RwLock::new(ProgressReporter::new(local_rank.unwrap_or(0))));
        let num_subprogress = local_world_size.map_or(0, |n| n - 1);

        let (models, devices, config, sep_style) = if quant.is_some()
            && matches!(quant.as_ref().unwrap().as_str(), "ggml" | "gguf")
        {
            let device = crate::new_device(device_ids[0]).unwrap();
            let path = paths.get_weight_filenames()[0].clone();
            info!(
                "Loading quantized {} model from file {}",
                self.name,
                path.display()
            );
            let s_cfg = specific_args.clone();
            let nlayers = {
                let mut file = try_api!(std::fs::File::open(path.clone()));
                let content = try_api!(
                    gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path.clone()))
                );
                let nlayers = match self.name.as_str() {
                    "llama" | "llama3" => GGUFLLaMa::get_num_of_layers(content),
                    "phi3" => GGUFPhi3::get_num_of_layers(content),
                    "qwen2" | "qwen3" => {
                        GGUFQWen::get_num_of_layers(self.name.as_str() == "qwen3", content)
                    }
                    "glm4" => GGUFGLM4::get_num_of_layers(content),
                    _ => panic!("Model not supported!"),
                };
                nlayers.unwrap()
            };
            let handle =
                progress_worker(Some(num_subprogress), nlayers, Arc::clone(&reporter)).await;
            let mut file = try_api!(std::fs::File::open(path.clone()));
            let content = try_api!(
                gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path.clone()))
            );
            let (model, config, sep_style) = match self.name.as_str() {
                "llama" => {
                    let model = try_api!(GGUFLLaMa::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        s_cfg,
                        Arc::clone(&reporter),
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::LlamaGGUF(model), cfg, SeparatorStyle::Llama)
                }
                "llama3" => {
                    let model = try_api!(GGUFLLaMa::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        s_cfg,
                        Arc::clone(&reporter),
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::LlamaGGUF(model), cfg, SeparatorStyle::Llama3)
                }
                "phi3" => {
                    let model = try_api!(GGUFPhi3::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        s_cfg,
                        Arc::clone(&reporter),
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::Phi3GGUF(model), cfg, SeparatorStyle::Phi)
                }
                "qwen2" | "qwen3" => {
                    let model = try_api!(GGUFQWen::from_gguf(
                        self.name.as_str() == "qwen3",
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        s_cfg,
                        Arc::clone(&reporter),
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::QWenGGUF(model), cfg, SeparatorStyle::Qwen)
                }
                "glm4" => {
                    let model = try_api!(GGUFGLM4::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        s_cfg,
                        Arc::clone(&reporter),
                    ));
                    let cfg = model.get_config().clone();
                    (LLMModel::GLM4GGUF(model), cfg, SeparatorStyle::GLM)
                }
                _ => panic!("Model not supported!"),
            };
            handle.join().unwrap();
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
                "qwen2" | "qwen3" => {
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
                "gemma3" => {
                    let config: Gemma3Config = try_api!(serde_json::from_slice(&try_api!(
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
                "glm4" => {
                    let config: GLMConfig = try_api!(serde_json::from_slice(&try_api!(
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

            info!("Model {:?}", config);

            info!("Loading {} model.", self.name);
            let handle = progress_worker(
                Some(num_subprogress),
                config.num_hidden_layers,
                Arc::clone(&reporter),
            )
            .await;

            use crate::openai::distributed::Comm;
            #[cfg(feature = "nccl")]
            let id = if comm_id.is_some() {
                comm_id.unwrap()
            } else {
                cudarc::nccl::safe::Id::new().unwrap()
            };
            #[cfg(feature = "nccl")]
            assert!(
                (comm_id.is_some() && device_ids.len() == 1)
                    || (comm_id.is_none() && device_ids.len() >= 1)
            );
            let results: Vec<_> = device_ids
                .par_iter()
                .enumerate()
                .map(|(rank, dev_id)| {
                    #[cfg(feature = "nccl")]
                    let rank = if global_rank.is_some() {
                        global_rank.unwrap()
                    } else {
                        rank
                    };
                    #[cfg(feature = "nccl")]
                    let num_shards = if global_world_size.is_some() {
                        global_world_size.unwrap()
                    } else {
                        device_ids.len()
                    };

                    let paths: Vec<PathBuf> = paths.get_weight_filenames();
                    let device = crate::new_device(*dev_id).unwrap();
                    #[cfg(feature = "nccl")]
                    let _ = device.as_cuda_device().unwrap().bind_to_thread();

                    #[cfg(feature = "nccl")]
                    tracing::warn!(
                        "create nccl comm channel rank {}, shards {}, id {:?}",
                        rank,
                        num_shards,
                        id
                    );

                    #[cfg(feature = "nccl")]
                    let comm = Rc::new(
                        Comm::from_rank(
                            device.as_cuda_device().unwrap().cuda_device(),
                            rank,
                            num_shards,
                            id,
                        )
                        .unwrap(),
                    );
                    #[cfg(feature = "nccl")]
                    tracing::warn!("nccl comm created for rank {}", rank);

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
                            LLMModel::Llama(
                                Llama::load(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            ),
                            SeparatorStyle::Llama,
                        ),
                        "llama3" => (
                            LLMModel::Llama(
                                Llama::load(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            ),
                            SeparatorStyle::Llama3,
                        ),
                        "phi2" => (
                            LLMModel::Phi2(
                                Phi2::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            ),
                            SeparatorStyle::Phi,
                        ),
                        "phi3" => (
                            LLMModel::Phi3(
                                Phi::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            ),
                            SeparatorStyle::Phi,
                        ),
                        "qwen2" | "qwen3" => (
                            LLMModel::Qwen(
                                Qwen::new(
                                    self.name.as_str() == "qwen3",
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            ),
                            SeparatorStyle::Qwen,
                        ),
                        "gemma" => (
                            LLMModel::Gemma(
                                Gemma::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            ),
                            SeparatorStyle::Gemma,
                        ),
                        "gemma3" => (
                            LLMModel::Gemma3(
                                Gemma3::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            ),
                            SeparatorStyle::Gemma,
                        ),
                        "mistral" => (
                            LLMModel::Mistral(
                                Mistral::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            ),
                            SeparatorStyle::Mistral,
                        ),
                        "yi" => (
                            LLMModel::Yi(
                                Yi::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            ),
                            SeparatorStyle::Yi,
                        ),
                        "stablelm" => (
                            LLMModel::StableLM(
                                StableLM::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            ),
                            SeparatorStyle::StableLM,
                        ),
                        "glm4" => (
                            LLMModel::GLM4(
                                GLM4::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            ),
                            SeparatorStyle::Llama,
                        ),
                        "deepseek" => (
                            LLMModel::DeepSeek(
                                DeepSeek::load(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            ),
                            SeparatorStyle::Llama3,
                        ),
                        _ => panic!("Model not supported!"),
                    };

                    Ok((model, device, sep))
                })
                .collect();
            handle.join().unwrap();
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
                        return Err(e);
                    }
                }
            }

            (models, devices, config, sep_style[0].clone())
        };

        warn!("Done loading.");

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
            temperature: specific_args.temperature,
            top_k: specific_args.top_k,
            top_p: specific_args.top_p,
            thinking: Some(specific_args.thinking),
        };

        #[cfg(feature = "nccl")]
        let global_rank = global_rank.unwrap_or(0);
        #[cfg(not(feature = "nccl"))]
        let global_rank = local_rank.unwrap_or(0);

        let pipelines = models
            .into_iter()
            .enumerate()
            .map(|(rank, model)| {
                let logits_processor = {
                    LogitsProcessor::new(
                        SAMPLING_SEED,
                        pipeline_config.temperature,
                        specific_args.top_k,
                        specific_args.top_p,
                    )
                };
                let tokenizer_file = paths.get_tokenizer_filename();
                let (tokenizer, chat_template, bos_token, eos_token): (
                    Tokenizer,
                    Option<String>,
                    Option<String>,
                    Option<String>,
                ) = if tokenizer_file.display().to_string() != "" && Path::exists(&tokenizer_file) {
                    let tokenizer = Tokenizer::from_file(tokenizer_file.clone())
                        .map_err(|x| APIError::new(x.to_string()))
                        .unwrap();

                    let tokenizer_cfg_file = paths.get_tokenizer_config_filename();
                    let (chat_template, bos, eos) = if Path::exists(&tokenizer_cfg_file) {
                        let tokenizer_cfg: Option<String> =
                            std::fs::read_to_string(tokenizer_cfg_file).ok();
                        let cfg_tokenizer: TokenizerConfig =
                            serde_json::from_str(tokenizer_cfg.unwrap().as_str()).unwrap();
                        let bos = if cfg_tokenizer.bos_token.is_some() {
                            match cfg_tokenizer.bos_token.unwrap() {
                                BosEosToken(Either::Left(Some(id))) => Some(id),
                                BosEosToken(Either::Right(Some(content))) => content.content.clone(),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        let eos = if cfg_tokenizer.eos_token.is_some() {
                            match cfg_tokenizer.eos_token.unwrap() {
                                BosEosToken(Either::Left(Some(id))) => Some(id),
                                BosEosToken(Either::Right(Some(content))) => content.content.clone(),
                                _ => None,
                            }
                        } else {
                            None
                        };

                        (cfg_tokenizer.chat_template, bos, eos)
                    } else {
                        (None, None, None)
                    };
                    (tokenizer, chat_template, bos, eos)
                } else if quant.is_some() && matches!(quant.as_ref().unwrap().as_str(), "ggml" | "gguf") {
                    use crate::backend::gguf::{get_gguf_info, Content, GGUFInfo};
                    let filename = paths.get_weight_filenames()[0].clone();
                    let mut reader = std::fs::File::open(filename).unwrap();
                    let mut readers = vec![&mut reader];
                    let content = Content::from_readers(&mut readers).unwrap();
                    let GGUFInfo {
                        tokenizer,
                        bos,
                        eos,
                        unk: _,
                        context_length: _,
                        chat_template,
                    } = get_gguf_info(&content).unwrap();
                    (
                        tokenizer,
                        chat_template,
                        bos,
                        eos,
                    )
                } else {
                    panic!("Missing tokenizer file!");
                };

                if rank == 0 && global_rank == 0 {
                    if chat_template.is_some() {
                        info!("Chat Template {} \n", chat_template.as_ref().unwrap());
                    } else {
                        info!("Warning: Missing tokenizer_config.json \n Warning: Chat Template not found, use built-in template which may not correct!");
                    }
                    info!("{:?}", pipeline_config);
                    info!("{:?}", specific_args);
                }

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
                        if let Some(token) = tokenizer.get_vocab(true).get(stop).copied() {
                            stop_token_ids.push(token)
                        };
                    }
                }

                if let Some(eos) = &eos_token {
                    if let Some(token) = tokenizer.get_vocab(true).get(eos).copied() {
                        if !stop_token_ids.contains(&token) {
                            stop_token_ids.push(token)
                        }
                    };
                }

                if let Some(template) = chat_template.as_ref() {
                    if template.contains("<|eom_id|>") {
                        tracing::warn!("custom stop token <|eom_id|> in chat template");
                        stop_token_ids.push(128008);
                    }
                    if template.contains("<|eot_id|>") {
                        tracing::warn!("custom stop token <|eot_id|> in chat template");
                        stop_token_ids.push(128009);
                    }
                    if template.contains("<|end|>") {
                        tracing::warn!("custom stop token <|end|> in chat template");
                        if let Some(token) = tokenizer.get_vocab(true).get("<|end|>").copied() {
                            stop_token_ids.push(token);
                        }
                    }
                }

                if stop_token_ids.is_empty() {
                    //if no eos_token defined in the config, use default
                    if let Some(token) = tokenizer.get_vocab(true).get("<|endoftext|>").copied() {
                        stop_token_ids.push(token);
                    }
                    if let Some(token) = tokenizer.get_vocab(true).get("<|end|>").copied() {
                        stop_token_ids.push(token);
                    } else if stop_token_ids.is_empty() {
                        let token = tokenizer.token_to_id(EOS_TOKEN).unwrap_or(0);
                        stop_token_ids.push(token);
                    }
                }
                tracing::warn!("stop_token_ids {:?}", stop_token_ids);
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
                    stream_decoders: RwLock::new(HashMap::new()),
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
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::Phi2(phi) => phi
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::Phi3(phi) => phi
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::Qwen(qwen) => qwen
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::Gemma(gemma) => gemma
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::Gemma3(gemma3) => gemma3
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::Mistral(mistral) => mistral
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::Yi(yi) => yi
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::StableLM(stablelm) => stablelm
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::GLM4(glm4) => glm4
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::DeepSeek(deepseek) => deepseek
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::Phi3GGUF(phi3) => phi3
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::LlamaGGUF(llama) => llama
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::QWenGGUF(qwen) => qwen
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
                .map_err(APIError::from),
            LLMModel::GLM4GGUF(glm4) => glm4
                .forward(&input_tokens, input_positions, kv_cache, input_metadata)
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
                .apply_batch_repeat_penalty(logits, panalties, reference_tokens)
                .unwrap()
        } else {
            logits.to_owned()
        };

        let group_ids: Vec<usize> = groups.iter().map(|group| group.group_id).collect();
        let param = &groups[0].sampling_params;
        let sampling_params =
            if param.temperature.is_some() && (param.top_k.is_some() || param.top_p.is_some()) {
                Some(param.to_owned())
            } else {
                None
            };

        let next_tokens = self
            .logits_processor
            .sample(&logits, &sampling_params)
            .unwrap();
        let result: Vec<TokenOrFinishReason> = next_tokens
            .into_par_iter()
            .enumerate()
            .map(|(i, next_token)| {
                let group_id = group_ids[i];
                let mut text = "".to_string();
                let mut decoder_map = self.stream_decoders.write().unwrap();
                match decoder_map.get_mut(&group_id) {
                    Some(decoder) => {
                        if let Some(output) = decoder.step(next_token) {
                            text = output
                        }
                    }
                    _ => {
                        let leaked: &'static _ = Box::leak(Box::new(self.tokenizer.clone()));
                        let decoder = leaked.decode_stream(false);
                        let wrapped = super::StreamWithTokenizer {
                            _tokenizer: unsafe { Box::from_raw(leaked as *const _ as *mut _) },
                            stream: decoder,
                        };
                        let mut boxed_decoder: Box<dyn super::DecodeStreamTrait + Send + Sync> =
                            Box::new(wrapped);
                        if let Some(output) = boxed_decoder.step(next_token) {
                            text = output
                        }
                        //stream decoder for the new request
                        decoder_map.insert(group_id, boxed_decoder);
                    }
                }

                let custom_stop_token_match = !custom_stop_tokens[i].is_empty()
                    && custom_stop_tokens[i].contains(&text.trim().to_string());

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

    pub fn tokenizer(&self) -> &Tokenizer {
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
            LLMModel::Qwen(qwen) => qwen.get_config().clone(),
            LLMModel::Gemma(gemma) => gemma.get_config().clone(),
            LLMModel::Gemma3(gemma3) => gemma3.get_config().clone(),
            LLMModel::Mistral(mistral) => mistral.get_config().clone(),
            LLMModel::Yi(yi) => yi.get_config().clone(),
            LLMModel::StableLM(stablelm) => stablelm.get_config().clone(),
            LLMModel::GLM4(glm4) => glm4.get_config().clone(),
            LLMModel::DeepSeek(deepseek) => deepseek.get_config().clone(),
            LLMModel::Phi3GGUF(phi3) => phi3.get_config().clone(),
            LLMModel::LlamaGGUF(llama) => llama.get_config().clone(),
            LLMModel::QWenGGUF(qwen) => qwen.get_config().clone(),
            LLMModel::GLM4GGUF(glm4) => glm4.get_config().clone(),
        }
    }

    pub fn get_dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn reset_decoder(&mut self) {
        let mut map = self.stream_decoders.write().unwrap();
        map.clear();
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

unsafe impl Send for DefaultPipeline {}
unsafe impl Sync for DefaultPipeline {}
