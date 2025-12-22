use super::{get_token, TokenOrFinishReason};
use crate::backend::gguf;
#[cfg(all(feature = "cuda", feature = "graph"))]
use crate::backend::graph::{CudaGraphFn, CudaGraphWrapper, GraphCapturer, ModelFn};
use crate::backend::progress::{progress_worker, ProgressReporter};
use crate::openai::logits_processor::LogitsProcessor;
use crate::openai::models::TokenID;
use crate::openai::requests::StopTokens;
use crate::openai::sampling_params::{GenerationConfig, Logprobs, TopLogprob};
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
            deepseek::DeepSeek, gemma::Gemma, gemma3::Gemma3, glm4::GLM4, llama::Llama,
            mistral::Mistral, phi2::Phi2, phi3::Phi, quantized_glm4::GGUFGLM4,
            quantized_llama::GGUFLLaMa, quantized_phi3::GGUFPhi3, quantized_qwen::GGUFQWen,
            quantized_qwen3_moe::GGUFQWenMoE, qwen::Qwen, qwen3_moe::Qwen3MoE, stable_lm::StableLM,
            yi::Yi, Config,
        },
        PipelineConfig,
    },
    InputMetadata,
};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};
use either::Either;
use either::Either::{Left, Right};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::Path;
pub use std::rc::Rc;
use std::{path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;
use tracing::{info, warn};

const EOS_TOKEN: &str = "</s>";
const SAMPLING_SEED: u64 = 299792458;
const MIN_GEN_TOKENS: usize = 128;
const MAX_GEN_TOKENS: usize = 16 * 1024;
pub enum LLMModel {
    Llama(Arc<Llama>),
    Phi2(Arc<Phi2>),
    Phi3(Arc<Phi>),
    Qwen(Arc<Qwen>),
    Qwen3MoE(Arc<Qwen3MoE>),
    Gemma(Arc<Gemma>),
    Gemma3(Arc<Gemma3>),
    Mistral(Arc<Mistral>),
    Yi(Arc<Yi>),
    StableLM(Arc<StableLM>),
    GLM4(Arc<GLM4>),
    DeepSeek(Arc<DeepSeek>),
    LlamaGGUF(Arc<GGUFLLaMa>),
    Phi3GGUF(Arc<GGUFPhi3>),
    QWenGGUF(Arc<GGUFQWen>),
    QWenGGUFMoE(Arc<GGUFQWenMoE>),
    GLM4GGUF(Arc<GGUFGLM4>),
}

/// top-p, multinomial, and argmax sampling are implemented. Beam search is not implemented.
pub struct DefaultPipeline {
    pub model: LLMModel,
    pub tokenizer: Tokenizer,
    pub logits_processor: LogitsProcessor,
    pub conversation: DefaultConversation,
    pub name: String,
    pub dtype: DType,
    pub device: Device,
    pub stop_token_ids: Vec<u32>,
    pub rank: usize,
    pub stream_decoders: RwLock<super::StreamDecoderMap>,
    #[cfg(all(feature = "cuda", feature = "graph"))]
    pub capturer: GraphCapturer<CudaGraphWrapper<CudaGraphFn>>,
}

pub struct DefaultLoader {
    model_id: Option<String>,
    weight_path: Option<String>,
    weight_file: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DefaultModelPaths {
    pub tokenizer_filename: PathBuf,
    pub tokenizer_config_filename: PathBuf,
    pub config_filename: PathBuf,
    pub generation_config_filename: PathBuf,
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
    fn get_generation_config_filename(&self) -> PathBuf {
        self.generation_config_filename.clone()
    }
}

impl DefaultLoader {
    pub fn new(
        model_id: Option<String>,
        weight_path: Option<String>,
        weight_file: Option<String>,
    ) -> Self {
        Self {
            model_id,
            weight_path,
            weight_file,
        }
    }
}

impl DefaultLoader {
    pub fn prepare_model_weights(
        &self,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<(DefaultModelPaths, bool)> {
        let (paths, gguf): (DefaultModelPaths, bool) = match (
            &self.model_id,
            &self.weight_path,
            &self.weight_file,
        ) {
            //model in a folder (safetensor format, huggingface folder structure)
            (None, Some(path), None) => (
                DefaultModelPaths {
                    tokenizer_filename: Path::new(path).join("tokenizer.json"),
                    tokenizer_config_filename: Path::new(path).join("tokenizer_config.json"),
                    config_filename: Path::new(path).join("config.json"),
                    filenames: if Path::new(path)
                        .join("model.safetensors.index.json")
                        .exists()
                    {
                        crate::hub_load_local_safetensors(path, "model.safetensors.index.json")?
                    } else {
                        //a single weight file case
                        let mut safetensors_files = Vec::<std::path::PathBuf>::new();
                        safetensors_files.insert(0, Path::new(path).join("model.safetensors"));
                        safetensors_files
                    },
                    generation_config_filename: if Path::new(path)
                        .join("generation_config.json")
                        .exists()
                    {
                        Path::new(path).join("generation_config.json")
                    } else {
                        "".into()
                    },
                },
                false,
            ),
            //model in a quantized file (gguf/ggml format)
            (None, path, Some(file)) => (
                DefaultModelPaths {
                    tokenizer_filename: PathBuf::new(),
                    tokenizer_config_filename: PathBuf::new(),
                    config_filename: PathBuf::new(),
                    filenames: {
                        let path = path.clone().unwrap_or("".to_string());
                        if Path::new(&path).join(file).exists() {
                            vec![Path::new(&path).join(file)]
                        } else {
                            panic!("Model file not found {file}");
                        }
                    },
                    generation_config_filename: "".into(),
                },
                true,
            ),
            (Some(_), None, Some(_)) => (self.download_gguf_model(None)?, true),
            (Some(_), None, None) => {
                //try download model anonymously
                let loaded = self.download_model(None, hf_token.clone(), hf_token_path.clone());
                if loaded.is_ok() {
                    (loaded.unwrap(), false)
                } else {
                    //if it's failed, try using huggingface token
                    info!("Try request model using cached huggingface token...");
                    if hf_token.is_none() && hf_token_path.is_none() {
                        //no token provided
                        let token_path = format!(
                            "{}/.cache/huggingface/token",
                            dirs::home_dir().unwrap().display()
                        );
                        if !Path::new(&token_path).exists() {
                            //also no token cache
                            use std::io::Write;
                            let mut input_token = String::new();
                            warn!("Unable to request model, please provide your huggingface token to download model:\n");
                            std::io::stdin()
                                .read_line(&mut input_token)
                                .expect("Failed to read token!");
                            std::fs::create_dir_all(Path::new(&token_path).parent().unwrap())
                                .unwrap();
                            let mut output = std::fs::File::create(token_path).unwrap();
                            write!(output, "{}", input_token.trim())
                                .expect("Failed to save token!");
                        }
                    }
                    (
                        self.download_model(None, hf_token.clone(), hf_token_path.clone())?,
                        false,
                    )
                }
            }
            _ => {
                candle_core::bail!("No model id or weight_path/weight_file provided!\n***Tips***: \n \t For local model weights, \
                    `--w <path/to/folder>` for safetensors models or `--f <path/to/gguf/file>` for gguf models.\n \
                    \t For remote safetensor models, `--m <model_id>` to download from HuggingFace hub. \
                    \n \t For remote gguf models, `--m <model_id> --f <weight_file>` to download from HuggingFace hub.");
            }
        };

        Ok((paths, gguf))
    }

    pub fn download_model(
        &self,
        revision: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<DefaultModelPaths> {
        assert!(self.model_id.is_some(), "No model id provided!");

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(get_token(hf_token, hf_token_path)?))
            .build()
            .map_err(candle_core::Error::wrap)?;
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            self.model_id.clone().unwrap(),
            RepoType::Model,
            revision.clone(),
        ));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .map_err(candle_core::Error::wrap)?;

        let config_filename = api.get("config.json").map_err(candle_core::Error::wrap)?;

        let tokenizer_config_filename = match api.get("tokenizer_config.json") {
            Ok(f) => f,
            _ => "".into(),
        };

        let generation_config_filename = match api.get("generation_config.json") {
            Ok(f) => f,
            _ => "".into(),
        };

        let mut filenames = vec![];
        for rfilename in api
            .info()
            .map_err(candle_core::Error::wrap)?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| x.ends_with(".safetensors"))
        {
            let filename = api.get(&rfilename).map_err(candle_core::Error::wrap)?;
            filenames.push(filename);
        }

        Ok(DefaultModelPaths {
            tokenizer_filename,
            tokenizer_config_filename,
            config_filename,
            filenames,
            generation_config_filename,
        })
    }

    pub fn download_gguf_model(&self, revision: Option<String>) -> Result<DefaultModelPaths> {
        assert!(self.model_id.is_some(), "No model id provided!");
        info!(
            "Downloading GGUF file {} from repo {}",
            self.weight_file.as_ref().unwrap(),
            self.model_id.as_ref().unwrap(),
        );
        let filename = self.weight_file.clone().unwrap();
        let api = hf_hub::api::sync::Api::new().unwrap();
        let revision = revision.unwrap_or("main".to_string());
        let mut filenames = vec![];
        let filename = api
            .repo(hf_hub::Repo::with_revision(
                self.model_id.clone().unwrap(),
                hf_hub::RepoType::Model,
                revision.to_string(),
            ))
            .get(filename.as_str())
            .map_err(candle_core::Error::wrap)?;
        filenames.push(filename);

        Ok(DefaultModelPaths {
            tokenizer_filename: "".into(),
            tokenizer_config_filename: "".into(),
            config_filename: "".into(),
            filenames,
            generation_config_filename: "".into(),
        })
    }

    //support loading in both multithreaded and multiprocess mode
    #[allow(unused_variables)]
    pub async fn load_model(
        &self,
        paths: DefaultModelPaths,
        dtype: DType,
        kv_cache_dtype: DType,
        gguf: bool,
        isq: Option<String>,
        block_size: usize,
        max_num_seqs: usize,
        device_ids: Vec<usize>, //pass only 1 device_id in multiprocess mode, otherwise, multiple device_ids in multithread mode
        #[cfg(feature = "nccl")] comm_id: Option<crate::openai::distributed::Id>, //must pass comm id in multiprocess mode
        local_rank: Option<usize>, //must pass current rank in multiprocess mode
        local_world_size: Option<usize>, //must pass the number of local devices used in multiprocess mode
        #[cfg(feature = "nccl")] global_rank: Option<usize>, //must pass current global rank in multi-node mode
        #[cfg(feature = "nccl")] global_world_size: Option<usize>, //must pass total number of devices used in multi-node mode
    ) -> Result<(Vec<Box<DefaultPipeline>>, PipelineConfig)> {
        let reporter = Arc::new(RwLock::new(ProgressReporter::new(local_rank.unwrap_or(0))));
        let num_subprogress = local_world_size.map_or(0, |n| n - 1);

        let (models, devices, config, sep_style) = if gguf {
            let device = crate::new_device(device_ids[0]).unwrap();
            let path = paths.get_weight_filenames()[0].clone();
            info!("Loading quantized model from file {}", path.display());
            let (arch, nlayers) = {
                let mut file = match std::fs::File::open(path.clone())
                    .map_err(candle_core::Error::wrap)
                {
                    Ok(file) => file,
                    Err(e) => {
                        tracing::error!("Failed to open gguf file {}: {}\n ***Tips: use `--w` to load safetensors models.", path.display(), e);
                        return Err(e);
                    }
                };
                let content = match gguf_file::Content::read(&mut file)
                    .map_err(|e| e.with_path(path.clone()))
                    .map_err(candle_core::Error::wrap)
                {
                    Ok(content) => content,
                    Err(e) => {
                        tracing::error!("Failed to open gguf file {}: {}\n ***Tips: use `--w` to load safetensors models.", path.display(), e);
                        return Err(e);
                    }
                };
                let (arch, nlayers) =
                    gguf::get_arch_and_num_of_layers(content).map_err(candle_core::Error::wrap)?;
                if !matches!(
                    arch.as_str(),
                    "llama"
                        | "llama3"
                        | "phi3"
                        | "qwen2"
                        | "qwen3"
                        | "qwen2moe"
                        | "qwen3moe"
                        | "glm4"
                ) {
                    panic!("Model arch {} not supported!", arch);
                } else {
                    info!("Quantized {} model has {} layers.", arch, nlayers,);
                }
                (arch, nlayers)
            };
            let handle =
                progress_worker(Some(num_subprogress), nlayers, Arc::clone(&reporter)).await;
            let mut file = std::fs::File::open(path.clone()).map_err(candle_core::Error::wrap)?;
            let content = gguf_file::Content::read(&mut file)
                .map_err(|e| e.with_path(path.clone()))
                .map_err(candle_core::Error::wrap)?;
            let (model, config, sep_style) = match arch.as_str() {
                "llama" => {
                    let model = GGUFLLaMa::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        kv_cache_dtype,
                        Arc::clone(&reporter),
                    )
                    .map_err(candle_core::Error::wrap)?;
                    let cfg = model.get_config().clone();
                    (
                        LLMModel::LlamaGGUF(Arc::new(model)),
                        cfg,
                        SeparatorStyle::Llama,
                    )
                }
                "llama3" => {
                    let model = GGUFLLaMa::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        kv_cache_dtype,
                        Arc::clone(&reporter),
                    )
                    .map_err(candle_core::Error::wrap)?;
                    let cfg = model.get_config().clone();
                    (
                        LLMModel::LlamaGGUF(Arc::new(model)),
                        cfg,
                        SeparatorStyle::Llama3,
                    )
                }
                "phi3" => {
                    let model = GGUFPhi3::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        kv_cache_dtype,
                        Arc::clone(&reporter),
                    )
                    .map_err(candle_core::Error::wrap)?;
                    let cfg = model.get_config().clone();
                    (
                        LLMModel::Phi3GGUF(Arc::new(model)),
                        cfg,
                        SeparatorStyle::Phi,
                    )
                }
                "qwen2" | "qwen3" => {
                    let model = GGUFQWen::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        kv_cache_dtype,
                        Arc::clone(&reporter),
                    )
                    .map_err(candle_core::Error::wrap)?;
                    let cfg = model.get_config().clone();
                    (
                        LLMModel::QWenGGUF(Arc::new(model)),
                        cfg,
                        SeparatorStyle::Qwen,
                    )
                }
                "qwen2moe" | "qwen3moe" => {
                    let model = GGUFQWenMoE::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        kv_cache_dtype,
                        Arc::clone(&reporter),
                    )
                    .map_err(candle_core::Error::wrap)?;
                    let cfg = model.get_config().clone();
                    (
                        LLMModel::QWenGGUFMoE(Arc::new(model)),
                        cfg,
                        SeparatorStyle::Qwen,
                    )
                }
                "glm4" => {
                    let model = GGUFGLM4::from_gguf(
                        &content,
                        &mut file,
                        &device,
                        dtype,
                        kv_cache_dtype,
                        Arc::clone(&reporter),
                    )
                    .map_err(candle_core::Error::wrap)?;
                    let cfg = model.get_config().clone();
                    (
                        LLMModel::GLM4GGUF(Arc::new(model)),
                        cfg,
                        SeparatorStyle::GLM,
                    )
                }
                _ => panic!("Model not supported!"),
            };
            handle.join().unwrap();
            (vec![model], vec![device], config.to_owned(), sep_style)
        } else {
            let cfile = paths.get_config_filename();
            let arch = Config::get_model_arch(&cfile)?;

            let mut config = match arch.as_str() {
                "LlamaForCausalLM" => Llama::load_config(&cfile, isq)?,
                "PhiForCausalLM" => Phi2::load_config(&cfile, isq)?,
                "Phi3ForCausalLM" => Phi::load_config(&cfile, isq)?,
                "Qwen2ForCausalLM" | "Qwen3ForCausalLM" => Qwen::load_config(&cfile, isq)?,
                "Qwen2MoeForCausalLM" | "Qwen3MoeForCausalLM" => {
                    Qwen3MoE::load_config(&cfile, isq)?
                }
                "Gemma2ForCausalLM" => Gemma::load_config(&cfile, isq)?,
                "Gemma3ForConditionalGeneration" => Gemma3::load_config(&cfile, isq)?,
                "MistralForCausalLM" => Mistral::load_config(&cfile, isq)?,
                "Mistral3ForConditionalGeneration" => Mistral::load_text_config(&cfile, isq)?,
                "yi" => Yi::load_config(&cfile, isq)?,
                "StableLmForCausalLM" => StableLM::load_config(&cfile, isq)?,
                "Glm4ForCausalLM" => GLM4::load_config(&cfile, isq)?,
                "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => {
                    DeepSeek::load_config(&cfile, isq)?
                }
                _ => panic!("Model not supported!"),
            };
            config.fp8_kvcache = Some(kv_cache_dtype == DType::U8);
            info!("Model {:?}", config);

            info!("Loading {} model.", arch);
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
                candle_core::cuda_backend::cudarc::nccl::safe::Id::new().unwrap()
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

                    let (model, sep) = match arch.as_str() {
                        "LlamaForCausalLM" => (
                            LLMModel::Llama(Arc::new(
                                Llama::load(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            )),
                            SeparatorStyle::Llama3,
                        ),
                        "PhiForCausalLM" => (
                            LLMModel::Phi2(Arc::new(
                                Phi2::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            )),
                            SeparatorStyle::Phi,
                        ),
                        "Phi3ForCausalLM" => (
                            LLMModel::Phi3(Arc::new(
                                Phi::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            )),
                            SeparatorStyle::Phi,
                        ),
                        "Qwen2ForCausalLM" | "Qwen3ForCausalLM" => (
                            LLMModel::Qwen(Arc::new(
                                Qwen::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            )),
                            SeparatorStyle::Qwen,
                        ),
                        "Qwen2MoeForCausalLM" | "Qwen3MoeForCausalLM" => (
                            LLMModel::Qwen3MoE(Arc::new(
                                Qwen3MoE::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            )),
                            SeparatorStyle::Qwen,
                        ),
                        "Gemma2ForCausalLM" => (
                            LLMModel::Gemma(Arc::new(
                                Gemma::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            )),
                            SeparatorStyle::Gemma,
                        ),
                        "Gemma3ForConditionalGeneration" => (
                            LLMModel::Gemma3(Arc::new(
                                Gemma3::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            )),
                            SeparatorStyle::Gemma,
                        ),
                        "MistralForCausalLM" | "Mistral3ForConditionalGeneration" => (
                            LLMModel::Mistral(Arc::new(
                                Mistral::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            )),
                            SeparatorStyle::Mistral,
                        ),
                        "yi" => (
                            LLMModel::Yi(Arc::new(
                                Yi::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            )),
                            SeparatorStyle::Yi,
                        ),
                        "StableLmForCausalLM" => (
                            LLMModel::StableLM(Arc::new(
                                StableLM::new(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            )),
                            SeparatorStyle::StableLM,
                        ),
                        "Glm4ForCausalLM" => (
                            LLMModel::GLM4(Arc::new(
                                GLM4::new(vb, &config, dtype, &device, comm, Arc::clone(&reporter))
                                    .unwrap(),
                            )),
                            SeparatorStyle::Llama,
                        ),
                        "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => (
                            LLMModel::DeepSeek(Arc::new(
                                DeepSeek::load(
                                    vb,
                                    &config,
                                    dtype,
                                    &device,
                                    comm,
                                    Arc::clone(&reporter),
                                )
                                .unwrap(),
                            )),
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
        let default_max_tokens = (config.max_seq_len / 5).clamp(MIN_GEN_TOKENS, MAX_GEN_TOKENS);

        let cfg_file = paths.get_generation_config_filename();
        let generation_cfg = if cfg_file.display().to_string() != "" && Path::exists(&cfg_file) {
            let str_cfg: Option<String> = std::fs::read_to_string(cfg_file).ok();
            let cfg: GenerationConfig = serde_json::from_str(str_cfg.unwrap().as_str()).unwrap();
            Some(cfg)
        } else {
            None
        };

        let pipeline_config = PipelineConfig {
            max_model_len: config.max_seq_len,
            default_max_tokens,
            generation_cfg,
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
                        None,
                        None,
                        None,
                        None,
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
                        .map_err(candle_core::Error::wrap).unwrap();

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
                } else if gguf {
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

                let conversation = DefaultConversation::new(
                    config.architectures.as_ref().unwrap()[0].clone(),
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
                );

                Box::new(DefaultPipeline::new(
                        model,
                        tokenizer,
                        logits_processor,
                        &config,
                        conversation,
                        dtype,
                        &devices[rank],
                        stop_token_ids,
                        rank,
                        #[cfg(all(feature = "cuda", feature = "graph"))]
                        block_size,
                        #[cfg(all(feature = "cuda", feature = "graph"))]
                        max_num_seqs,
                    ).unwrap()
                )
            })
            .collect();

        Ok((pipelines, pipeline_config))
    }
}

impl DefaultPipeline {
    pub fn new(
        model: LLMModel,
        tokenizer: Tokenizer,
        logits_processor: LogitsProcessor,
        config: &Config,
        conversation: DefaultConversation,
        dtype: DType,
        device: &Device,
        stop_token_ids: Vec<u32>,
        rank: usize,
        #[cfg(all(feature = "cuda", feature = "graph"))] block_size: usize,
        #[cfg(all(feature = "cuda", feature = "graph"))] max_num_seqs: usize,
    ) -> Result<Self> {
        #[cfg(all(feature = "cuda", feature = "graph"))]
        let wrapper = crate::graph_model_wrapper!(
            model,
            device,
            Llama,
            Phi2,
            Phi3,
            Qwen,
            Qwen3MoE,
            Gemma,
            Gemma3,
            Mistral,
            Yi,
            StableLM,
            GLM4,
            DeepSeek,
            LlamaGGUF,
            Phi3GGUF,
            QWenGGUF,
            QWenGGUFMoE,
            GLM4GGUF,
        );
        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            conversation,
            name: config.architectures.as_ref().unwrap()[0].clone(),
            dtype,
            device: device.clone(),
            stop_token_ids,
            rank,
            stream_decoders: RwLock::new(HashMap::new()),
            #[cfg(all(feature = "cuda", feature = "graph"))]
            capturer: GraphCapturer::new(
                wrapper,
                max_num_seqs,
                config.max_seq_len,
                block_size,
                config.hidden_size,
            ),
        })
    }

    pub fn forward(
        &self,
        input_tokens: Tensor,
        input_positions: &Tensor,
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", feature = "graph"))]
        if !input_metadata.is_prefill && self.capturer.is_captured(input_tokens.dim(0)?) {
            return self
                .capturer
                .replay(&input_tokens, &input_positions, &input_metadata);
        }

        match &self.model {
            LLMModel::Llama(llama) => {
                llama.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Phi2(phi) => {
                phi.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Phi3(phi) => {
                phi.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Qwen(qwen) => {
                qwen.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Qwen3MoE(qwen) => {
                qwen.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Gemma(gemma) => {
                gemma.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Gemma3(gemma3) => {
                gemma3.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Mistral(mistral) => {
                mistral.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Yi(yi) => {
                yi.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::StableLM(stablelm) => {
                stablelm.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::GLM4(glm4) => {
                glm4.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::DeepSeek(deepseek) => {
                deepseek.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::Phi3GGUF(phi3) => {
                phi3.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::LlamaGGUF(llama) => {
                llama.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::QWenGGUF(qwen) => {
                qwen.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::QWenGGUFMoE(qwen) => {
                qwen.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
            LLMModel::GLM4GGUF(glm4) => {
                glm4.forward(&input_tokens, input_positions, kv_cache, input_metadata)
            }
        }
    }

    pub fn sample(
        &mut self,
        logits: &Tensor,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Result<Vec<TokenOrFinishReason>> {
        let (
            tokens_generated,
            custom_stop_tokens,
            frequency_panalties,
            presence_panalties,
            reference_tokens,
        ): (
            Vec<i32>,
            Vec<Vec<String>>,
            Vec<f32>,
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

                let ref_tokens = if (sampling_params.frequency_penalty != 0.
                    || sampling_params.presence_penalty != 0.)
                    && sampling_params.repeat_last_n.unwrap_or(128) < generated
                {
                    let start_at = tokens
                        .len()
                        .saturating_sub(sampling_params.repeat_last_n.unwrap_or(128));
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
                    sampling_params.frequency_penalty,
                    sampling_params.presence_penalty,
                    ref_tokens,
                )
            })
            .collect::<Vec<(i32, Vec<String>, f32, f32, Vec<u32>)>>()
            .into_iter()
            .fold(
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                |mut acc, (gen, stop, penalty1, penalty2, ref_t)| {
                    acc.0.push(gen);
                    acc.1.push(stop);
                    acc.2.push(penalty1);
                    acc.3.push(penalty2);
                    acc.4.push(ref_t);
                    acc
                },
            );

        let logits = if frequency_panalties.iter().any(|&v| v != 1.0 && v != 0.)
            || presence_panalties.iter().any(|&v| v != 1.0 && v != 0.)
        {
            self.logits_processor.apply_batch_repeat_penalty(
                logits,
                frequency_panalties,
                presence_panalties,
                reference_tokens,
            )?
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

        let next_tokens = self.logits_processor.sample(&logits, &sampling_params)?;
        let result: Vec<TokenOrFinishReason> = next_tokens
            .into_par_iter()
            .enumerate()
            .map(|(i, next_token)| {
                let group_id = group_ids[i];
                let mut text = "".to_string();
                let mut decoder_map = self.stream_decoders.write();
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
                        token: next_token,
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
            LLMModel::Qwen3MoE(qwen) => qwen.get_config().clone(),
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
            LLMModel::QWenGGUFMoE(qwen) => qwen.get_config().clone(),
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
        let mut map = self.stream_decoders.write();
        map.clear();
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    #[cfg(all(feature = "cuda", feature = "graph"))]
    pub fn warmup_capture(&mut self, kv_caches: Option<&Vec<(Tensor, Tensor)>>) -> Result<()> {
        self.capturer.capture(&self.device, kv_caches)
    }
}

unsafe impl Send for DefaultPipeline {}
unsafe impl Sync for DefaultPipeline {}
