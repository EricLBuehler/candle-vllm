use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::openai::{
    responses::APIError, sampling_params::SamplingParams, ModulePipeline, PipelineConfig,
    TokenizerWrapper,
};
use candle_core::{DType, Device, Tensor};
use candle_lora_transformers::varbuilder_utils::from_mmaped_safetensors;
use candle_transformers::{
    generation::LogitsProcessor,
    models::llama::{Cache, Llama, LlamaConfig},
};
use hf_hub::{
    api::sync::{ApiBuilder, ApiError},
    Repo, RepoType,
};
use tokenizers::Tokenizer;

const EOS_TOKEN: &str = "</s>";
const NAME: &str = "llama";
const SAMPLING_SEED: u64 = 299792458;

#[derive(Debug, Clone)]
pub struct LlamaSpecifcConfig {
    no_kv_cache: bool,
    repeat_last_n: usize,
    use_flash_attn: bool,
}

impl Default for LlamaSpecifcConfig {
    fn default() -> Self {
        Self {
            no_kv_cache: false,
            repeat_last_n: 64,
            use_flash_attn: false,
        }
    }
}

pub struct LlamaPipeline {
    llama: Llama,
    args: LlamaSpecifcConfig,
    cache: Cache,
    tokenizer: Tokenizer,
}

pub struct ModelPaths {
    tokenizer_filename: PathBuf,
    config_filename: PathBuf,
    filenames: Vec<PathBuf>,
}

impl LlamaPipeline {
    pub fn download_model<P: AsRef<Path>>(
        model_id: String,
        revision: Option<String>,
        hf_token: P,
    ) -> Result<ModelPaths, ApiError> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(fs::read_to_string(hf_token)?))
            .build()?;
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api.get("tokenizer.json")?;

        let config_filename = api.get("config.json")?;

        let mut filenames = vec![];
        for rfilename in api.info()?.siblings.iter().map(|x| x.rfilename.clone()) {
            let filename = api.get(&rfilename)?;
            filenames.push(filename);
        }

        Ok(ModelPaths {
            tokenizer_filename,
            config_filename,
            filenames,
        })
    }

    pub fn new_default(
        paths: ModelPaths,
        args: LlamaSpecifcConfig,
        dtype: DType,
        device: Device,
    ) -> Result<(Self, PipelineConfig), APIError> {
        let config: LlamaConfig = serde_json::from_slice(
            &std::fs::read(paths.config_filename).map_err(APIError::new_from_io_err)?,
        )
        .map_err(APIError::new_from_serde_err)?;
        let config = config.into_config(args.use_flash_attn);
        let vb = from_mmaped_safetensors(&paths.filenames, dtype, &device)
            .map_err(APIError::new_from_candle_err)?;

        let cache = Cache::new(!args.no_kv_cache, dtype, &config, &device)
            .map_err(APIError::new_from_candle_err)?;

        let llama = Llama::load(vb, &cache, &config).map_err(APIError::new_from_candle_err)?;

        let tokenizer = Tokenizer::from_file(paths.tokenizer_filename)
            .map_err(|x| APIError::new(x.to_string()))?;

        //max is https://huggingface.co/docs/transformers/model_doc/llama2#transformers.LlamaConfig.max_position_embeddings
        let pipeline_config = PipelineConfig {
            max_model_len: 4096,
        };
        Ok((
            Self {
                llama,
                args,
                cache,
                tokenizer,
            },
            pipeline_config,
        ))
    }
}

impl<'s> ModulePipeline<'s> for LlamaPipeline {
    fn forward(
        &mut self,
        xs: &tokenizers::Encoding,
        sampling: SamplingParams,
        device: Device,
    ) -> Result<String, APIError> {
        let mut tokens = xs.get_ids().to_vec();

        let eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN);

        let mut logits_processor = LogitsProcessor::new(
            SAMPLING_SEED,
            Some(sampling.temperature.try_into().unwrap()),
            Some(sampling.top_p.try_into().unwrap()),
        );

        let mut index_pos = 0;
        let mut index = 0;
        let mut result = "".to_string();
        loop {
            let context_size = if self.cache.use_kv_cache && index > 0 {
                1
            } else {
                tokens.len()
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &device)
                .map_err(APIError::new_from_candle_err)?
                .unsqueeze(0)
                .map_err(APIError::new_from_candle_err)?;
            let logits = self
                .llama
                .forward(&input, index_pos)
                .map_err(APIError::new_from_candle_err)?;
            let logits = logits.squeeze(0).map_err(APIError::new_from_candle_err)?;
            let logits = if sampling.repetition_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    sampling.repetition_penalty,
                    &tokens[start_at..],
                )
                .map_err(APIError::new_from_candle_err)?
            };
            index_pos += ctxt.len();

            let next_token = logits_processor
                .sample(&logits)
                .map_err(APIError::new_from_candle_err)?;
            tokens.push(next_token);

            // Extracting the last token as a string is complicated, here we just apply some simple
            // heuristics as it seems to work well enough for this example. See the following for more
            // details:
            // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                result.push_str(&text);
            }
            if Some(next_token) == eos_token_id {
                break;
            }

            index += 1;
        }

        Ok(result)
    }

    fn name(&self) -> String {
        NAME.to_string()
    }

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String> {
        &self.tokenizer
    }
}
