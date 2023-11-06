use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::openai::{
    responses::{APIError, ChatCompletionUsageResponse},
    sampling_params::SamplingParams,
    ModelLoader, ModelPaths, ModulePipeline, PipelineConfig, TokenizerWrapper,
};
use candle_core::{DType, Device, Tensor};
use candle_lora_transformers::varbuilder_utils::from_mmaped_safetensors;
use candle_sampling::logits_processor::{LogitsProcessor, SamplingMethod};
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
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

/// top-p, multinomial, and argmax sampling are implemented. Beam search is not implemented.
pub struct LlamaPipeline {
    llama: Llama,
    args: LlamaSpecifcConfig,
    cache: Cache,
    tokenizer: Tokenizer,
}

pub struct LlamaLoader;

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

impl<'a, P: AsRef<Path>> ModelLoader<'a, P> for LlamaLoader {
    fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<P>,
    ) -> Result<Box<dyn ModelPaths>, APIError> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(
                fs::read_to_string(hf_token.unwrap()).map_err(APIError::new_from_io_err)?,
            ))
            .build()
            .map_err(APIError::new_from_hf_err)?;
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .map_err(APIError::new_from_hf_err)?;

        let config_filename = api.get("config.json").map_err(APIError::new_from_hf_err)?;

        let mut filenames = vec![];
        for rfilename in api
            .info()
            .map_err(APIError::new_from_hf_err)?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| x.ends_with(".safetensors"))
        {
            let filename = api.get(&rfilename).map_err(APIError::new_from_hf_err)?;
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
        let args = LlamaSpecifcConfig::default();

        let config: LlamaConfig = serde_json::from_slice(
            &std::fs::read(paths.get_config_filename()).map_err(APIError::new_from_io_err)?,
        )
        .map_err(APIError::new_from_serde_err)?;
        let config = config.into_config(args.use_flash_attn);

        println!("Loading Llama model.");

        let vb = from_mmaped_safetensors(paths.get_weight_filenames(), dtype, &device, false)
            .map_err(APIError::new_from_candle_err)?;

        let cache = Cache::new(!args.no_kv_cache, dtype, &config, &device)
            .map_err(APIError::new_from_candle_err)?;

        let llama = Llama::load(vb, &cache, &config).map_err(APIError::new_from_candle_err)?;

        let tokenizer = Tokenizer::from_file(paths.get_tokenizer_filename())
            .map_err(|x| APIError::new(x.to_string()))?;

        println!("Done loading.");

        //max is https://huggingface.co/docs/transformers/model_doc/llama2#transformers.LlamaConfig.max_position_embeddings
        let pipeline_config = PipelineConfig {
            max_model_len: 4096,
        };
        Ok((
            Box::new(LlamaPipeline {
                llama,
                args,
                cache,
                tokenizer,
            }),
            pipeline_config,
        ))
    }
}

impl LlamaPipeline {
    fn forward_inner(
        &mut self,
        tokens: &mut Vec<u32>,
        sampling: &SamplingParams,
        device: &Device,
        eos_token_id: &Option<u32>,
        logits_processor: &mut LogitsProcessor,
    ) -> Result<(String, ChatCompletionUsageResponse), APIError> {
        let mut index_pos = 0;
        let mut index = 0;
        let mut result = "".to_string();
        let mut tokens_generated = 0;
        loop {
            let context_size = if self.cache.use_kv_cache && index > 0 {
                1
            } else {
                tokens.len()
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, device)
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
            tokens_generated += 1;

            // Extracting the last token as a string is complicated, here we just apply some simple
            // heuristics as it seems to work well enough for this example. See the following for more
            // details:
            // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                result.push_str(&text);
            }
            if &Some(next_token) == eos_token_id {
                break;
            }

            index += 1;
        }

        Ok((
            result,
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
        xs: &tokenizers::Encoding,
        sampling: SamplingParams,
        device: Device,
    ) -> Result<(Vec<String>, ChatCompletionUsageResponse), APIError> {
        let eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN);

        let mut logits_processor = LogitsProcessor::new(
            SAMPLING_SEED,
            Some(sampling.temperature.try_into().unwrap()),
            SamplingMethod::TopP(sampling.top_p.try_into().unwrap()),
        );

        let mut tokens_generated = 0;
        let mut choices = Vec::new();
        for _ in 0..sampling.n {
            let mut tokens = xs.get_ids().to_vec();

            let (result, tokens_gen) = self.forward_inner(
                &mut tokens,
                &sampling,
                &device,
                &eos_token_id,
                &mut logits_processor,
            )?;
            tokens_generated += tokens_gen.completion_tokens;
            choices.push(result);
        }

        Ok((
            choices,
            ChatCompletionUsageResponse {
                completion_tokens: tokens_generated,
                prompt_tokens: xs.len(),
                total_tokens: tokens_generated + xs.len(),
            },
        ))
    }

    fn name(&self) -> String {
        NAME.to_string()
    }

    fn tokenizer(&self) -> &dyn TokenizerWrapper<'s, String> {
        &self.tokenizer
    }
}
