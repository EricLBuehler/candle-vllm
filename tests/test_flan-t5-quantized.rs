#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::io::Write;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use candle_transformers::models::quantized_t5 as t5;

use anyhow::{Error as E, Result};
use axum::response::IntoResponse;
use axum::{
    extract::Json,
    http::{self, Method},
    routing::post,
    Router,
};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_vllm::openai::responses::APIError;
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, api::sync::ApiRepo, Repo, RepoType};
use tokenizers::Tokenizer;
use tower_http::cors::{AllowOrigin, CorsLayer};

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    T5Small,
    FlanT5Small,
    FlanT5Base,
    FlanT5Large,
    FlanT5Xl,
    FlanT5Xxl,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]

struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    // Enable/disable decoding.
    #[arg(long, default_value = "false")]
    disable_cache: bool,

    /// Use this prompt, otherwise compute sentence similarities.
    // #[arg(long)]
    // prompt: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The model size to use.
    #[arg(long, default_value = "flan-t5-large")]
    which: Which,
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: PathBuf,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<(Self, Tokenizer)> {
        let device = Device::Cpu;
        let default_model = "lmz/candle-quantized-t5".to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, "main".to_string()),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = match &args.config_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match args.which {
                Which::T5Small => api.get("config.json")?,
                Which::FlanT5Small => api.get("config-flan-t5-small.json")?,
                Which::FlanT5Base => api.get("config-flan-t5-base.json")?,
                Which::FlanT5Large => api.get("config-flan-t5-large.json")?,
                Which::FlanT5Xl => api.get("config-flan-t5-xl.json")?,
                Which::FlanT5Xxl => api.get("config-flan-t5-xxl.json")?,
            },
        };
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = match &args.weight_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match args.which {
                Which::T5Small => api.get("model.gguf")?,
                Which::FlanT5Small => api.get("model-flan-t5-small.gguf")?,
                Which::FlanT5Base => api.get("model-flan-t5-base.gguf")?,
                Which::FlanT5Large => api.get("model-flan-t5-large.gguf")?,
                Which::FlanT5Xl => api.get("model-flan-t5-xl.gguf")?,
                Which::FlanT5Xxl => api.get("model-flan-t5-xxl.gguf")?,
            },
        };

        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_model(&self) -> Result<t5::T5ForConditionalGeneration> {
        let vb = t5::VarBuilder::from_gguf(&self.weights_filename, &self.device)?;
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    }

    fn get_local_or_remote_file(filename: &str, api: &ApiRepo) -> Result<PathBuf> {
        let local_filename = std::path::PathBuf::from(filename);
        if local_filename.exists() {
            Ok(local_filename)
        } else {
            Ok(api.get(filename)?)
        }
    }
}
fn generate_answer(_prompt: String, args: &Args) -> Result<String> {
    let mut generated_text = String::new();

    let (_builder, mut _tokenizer) = T5ModelBuilder::load(args)?;
    let device = &_builder.device;
    let _tokenizer = _tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let _tokens = _tokenizer
        .encode(_prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let input_token_ids = Tensor::new(&_tokens[..], device)?.unsqueeze(0)?;
    let mut model = _builder.build_model()?;
    let mut output_token_ids = [_builder.config.pad_token_id as u32].to_vec();
    let temperature = 0.8f64;

    let mut logits_processor = LogitsProcessor::new(299792458, Some(temperature), None);
    let encoder_output = model.encode(&input_token_ids)?;

    let start = std::time::Instant::now();

    for index in 0.. {
        if output_token_ids.len() > 512 {
            break;
        }
        let decoder_token_ids = if index == 0 || !_builder.config.use_cache {
            Tensor::new(output_token_ids.as_slice(), device)?.unsqueeze(0)?
        } else {
            let last_token = *output_token_ids.last().unwrap();
            Tensor::new(&[last_token], device)?.unsqueeze(0)?
        };
        let logits = model
            .decode(&decoder_token_ids, &encoder_output)?
            .squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = output_token_ids.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &output_token_ids[start_at..],
            )?
        };

        let next_token_id = logits_processor.sample(&logits)?;
        if next_token_id as usize == _builder.config.eos_token_id {
            break;
        }
        output_token_ids.push(next_token_id);
        if let Some(text) = _tokenizer.id_to_token(next_token_id) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            generated_text.push_str(&text);
            print!("{}", text);
            std::io::stdout().flush()?;
        }
    }
    let dt = start.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)\n",
        output_token_ids.len(),
        output_token_ids.len() as f64 / dt.as_secs_f64(),
    );

    Ok(generated_text)
}

// request struct
#[derive(Deserialize)]
pub struct Request {
    prompt: String,
}

#[derive(Serialize)]
pub struct Response {
    answer: String,
}

impl IntoResponse for Response {
    fn into_response(self) -> axum::response::Response {
        Json(self.answer).into_response()
    }
}

#[utoipa::path(
    post,
    tag = "test",
    path = "/v1/chat/completions",
    request_body = Request,
    responses((status = 200, description = "Chat completions"))
)]
pub async fn generate(req_body: Json<Request>) -> axum::response::Response {
    let args = Args::parse();
    let generated_answer = generate_answer(req_body.prompt.clone(), &args);
    Json(Response {
        answer: generated_answer.unwrap(),
    })
    .into_response()
}

#[tokio::main]
async fn main() -> Result<(), APIError> {
    let allow_origin = AllowOrigin::any();
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    let app = Router::new()
        .layer(cors_layer)
        .route("/v1/chat/completions", post(generate));

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:2000"))
        .await
        .map_err(|e| APIError::new(e.to_string()))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| APIError::new(e.to_string()))
}
