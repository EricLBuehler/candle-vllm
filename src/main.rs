use axum::{
    http::{self, Method},
    routing::post,
    Router,
};
use candle_core::{DType, Device};
use candle_vllm::openai::openai_server::chat_completions;
use candle_vllm::openai::pipelines::llm_engine::LLMEngine;
use candle_vllm::openai::pipelines::pipeline::DefaultModelPaths;
use candle_vllm::openai::responses::APIError;
use candle_vllm::openai::OpenAIServerData;
use candle_vllm::scheduler::cache_engine::CacheConfig;
use candle_vllm::scheduler::SchedulerConfig;
use candle_vllm::{get_model_loader, hub_load_local_safetensors, ModelSelected};
use clap::Parser;
use std::{path::PathBuf, sync::Arc};
const SIZE_IN_MB: usize = 1024 * 1024;
use candle_vllm::openai::models::Config;
use std::path::Path;
use tokio::sync::Notify;
use tower_http::cors::{AllowOrigin, CorsLayer};
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Huggingface token environment variable (optional). If not specified, load using hf_token_path.
    #[arg(long)]
    hf_token: Option<String>,

    /// Huggingface token file (optional). If neither `hf_token` or `hf_token_path` are specified this is used with the value
    /// of `~/.cache/huggingface/token`
    #[arg(long)]
    hf_token_path: Option<String>,

    /// Port to serve on (localhost:port)
    #[arg(long)]
    port: u16,

    /// Set verbose mode (print all requests)
    #[arg(long)]
    verbose: bool,

    #[clap(subcommand)]
    command: ModelSelected,

    /// Maximum number of sequences to allow
    #[arg(long, default_value_t = 256)]
    max_num_seqs: usize,

    /// Size of a block
    #[arg(long, default_value_t = 32)]
    block_size: usize,

    /// if weight_path is passed, it will ignore the model_id
    #[arg(long)]
    model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online), path must include last "/"
    #[arg(long)]
    weight_path: Option<String>,

    /// The quantized weight file name (for gguf/ggml file)
    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Available GPU memory for kvcache (MB)
    #[arg(long, default_value_t = 4096)]
    kvcache_mem_gpu: usize,

    /// Available CPU memory for kvcache (MB)
    #[arg(long, default_value_t = 4096)]
    kvcache_mem_cpu: usize,

    /// Record conversation (default false, the client need to record chat history)
    #[arg(long)]
    record_conversation: bool,

    #[arg(long, value_delimiter = ',')]
    device_ids: Option<Vec<usize>>,
}

fn get_cache_config(
    kvcache_mem_gpu: usize,
    kvcache_mem_cpu: usize,
    block_size: usize,
    config: &Config,
    num_shards: usize,
) -> CacheConfig {
    let dsize = config.kv_cache_dtype.size_in_bytes();
    let num_gpu_blocks = kvcache_mem_gpu * SIZE_IN_MB
        / dsize
        / block_size
        / (config.num_key_value_heads / num_shards)
        / config.get_head_size()
        / config.num_hidden_layers
        / 2;
    let num_cpu_blocks = kvcache_mem_cpu * SIZE_IN_MB
        / dsize
        / block_size
        / (config.num_key_value_heads / num_shards)
        / config.get_head_size()
        / config.num_hidden_layers
        / 2;
    CacheConfig {
        block_size: block_size,
        num_gpu_blocks: Some(num_gpu_blocks),
        num_cpu_blocks: Some(num_cpu_blocks),
        fully_init: true,
        dtype: config.kv_cache_dtype,
    }
}

#[tokio::main]
async fn main() -> Result<(), APIError> {
    let args = Args::parse();
    let (loader, model_id, quant) = get_model_loader(args.command, args.model_id.clone());
    if args.model_id.is_none() {
        println!("No model id specified, using the default model or specified in the weight_path!");
    }

    let paths = match (&args.weight_path, &args.weight_file) {
        //model in a folder (safetensor format, huggingface folder structure)
        (Some(path), None) => DefaultModelPaths {
            tokenizer_filename: Path::new(path).join("tokenizer.json"),
            tokenizer_config_filename: Path::new(path).join("tokenizer_config.json"),
            config_filename: Path::new(path).join("config.json"),
            filenames: if Path::new(path)
                .join("model.safetensors.index.json")
                .exists()
            {
                hub_load_local_safetensors(path, "model.safetensors.index.json").unwrap()
            } else {
                //a single weight file case
                let mut safetensors_files = Vec::<std::path::PathBuf>::new();
                safetensors_files.insert(0, Path::new(path).join("model.safetensors"));
                safetensors_files
            },
        },
        //model in a quantized file (gguf/ggml format)
        (Some(path), Some(file)) => DefaultModelPaths {
            tokenizer_filename: {
                //we need to download tokenizer for the ggufl/ggml model
                let api = hf_hub::api::sync::Api::new().unwrap();
                let api = api.model(model_id.clone());
                api.get("tokenizer.json").unwrap()
            },
            tokenizer_config_filename: {
                let api = hf_hub::api::sync::Api::new().unwrap();
                let api = api.model(model_id.clone());
                match api.get("tokenizer_config.json") {
                    Ok(f) => f,
                    _ => "".into(),
                }
            },
            config_filename: PathBuf::new(),
            filenames: if Path::new(path).join(file).exists() {
                vec![Path::new(path).join(file).into()]
            } else {
                panic!("Model file not found {}", file);
            },
        },
        _ => {
            if args.hf_token.is_none() && args.hf_token_path.is_none() {
                //no token provided
                let token_path = format!(
                    "{}/.cache/huggingface/token",
                    dirs::home_dir()
                        .ok_or(APIError::new_str("No home directory"))?
                        .display()
                );
                if !Path::new(&token_path).exists() {
                    //also no token cache
                    use std::io::Write;
                    let mut input_token = String::new();
                    println!("Please provide your huggingface token to download model:\n");
                    std::io::stdin()
                        .read_line(&mut input_token)
                        .expect("Failed to read token!");
                    std::fs::create_dir_all(Path::new(&token_path).parent().unwrap()).unwrap();
                    let mut output = std::fs::File::create(token_path).unwrap();
                    write!(output, "{}", input_token.trim()).expect("Failed to save token!");
                }
            }
            loader.download_model(model_id, None, args.hf_token, args.hf_token_path)?
        }
    };

    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => panic!("Unsupported dtype {dtype}"),
        None => DType::BF16,
    };

    let device_ids: Vec<usize> = match args.device_ids {
        Some(ids) => ids,
        _ => vec![0usize],
    };
    let num_shards = device_ids.len();
    use candle_vllm::scheduler::cache_engine::CacheEngine;
    let (default_pipelines, pipeline_config) =
        loader.load_model(paths, dtype, &quant, device_ids).await?;

    let mut config: Option<Config> = None;
    let mut cache_config: Option<CacheConfig> = None;

    let pipelines = default_pipelines
        .into_iter()
        .map(|pipeline| {
            let cfg = pipeline.get_model_config();
            let cache_cfg = get_cache_config(
                args.kvcache_mem_gpu,
                args.kvcache_mem_cpu,
                args.block_size,
                &cfg,
                num_shards,
            );
            let cache_engine = CacheEngine::new(
                &cfg,
                &cache_cfg,
                cache_cfg.dtype,
                &pipeline.device(),
                num_shards,
            )
            .unwrap();
            if config.is_none() {
                config = Some(cfg.clone());
            }
            if cache_config.is_none() {
                cache_config = Some(cache_cfg.clone());
            }
            (pipeline.rank(), (pipeline, cache_engine))
        })
        .collect();

    let cache_config = cache_config.as_ref().unwrap().clone();
    let config = config.as_ref().unwrap().clone();
    println!("Cache config {:?}", cache_config);
    let finish_notify = Arc::new(Notify::new());
    let llm_engine = LLMEngine::new(
        pipelines,
        SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
        },
        &cache_config,
        &config,
        Arc::new(Notify::new()),
        finish_notify.clone(),
    )?;

    let server_data = OpenAIServerData {
        pipeline_config,
        model: llm_engine,
        record_conversation: args.record_conversation,
        device: Device::Cpu,
        finish_notify: finish_notify.clone(),
    };

    println!("Server started at http://127.0.0.1:{}.", args.port);

    let allow_origin = AllowOrigin::any();
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    let app = Router::new()
        .layer(cors_layer)
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(Arc::new(server_data));

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.port))
        .await
        .map_err(|e| APIError::new(e.to_string()))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| APIError::new(e.to_string()))?;

    Ok(())
}
