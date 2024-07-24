use axum::{
    http::{self, Method},
    routing::post,
    Router,
};
use candle_core::{DType, Device};
use candle_examples;
use candle_vllm::openai::openai_server::chat_completions;
use candle_vllm::openai::pipelines::llm_engine::LLMEngine;
use candle_vllm::openai::pipelines::pipeline::DefaultModelPaths;
use candle_vllm::openai::responses::APIError;
use candle_vllm::openai::OpenAIServerData;
use candle_vllm::scheduler::cache_engine::CacheConfig;
use candle_vllm::scheduler::SchedulerConfig;
use candle_vllm::{get_model_loader, hub_load_local_safetensors, ModelSelected};
use clap::Parser;
use std::sync::Arc;
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
}

#[tokio::main]
async fn main() -> Result<(), APIError> {
    let args = Args::parse();
    let (loader, model_id) = get_model_loader(args.command, args.model_id.clone());
    if args.model_id.is_none() {
        println!("No model id specified, using the default model or specified in the weight_path!");
    }

    let paths = match &args.weight_path {
        Some(path) => Box::new(DefaultModelPaths {
            tokenizer_filename: (path.to_owned() + "tokenizer.json").into(),
            config_filename: (path.to_owned() + "config.json").into(),
            filenames: if Path::new(&(path.to_owned() + "model.safetensors.index.json")).exists() {
                hub_load_local_safetensors(path, "model.safetensors.index.json").unwrap()
            } else {
                //a single weight file case
                let mut safetensors_files = Vec::<std::path::PathBuf>::new();
                safetensors_files.insert(0, (path.to_owned() + "model.safetensors").into());
                safetensors_files
            },
        }),
        _ => loader.download_model(model_id, None, args.hf_token, args.hf_token_path)?,
    };

    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => panic!("Unsupported dtype {dtype}"),
        None => DType::BF16,
    };

    let device = candle_examples::device(args.cpu).unwrap();
    let model = loader.load_model(paths, dtype, device)?;
    let config: Config = model.0.get_model_config();
    let dsize = config.kv_cache_dtype.size_in_bytes();
    let num_gpu_blocks = args.kvcache_mem_gpu * SIZE_IN_MB
        / dsize
        / args.block_size
        / config.num_key_value_heads
        / config.get_head_size()
        / config.num_hidden_layers
        / 2;
    let num_cpu_blocks = args.kvcache_mem_cpu * SIZE_IN_MB
        / dsize
        / args.block_size
        / config.num_key_value_heads
        / config.get_head_size()
        / config.num_hidden_layers
        / 2;
    let cache_config = CacheConfig {
        block_size: args.block_size,
        num_gpu_blocks: Some(num_gpu_blocks),
        num_cpu_blocks: Some(num_cpu_blocks),
        fully_init: true,
        dtype: config.kv_cache_dtype,
    };
    println!("Cache config {:?}", cache_config);
    let finish_notify = Arc::new(Notify::new());
    let llm_engine = LLMEngine::new(
        model.0,
        SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
        },
        cache_config,
        Arc::new(Notify::new()),
        finish_notify.clone(),
    )?;

    let server_data = OpenAIServerData {
        pipeline_config: model.1,
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
