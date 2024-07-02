use actix_web::middleware::Logger;
use actix_web::web::Data;
use actix_web::{App, HttpServer};
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
use futures::lock::Mutex;
use std::sync::Arc;
const SIZE_IN_MB: usize = 1024 * 1024;

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

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
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

#[actix_web::main]
async fn main() -> Result<(), APIError> {
    let args = Args::parse();
    let (loader, model_id) = get_model_loader(args.command);

    let paths = match &args.weight_path {
        Some(path) => Box::new(DefaultModelPaths {
            tokenizer_filename: (path.to_owned() + "tokenizer.json").into(),
            config_filename: (path.to_owned() + "config.json").into(),
            filenames: hub_load_local_safetensors(path, "model.safetensors.index.json").unwrap(),
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

    let dsize = dtype.size_in_bytes();
    let device = candle_examples::device(args.cpu).unwrap();
    let model = loader.load_model(paths, dtype, device)?;
    let config = model.0.get_model_config();
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
    };
    println!("Cache config {:?}", cache_config);

    let llm_engine = LLMEngine::new(
        model.0,
        SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
        },
        cache_config,
    )?;

    let server_data = OpenAIServerData {
        pipeline_config: model.1,
        model: Arc::new(Mutex::new(llm_engine)),
        record_conversation: args.record_conversation,
        device: Device::Cpu,
    };

    println!("Server started at http://127.0.0.1:{}.", args.port);
    if args.verbose {
        env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

        HttpServer::new(move || {
            App::new()
                .wrap(Logger::default())
                .service(chat_completions)
                .app_data(Data::new(server_data.clone()))
        })
        .bind(("127.0.0.1", args.port))
        .map_err(|e| APIError::new(e.to_string()))?
        .run()
        .await
        .map_err(|e| APIError::new(e.to_string()))?;
    } else {
        HttpServer::new(move || {
            App::new()
                .service(chat_completions)
                .app_data(Data::new(server_data.clone()))
        })
        .bind(("127.0.0.1", args.port))
        .map_err(|e| APIError::new(e.to_string()))?
        .run()
        .await
        .map_err(|e| APIError::new(e.to_string()))?;
    }

    Ok(())
}
