use axum::{
    http::{self, Method},
    routing::post,
    Router,
};
use candle_core::{DType, Device};
#[cfg(feature = "nccl")]
use candle_vllm::backend::heartbeat;
use candle_vllm::openai::openai_server::chat_completions;
use candle_vllm::openai::pipelines::llm_engine::LLMEngine;
use candle_vllm::openai::pipelines::pipeline::DefaultModelPaths;
use candle_vllm::openai::responses::APIError;
use candle_vllm::openai::OpenAIServerData;
use candle_vllm::scheduler::cache_engine::{CacheConfig, CacheEngine};
use candle_vllm::scheduler::SchedulerConfig;
use candle_vllm::{get_model_loader, hub_load_local_safetensors, ModelSelected};
use clap::Parser;
use std::{path::PathBuf, sync::Arc};
use tracing::{info, warn};
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
    #[arg(long, default_value_t = 128)]
    kvcache_mem_cpu: usize,

    /// Record conversation (default false, the client need to record chat history)
    #[arg(long)]
    record_conversation: bool,

    #[arg(long, value_delimiter = ',')]
    device_ids: Option<Vec<usize>>,

    /// Maximum waiting time for processing parallel requests (in milliseconds).
    /// A larger value means the engine can hold more requests and process them in a single generation call.
    #[arg(long, default_value_t = 500)]
    holding_time: usize,

    //Whether the program running in multiprocess or multithread model for parallel inference
    #[arg(long, default_value_t = false)]
    multi_process: bool,

    #[arg(long, default_value_t = false)]
    log: bool,
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
        / config.k_head_dim()
        / config.num_hidden_layers
        / 2;
    let num_cpu_blocks = kvcache_mem_cpu * SIZE_IN_MB
        / dsize
        / block_size
        / (config.num_key_value_heads / num_shards)
        / config.k_head_dim()
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

fn config_log(
    logger: ftail::Ftail,
    log_enable: bool,
    log_file: String,
) -> Result<(), ftail::error::FtailError> {
    if !log_enable {
        return Ok(());
    }
    use tracing::log::LevelFilter;
    let mut cfg_filter = LevelFilter::Warn;
    if let Ok(level) = std::env::var("RUST_LOG") {
        let log_level_names: [&str; 6] = ["OFF", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"];
        let log_levels: [LevelFilter; 6] = [
            LevelFilter::Off,
            LevelFilter::Error,
            LevelFilter::Warn,
            LevelFilter::Info,
            LevelFilter::Debug,
            LevelFilter::Trace,
        ];
        let level = level.to_uppercase();
        for (i, name) in log_level_names.to_vec().into_iter().enumerate() {
            if level.find(name).is_some() {
                cfg_filter = log_levels[i]
            }
        }
    };
    if std::fs::exists(&log_file).is_ok() {
        let _ = std::fs::remove_file(&log_file);
    }
    logger
        .console(cfg_filter)
        .single_file(log_file.as_str(), true, cfg_filter)
        .init()
}

#[tokio::main]
async fn main() -> Result<(), APIError> {
    let args = Args::parse();
    if !args.log {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    let (loader, model_id, quant) = get_model_loader(args.command, args.model_id.clone());
    if args.model_id.is_none() && args.weight_path.is_none() && args.weight_file.is_none() {
        info!("No model id specified, using the default model_id or specified in the weight_path to retrieve config files!");
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
        (path, Some(file)) => DefaultModelPaths {
            tokenizer_filename: PathBuf::new(),
            tokenizer_config_filename: PathBuf::new(),
            config_filename: PathBuf::new(),
            filenames: {
                let path = path.clone().unwrap_or("".to_string());
                if Path::new(&path).join(file).exists() {
                    vec![Path::new(&path).join(file).into()]
                } else {
                    panic!("Model file not found {}", file);
                }
            },
        },
        _ => {
            //try download model anonymously
            let loaded = loader.download_model(
                model_id.clone(),
                args.weight_file.clone(),
                quant.clone(),
                None,
                args.hf_token.clone(),
                args.hf_token_path.clone(),
            );
            if loaded.is_ok() {
                loaded.unwrap()
            } else {
                //if it's failed, try using huggingface token
                info!("Try request model using cached huggingface token...");
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
                        warn!("Unable to request model, please provide your huggingface token to download model:\n");
                        std::io::stdin()
                            .read_line(&mut input_token)
                            .expect("Failed to read token!");
                        std::fs::create_dir_all(Path::new(&token_path).parent().unwrap()).unwrap();
                        let mut output = std::fs::File::create(token_path).unwrap();
                        write!(output, "{}", input_token.trim()).expect("Failed to save token!");
                    }
                }
                loader.download_model(
                    model_id,
                    args.weight_file,
                    quant.clone(),
                    None,
                    args.hf_token,
                    args.hf_token_path,
                )?
            }
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
    let local_world_size = device_ids.len();
    let mut num_shards = local_world_size;
    #[cfg(not(feature = "nccl"))]
    assert!(
        num_shards == 1,
        "More than one shard was given, but NCCL is not enabled for parallel inference!"
    );
    let logger = ftail::Ftail::new();
    let mut port = args.port;
    #[cfg(feature = "nccl")]
    let (pipelines, global_rank, daemon_manager) = if args.multi_process {
        use candle_vllm::openai::communicator::init_subprocess;
        let (id, local_rank, global_rank, global_world_size, daemon_manager) =
            init_subprocess(device_ids.clone()).unwrap();
        if global_rank != 0 {
            port = port + global_rank as u16; //processes other than rank 0 use fake server port since they do not perform response
        }
        num_shards = global_world_size;
        let log_file = format!("candle-vllm-rank-{}.log", global_rank);
        let _ = config_log(logger, args.log, log_file);

        warn!("subprocess rank {} started!", global_rank);
        heartbeat::heartbeat_worker(Some(local_world_size - 1)).await;

        (
            loader
                .load_model(
                    paths,
                    dtype,
                    &quant,
                    vec![device_ids[local_rank]],
                    Some(id),
                    Some(local_rank),
                    Some(local_world_size),
                    Some(global_rank),
                    Some(global_world_size),
                )
                .await,
            global_rank,
            Some(daemon_manager),
        )
    } else {
        use candle_vllm::openai::communicator::DaemonManager;
        DaemonManager::set_master_rank(true); //master rank default for multithreaded mode
        let log_file = format!("candle-vllm-{}ranks.log", device_ids.len());
        let _ = config_log(logger, args.log, log_file);
        (
            loader
                .load_model(
                    paths, dtype, &quant, device_ids, None, None, None, None, None,
                )
                .await,
            0,
            None,
        )
    };

    #[cfg(feature = "nccl")]
    info!(
        "parallel model: {}!",
        if args.multi_process {
            "multiprocess"
        } else {
            "multithread"
        }
    );

    #[cfg(not(feature = "nccl"))]
    let (pipelines, global_rank) = {
        let log_file = format!("candle-vllm.log");
        let _ = config_log(logger, args.log, log_file);
        (
            loader
                .load_model(paths, dtype, &quant, device_ids, None, None)
                .await,
            0,
        )
    };

    let (default_pipelines, pipeline_config) = match pipelines {
        Err(e) => panic!("{:?}", e),
        Ok((p, c)) => (p, c),
    };
    let mut config: Option<Config> = None;
    let mut cache_config: Option<CacheConfig> = None;

    let pipelines = default_pipelines
        .into_iter()
        .map(|pipeline| {
            let cfg = pipeline.get_model_config();
            let cache_cfg = get_cache_config(
                args.kvcache_mem_gpu,
                args.kvcache_mem_cpu, //dummy 512MB for cpu
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
    info!("Cache config {:?}", cache_config);

    let llm_engine = LLMEngine::new(
        pipelines,
        SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
        },
        &cache_config,
        &config,
        Arc::new(Notify::new()),
        args.holding_time,
        num_shards,
        args.multi_process,
        #[cfg(feature = "nccl")]
        daemon_manager,
    )?;

    let max_model_len = pipeline_config.max_model_len;
    let kvcached_tokens = cache_config.num_gpu_blocks.unwrap() * cache_config.block_size;
    let server_data = OpenAIServerData {
        pipeline_config,
        model: llm_engine,
        record_conversation: args.record_conversation,
        device: Device::Cpu,
    };

    if global_rank != 0 {
        info!("\nDaemon service started at rank {}.", global_rank);
    }

    #[cfg(feature = "nccl")]
    if args.multi_process {
        let e = server_data.model.read();
        let mut daemon_manager = e.daemon_manager.write();
        daemon_manager.as_mut().unwrap().mpi_sync();
    }

    if global_rank == 0 {
        info!("Maximum Model Length (affected by `--kvcache-mem-gpu` and the number of ranks):");
        for batch in [1, 8] {
            println!(
                "-> Batch {}: {}",
                batch,
                std::cmp::min(kvcached_tokens / batch, max_model_len)
            );
        }
        warn!("Server started at http://0.0.0.0:{}.", port);
    }

    let allow_origin = AllowOrigin::any();
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    let app = Router::new()
        .layer(cors_layer)
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(Arc::new(server_data));

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .map_err(|e| APIError::new(e.to_string()))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| APIError::new(e.to_string()))?;

    Ok(())
}
