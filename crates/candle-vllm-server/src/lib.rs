use crate::config::validation::{validate_mcp, validate_models};
use crate::config::{McpConfig, ModelRegistryConfig};
use crate::models_config::{to_model_registry, ModelsState};
use crate::routes::build_router;
use crate::state::model_manager::ModelManager;
use candle_vllm_openai::model_registry::ModelRegistry;
pub mod config;
pub mod state;
use axum::{
    http::{self, Method},
};
use candle_core::{DType, Device, Result};
#[cfg(feature = "nccl")]
use candle_vllm_core::backend::heartbeat;
use candle_vllm_core::scheduler::cache_engine::{CacheConfig, CacheEngine};
use candle_vllm_core::scheduler::SchedulerConfig;
use candle_vllm_openai::model_registry::ModelAlias;
use candle_vllm_openai::pipelines::llm_engine::LLMEngine;
use candle_vllm_openai::pipelines::pipeline::DefaultLoader;
use candle_vllm_openai::sampling_params::GenerationConfig;
use candle_vllm_openai::OpenAIServerData;
use candle_vllm_responses::session::ResponsesSession;
use clap::Parser;
use futures::executor;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
mod models_config;
mod routes;
use tracing::{info, warn};
const SIZE_IN_MB: usize = 1024 * 1024;
use candle_vllm_openai::models::Config;
use rustchatui::start_ui_server;
use tokio::sync::Notify;
use tower_http::cors::{Any, CorsLayer};
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

    /// Host address to bind to, to serve on host:port
    #[arg(long = "h", default_value = "0.0.0.0")]
    host: String,

    /// Port to serve on (host:port)
    #[arg(long = "p", default_value_t = 2000)]
    port: u16,

    /// Set verbose mode (print all requests)
    #[arg(long)]
    verbose: bool,

    /// Maximum number of sequences to allow
    #[arg(long, default_value_t = 16)]
    max_num_seqs: usize,

    /// Size of a block
    #[arg(long, default_value_t = 64)]
    block_size: usize,

    /// if weight_path is passed, it will ignore the model_id
    #[arg(long = "m")]
    model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online), path must include last "/"
    #[arg(long = "w")]
    weight_path: Option<String>,

    /// The quantized weight file name (for gguf/ggml file)
    #[arg(long = "f")]
    weight_file: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    #[arg(long)]
    isq: Option<String>,

    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Available GPU memory for kvcache (MB)
    #[arg(long = "mem", default_value_t = 4096)]
    kvcache_mem_gpu: usize,

    /// Available CPU memory for kvcache (MB)
    #[arg(long, default_value_t = 128)]
    kvcache_mem_cpu: usize,

    /// Record conversation (default false, the client need to record chat history)
    #[arg(long)]
    record_conversation: bool,

    #[arg(long = "d", value_delimiter = ',')]
    device_ids: Option<Vec<usize>>,

    /// Maximum waiting time for processing parallel requests (in milliseconds).
    /// A larger value means the engine can hold more requests and process them in a single generation call.
    #[arg(long, default_value_t = 500)]
    holding_time: usize,

    //Whether the program is forced running in multithread model for parallel inference (for debug)
    #[arg(long, default_value_t = false)]
    multithread: bool,

    #[arg(long, default_value_t = false)]
    log: bool,

    #[arg(long)]
    temperature: Option<f32>,

    #[arg(long)]
    top_p: Option<f32>,

    #[arg(long)]
    min_p: Option<f32>,

    #[arg(long)]
    top_k: Option<isize>,

    #[arg(long)]
    frequency_penalty: Option<f32>,

    #[arg(long)]
    presence_penalty: Option<f32>,

    #[arg(long)]
    prefill_chunk_size: Option<usize>,

    #[arg(long, default_value_t = false)]
    fp8_kvcache: bool,

    #[arg(long, default_value_t = false)]
    ui_server: bool, //start candle-vllm with built-in web server

    /// Path to mcp.json configuration
    #[arg(long)]
    mcp_config: Option<String>,

    /// Maximum queue size per model (default: 10)
    #[arg(long, default_value_t = 10)]
    queue_size: usize,

    /// Request timeout in seconds (default: 30)
    #[arg(long, default_value_t = 30)]
    request_timeout: u64,
}

fn apply_model_alias(args: &mut Args, alias: &ModelAlias) {
    if let Some(id) = alias.model_id.clone() {
        args.model_id = Some(id);
    }
    if alias.weight_path.is_some() {
        if args.weight_path.is_some() && args.weight_path != alias.weight_path {
            warn!(
                "Overriding CLI weight_path with models.yaml entry for '{}'",
                alias.name
            );
        }
        args.weight_path = alias.weight_path.clone();
    }
    if alias.weight_file.is_some() {
        if args.weight_file.is_some() && args.weight_file != alias.weight_file {
            warn!(
                "Overriding CLI weight_file with models.yaml entry for '{}'",
                alias.name
            );
        }
        args.weight_file = alias.weight_file.clone();
    }
    if alias.dtype.is_some() {
        args.dtype = alias.dtype.clone();
    }
    if alias.block_size.is_some() {
        args.block_size = alias.block_size.unwrap();
    }
    if alias.max_num_seqs.is_some() {
        args.max_num_seqs = alias.max_num_seqs.unwrap();
    }
    if alias.kvcache_mem_gpu.is_some() {
        args.kvcache_mem_gpu = alias.kvcache_mem_gpu.unwrap();
    }
    if alias.kvcache_mem_cpu.is_some() {
        args.kvcache_mem_cpu = alias.kvcache_mem_cpu.unwrap();
    }
    if alias.prefill_chunk_size.is_some() {
        args.prefill_chunk_size = alias.prefill_chunk_size;
    }
    if alias.multithread.is_some() {
        args.multithread = alias.multithread.unwrap_or(false);
    }
    if alias.device_ids.is_some() {
        args.device_ids = alias.device_ids.clone();
    }
    if alias.temperature.is_some() {
        args.temperature = alias.temperature;
    }
    if alias.top_p.is_some() {
        args.top_p = alias.top_p;
    }
    if alias.top_k.is_some() {
        args.top_k = alias.top_k;
    }
    if alias.frequency_penalty.is_some() {
        args.frequency_penalty = alias.frequency_penalty;
    }
    if alias.presence_penalty.is_some() {
        args.presence_penalty = alias.presence_penalty;
    }
    if alias.isq.is_some() {
        args.isq = alias.isq.clone();
    }
}

fn load_model_registry() -> (
    Option<ModelRegistry>,
    HashMap<String, String>,
    Option<Duration>,
) {
    let mut validation = HashMap::new();
    let mut idle_unload = None;
    // Load models config: env var > default locations
    let models_config_path = std::env::var("CANDLE_VLLM_MODELS_CONFIG")
        .ok()
        .filter(|p| std::path::Path::new(p).exists());
    
    let candidates: Vec<&str> = if let Some(ref path) = models_config_path {
        vec![path.as_str()]
    } else {
        vec!["models.yaml", "models.yml"]
    };
    
    for candidate in candidates {
        if std::path::Path::new(candidate).exists() {
            match ModelRegistryConfig::load(candidate) {
                Ok(cfg) => {
                    idle_unload = cfg.idle_unload_secs.map(Duration::from_secs);
                    match validate_models(&cfg) {
                        Ok(_) => {
                            for m in &cfg.models {
                                validation.insert(m.name.clone(), "valid".to_string());
                            }
                        }
                        Err(errs) => {
                            let msg = format!("invalid: {}", errs.join("; "));
                            for m in &cfg.models {
                                validation.insert(m.name.clone(), msg.clone());
                            }
                        }
                    }
                    let registry = to_model_registry(&cfg);
                    return (Some(registry), validation, idle_unload);
                }
                Err(err) => {
                    warn!("Failed to load models registry {candidate}: {err}");
                }
            }
        }
    }
    (None, validation, idle_unload)
}

fn get_cache_config(
    kvcache_mem_gpu: usize,
    kvcache_mem_cpu: usize,
    block_size: usize,
    config: &Config,
    kv_dtype: DType,
    num_shards: usize,
) -> CacheConfig {
    let dsize = kv_dtype.size_in_bytes();
    let num_gpu_blocks = kvcache_mem_gpu * SIZE_IN_MB
        / dsize
        / block_size
        / (config.num_key_value_heads.unwrap() / num_shards)
        / config.k_head_dim()
        / config.num_hidden_layers
        / 2;
    let num_cpu_blocks = kvcache_mem_cpu * SIZE_IN_MB
        / dsize
        / block_size
        / (config.num_key_value_heads.unwrap() / num_shards)
        / config.k_head_dim()
        / config.num_hidden_layers
        / 2;
    CacheConfig {
        block_size,
        num_gpu_blocks: Some(num_gpu_blocks),
        num_cpu_blocks: Some(num_cpu_blocks),
        fully_init: true,
        dtype: kv_dtype,
        kvcache_mem_gpu,
    }
}

fn config_log(logger: ftail::Ftail, log_enable: bool, log_file: String) -> Result<()> {
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
        for (i, name) in log_level_names.iter().copied().enumerate() {
            if level.contains(name) {
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
        .map_err(candle_core::Error::wrap)
}

fn get_dtype(dtype: Option<String>) -> DType {
    let dtype = match dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => panic!("Unsupported dtype {dtype}"),
        None => DType::BF16,
    };

    #[cfg(feature = "cuda")]
    let dtype = {
        use candle_core::cuda_backend::cudarc::driver::result::{device, init};
        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
        match (init(), device::get(0)) {
            (Ok(_), Ok(d)) => {
                let (compute_major, compute_minor) = unsafe {
                    (
                        device::get_attribute(
                            d,
                            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        )
                        .unwrap_or(8),
                        device::get_attribute(
                            d,
                            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        )
                        .unwrap_or(8),
                    )
                };
                info!(
                    "CUDA compute capability: {}.{}",
                    compute_major, compute_minor,
                );
                if dtype != DType::F32 && compute_major < 8 {
                    warn!(
                        "CUDA compute capability: {} (<8), switched to F16 cause no BF16 support.",
                        compute_major
                    );
                    DType::F16
                } else {
                    dtype
                }
            }
            _ => dtype,
        }
    };
    dtype
}

#[allow(unused_mut)]
pub async fn run() -> Result<()> {
    let mut args = Args::parse();
    if !args.log {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    // Apply model alias before using args fields
    let (registry, validation, idle_unload) = load_model_registry();
    if let (Some(registry_ref), Some(name)) = (registry.as_ref(), args.model_id.clone()) {
        if let Some(alias) = registry_ref.find(&name) {
            info!("Using model alias '{}' from models.yaml", name);
            apply_model_alias(&mut args, &alias);
        }
    }

    let loader = Box::new(DefaultLoader::new(
        args.model_id.clone(),
        args.weight_path.clone(),
        args.weight_file.clone(),
    ));

    let (paths, gguf) =
        loader.prepare_model_weights(args.hf_token.clone(), args.hf_token_path.clone())?;

    let dtype = get_dtype(args.dtype.clone());
    let kv_cache_dtype = if args.fp8_kvcache { DType::U8 } else { dtype };

    if cfg!(feature = "flash-decoding") {
        assert!(
            !args.fp8_kvcache,
            "fp8 kvcache is not compatible with `flash-decoding` feature!"
        );
    }

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

    if gguf && num_shards > 1 {
        panic!("Multiple device-ids detected: ggml/gguf model is not supported for multi-rank inference! \n\t*** Tips: use unquantized safetensors models (`--w`) with ISQ (e.g., `--isq q4k`) for multi-gpu inference!");
    }

    if gguf && args.isq.is_some() {
        panic!("Quantized gguf/ggml model does not support isq option!");
    }

    assert!(
        args.prefill_chunk_size.is_none() || args.prefill_chunk_size.unwrap() % 1024 == 0,
        "Error: prefill_chunk_size must be divisible by 1024!"
    );

    let multi_process = if num_shards > 1 {
        if args.multithread {
            tracing::warn!("The program is forced running under multithread mode (for debug purpose), which may not stable!");
            false
        } else {
            tracing::warn!("Multi-process mode is automatically enabled for multi-rank inference!");
            true
        }
    } else {
        !args.multithread
    };

    #[cfg(all(feature = "cuda", feature = "graph"))]
    {
        assert!(
            multi_process,
            "Graph capture is only available under multi process mode!"
        );
        if args.max_num_seqs > 16 {
            tracing::warn!("Higher GPU memory required for capturing large batch!");
        }
    }

    let logger: ftail::Ftail = ftail::Ftail::new();
    let host = args.host.clone();
    let mut port = args.port;
    let ui_server = args.ui_server;
    let ui_port = args.port;
    #[cfg(feature = "nccl")]
    let (pipelines, global_rank, daemon_manager) = if multi_process {
        use candle_vllm_openai::communicator::init_subprocess;
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
                    kv_cache_dtype,
                    gguf,
                    args.isq.clone(),
                    args.block_size,
                    args.max_num_seqs,
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
        use candle_vllm_openai::communicator::DaemonManager;
        DaemonManager::set_master_rank(true); //master rank default for multithreaded mode
        let log_file = format!("candle-vllm-{}ranks.log", device_ids.len());
        let _ = config_log(logger, args.log, log_file);
        (
            loader
                .load_model(
                    paths,
                    dtype,
                    kv_cache_dtype,
                    gguf,
                    args.isq.clone(),
                    args.block_size,
                    args.max_num_seqs,
                    device_ids,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await,
            0,
            None,
        )
    };

    #[cfg(feature = "nccl")]
    info!(
        "parallel model: {}!",
        if multi_process {
            "multiprocess"
        } else {
            "multithread"
        }
    );

    #[cfg(not(feature = "nccl"))]
    let (pipelines, global_rank) = {
        let log_file = "candle-vllm.log".to_string();
        let _ = config_log(logger, args.log, log_file);
        (
            loader
                .load_model(
                    paths,
                    dtype,
                    kv_cache_dtype,
                    gguf,
                    args.isq.clone(),
                    args.block_size,
                    args.max_num_seqs,
                    device_ids,
                    None,
                    None,
                )
                .await,
            0,
        )
    };

    let (default_pipelines, mut pipeline_config) = match pipelines {
        Err(e) => panic!("{e:?}"),
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
                kv_cache_dtype,
                num_shards,
            );
            let cache_engine = CacheEngine::new(
                &cfg,
                &cache_cfg,
                cache_cfg.dtype,
                pipeline.device(),
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
        multi_process,
        #[cfg(feature = "nccl")]
        daemon_manager,
        args.prefill_chunk_size,
    )?;

    if args.temperature.is_some() || pipeline_config.generation_cfg.is_none() {
        //overwrite the generation config when temperature (and others) specified in arguments
        //disable multinomial sampling (generation randomness) by setting `temperature` as 0
        pipeline_config.generation_cfg = Some(GenerationConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
            frequency_penalty: args.frequency_penalty,
            presence_penalty: args.presence_penalty,
        })
    } else {
        pipeline_config
            .generation_cfg
            .as_mut()
            .unwrap()
            .frequency_penalty = args.frequency_penalty;
        pipeline_config
            .generation_cfg
            .as_mut()
            .unwrap()
            .presence_penalty = args.presence_penalty;
        pipeline_config.generation_cfg.as_mut().unwrap().min_p = args.min_p;
    }

    info!("Pipeline config {:?}", pipeline_config);

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
    if multi_process {
        let e = server_data.model.read();
        let mut daemon_manager = e.daemon_manager.write();
        daemon_manager.as_mut().unwrap().mpi_sync();
    }

    #[cfg(all(feature = "cuda", feature = "graph"))]
    LLMEngine::graph_capture(&server_data.model).unwrap();

    if global_rank == 0 {
        warn!(
            "Maximum Model Length (affected by `--mem` (kvcache-mem-gpu) and the number of ranks):"
        );
        for batch in [1, 8] {
            println!(
                "-> Batch {}: {}",
                batch,
                std::cmp::min(kvcached_tokens / batch, max_model_len)
            );
        }
        warn!("Server started at http://{host}:{port}/v1/");
    }

    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(Any) // same as "*"
        .allow_methods(Any)
        .allow_headers(Any);

    let models = ModelsState::new(registry, validation, idle_unload);

    // Load MCP config: CLI arg > env var > default location
    let mcp_config_path = args.mcp_config.clone()
        .or_else(|| std::env::var("CANDLE_VLLM_MCP_CONFIG").ok())
        .or_else(|| {
            if std::path::Path::new("mcp.json").exists() {
                Some("mcp.json".to_string())
            } else {
                None
            }
        });
    let mcp_session = if let Some(path) = mcp_config_path {
        info!("Loading MCP config from: {}", path);
        match McpConfig::load(&path).and_then(|cfg| match validate_mcp(&cfg) {
            Ok(_) => Ok(cfg),
            Err(errs) => Err(anyhow::anyhow!(errs.join(", "))),
        }) {
            Ok(cfg) => {
                info!("MCP config loaded successfully with {} server(s)", cfg.servers.len());
                match serde_json::to_value(&cfg)
                    .map_err(anyhow::Error::from)
                    .and_then(|v| executor::block_on(ResponsesSession::from_config_value(&v)))
                {
                    Ok(session) => {
                        info!("MCP session initialized successfully");
                        Some(Arc::new(session))
                    }
                    Err(err) => {
                        warn!("Failed to initialize MCP session from {path}: {err}");
                        None
                    }
                }
            }
            Err(err) => {
                warn!("Failed to load MCP config {path}: {err}");
                None
            }
        }
    } else {
        info!("No MCP config found (checked CLI arg, CANDLE_VLLM_MCP_CONFIG env var, and mcp.json)");
        None
    };

    // Create model manager with queue configuration
    let model_manager = Arc::new(ModelManager::with_queue_config(
        models.clone(),
        10, // max switch queue
        args.queue_size,
        Duration::from_secs(args.request_timeout),
    ));

    let app_state = routes::AppState {
        models,
        data: Arc::new(server_data),
        mcp: mcp_session,
        model_manager: Some(model_manager),
    };

    let app = build_router(app_state).layer(cors_layer);

    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}"))
        .await
        .map_err(candle_core::Error::wrap)?;

    let mut tasks = Vec::new();
    tasks.push(tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("Chat API server error: {e:?}");
        }
    }));

    // Usage example: https://github.com/guoqingbao/rustchatui/blob/main/ReadMe.md
    if ui_server && global_rank == 0 {
        tasks.push(tokio::spawn(async move {
            start_ui_server((ui_port - 1) as u16, Some(ui_port as u16), None, None)
                .await
                .unwrap();
        }));
    }

    futures::future::try_join_all(tasks)
        .await
        .map_err(candle_core::Error::wrap)?;

    Ok(())
}
