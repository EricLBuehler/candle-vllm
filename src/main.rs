use axum::{
    extract::State,
    http::{self, Method},
    routing::{get, post},
    Json, Router,
};
use candle_core::{DType, Device, Result};
#[cfg(feature = "nccl")]
use candle_vllm::backend::heartbeat;
use candle_vllm::openai::models::Config;
use candle_vllm::openai::openai_server::{chat_completions, create_embeddings};
use candle_vllm::openai::pipelines::llm_engine::LLMEngine;
use candle_vllm::openai::pipelines::pipeline::DefaultLoader;
use candle_vllm::openai::sampling_params::GenerationConfig;
use candle_vllm::openai::utils::{
    bind_addr_for_rank, bind_api_listener, ensure_server_bindings_or_exit,
    resolve_server_bind_addr, tcp_api_url, ApiListener, ServerBindAddr,
};
use candle_vllm::openai::{kv_cache_capacity_tokens, OpenAIServerData};
use candle_vllm::scheduler::cache_engine::{CacheConfig, CacheEngine};
use candle_vllm::scheduler::prefix_cache::PrefixCacheConfig;
use candle_vllm::scheduler::SchedulerConfig;
use clap::Parser;
use colored::*;
use local_ip_address::local_ip;
use rustchatui::start_ui_server;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Notify;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
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

    /// Bind address. Supports host, host:port, [ipv6]:port, tcp://host[:port], and unix:///path.
    #[arg(long = "h", default_value = "0.0.0.0")]
    host: String,

    /// TCP port to serve on when --h does not include a port.
    #[arg(long = "p", default_value_t = 2000)]
    port: u16,

    /// Set verbose mode (print all requests)
    #[arg(long)]
    verbose: bool,

    /// Maximum number of sequences to allow
    #[arg(long, default_value_t = 8)]
    max_num_seqs: usize,

    /// Size of a block
    #[arg(long)]
    block_size: Option<usize>,

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

    /// Fixed GPU memory budget for kvcache (MB). Used when auto sizing is unavailable.
    #[arg(long = "mem", default_value_t = 4096)]
    kvcache_mem_gpu: usize,

    /// After model loading, the fraction of remaining GPU memory available for KV cache.
    /// Takes priority over `--mem` on CUDA/Metal.
    #[arg(long)]
    kv_fraction: Option<f32>,

    /// Fraction of the auto-sized combined cache budget reserved for hybrid Mamba/GDN states.
    #[arg(long)]
    mamba_fraction: Option<f32>,

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
    #[arg(long, default_value_t = 100)]
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

    /// KV cache dtype: auto (default), fp8, turbo8, turbo4, turbo3
    #[arg(long)]
    kvcache_dtype: Option<String>,

    /// Disable prefix cache (enabled by default).
    #[arg(long, default_value_t = false)]
    disable_prefix_cache: bool,

    /// Prefix cache size limit in tokens (rounded down to block size).
    #[arg(long)]
    prefix_cache_max_tokens: Option<usize>,

    /// Disable CUDA graph capture (enabled by default on CUDA builds).
    #[arg(long, default_value_t = false)]
    disable_cuda_graph: bool,

    #[arg(long, default_value_t = false)]
    ui_server: bool, //start candle-vllm with built-in web server

    /// MCP server command (single server mode)
    #[arg(long)]
    mcp_command: Option<String>,

    /// MCP server arguments (comma-separated)
    #[arg(long, value_delimiter = ',')]
    mcp_args: Option<Vec<String>>,

    /// Path to MCP config file (multi-server mode)
    #[arg(long)]
    mcp_config: Option<String>,

    /// Force a specific tool parser backend (for example: qwen, qwen_coder, json, mistral).
    #[arg(long)]
    enforce_parser: Option<String>,

    /// YARN RoPE scaling factor (explicit override, no auto-calculation)
    #[arg(long)]
    yarn_scaling_factor: Option<f64>,

    /// Number of nodes for multi-node tensor parallel inference (default: 1 = single node)
    #[arg(long, default_value_t = 1)]
    num_nodes: usize,

    /// Rank of this node (0 = master, 1..num_nodes-1 = workers)
    #[arg(long, default_value_t = 0)]
    node_rank: usize,

    /// Master node address for multi-node coordination (e.g. "192.168.1.100")
    #[arg(long, default_value = "")]
    master_addr: String,

    /// Master node port for multi-node NCCL ID exchange
    #[arg(long, default_value_t = 29500)]
    master_port: u16,
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

#[tokio::main]
#[allow(unused_mut)]
async fn main() -> Result<()> {
    let args = Args::parse();
    if !args.log {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    let loader = Box::new(DefaultLoader::new(
        args.model_id,
        args.weight_path,
        args.weight_file,
        args.enforce_parser.clone(),
        args.yarn_scaling_factor,
    ));

    let (paths, gguf) = loader.prepare_model_weights(args.hf_token, args.hf_token_path)?;

    let dtype = candle_vllm::get_dtype(args.dtype);
    let kvcache_dtype_enum = if let Some(ref s) = args.kvcache_dtype {
        candle_vllm::openai::models::KvCacheDtype::from_str_opt(s).unwrap_or_else(|| {
            panic!(
                "Invalid --kvcache-dtype value: {}. Use auto/fp8/turbo8/turbo4/turbo3.",
                s
            )
        })
    } else {
        candle_vllm::openai::models::KvCacheDtype::Auto
    };
    let kv_cache_dtype = if kvcache_dtype_enum.is_fp8_keys() {
        DType::U8
    } else {
        dtype
    };
    candle_vllm::openai::models::KvCacheDtype::set_global(kvcache_dtype_enum);

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

    if gguf && args.isq.is_some() {
        panic!("Quantized gguf/ggml model does not support isq option!");
    }

    assert!(
        args.prefill_chunk_size.is_none() || args.prefill_chunk_size.unwrap() % 1024 == 0,
        "Error: prefill_chunk_size must be divisible by 1024!"
    );

    // Multi-node validation
    let is_multi_node = args.num_nodes > 1;
    let master_addr = if is_multi_node && args.node_rank == 0 && args.master_addr.is_empty() {
        "0.0.0.0".to_string()
    } else {
        args.master_addr.clone()
    };
    if is_multi_node {
        assert!(
            args.node_rank < args.num_nodes,
            "--node-rank ({}) must be less than --num-nodes ({})",
            args.node_rank,
            args.num_nodes
        );
        assert!(
            args.node_rank == 0 || (!master_addr.is_empty() && master_addr != "0.0.0.0"),
            "--master-addr must be the reachable master node address on worker nodes"
        );
        assert!(
            args.master_port < u16::MAX,
            "--master-port must be less than 65535 for multi-node coordination"
        );
        tracing::info!(
            "Multi-node mode: {} nodes, this is node {} ({}), master at {}:{}",
            args.num_nodes,
            args.node_rank,
            if args.node_rank == 0 {
                "master"
            } else {
                "worker"
            },
            master_addr,
            args.master_port,
        );
        num_shards = local_world_size * args.num_nodes;
    }

    #[cfg(feature = "nccl")]
    let multi_node_config = if is_multi_node {
        Some(candle_vllm::openai::communicator::MultiNodeConfig {
            num_nodes: args.num_nodes,
            node_rank: args.node_rank,
            master_addr: master_addr.clone(),
            master_port: args.master_port,
            local_num_gpus: local_world_size,
        })
    } else {
        None
    };

    let multi_process = if num_shards > 1 || is_multi_node {
        if args.multithread && !is_multi_node {
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
            multi_process || (!multi_process && num_shards == 1),
            "Graph capture is only available under multiprocess mode or single-rank multithread mode!"
        );
        if args.max_num_seqs > 16 {
            tracing::warn!("Higher GPU memory required for capturing large batch!");
        }
    }

    let block_size = args
        .block_size
        .unwrap_or(if cfg!(feature = "cuda") { 64 } else { 32 });
    let logger: ftail::Ftail = ftail::Ftail::new();
    let host = args.host;
    let base_bind_addr = resolve_server_bind_addr(&host, args.port)?;
    ensure_server_bindings_or_exit(&base_bind_addr, args.ui_server)?;

    #[cfg(feature = "nccl")]
    let (pipelines, global_rank, daemon_manager) = if multi_process {
        use candle_vllm::openai::communicator::init_subprocess;
        let (id, local_rank, global_rank, global_world_size, daemon_manager) =
            init_subprocess(device_ids.clone(), multi_node_config.as_ref()).unwrap();
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
                    block_size,
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
        use candle_vllm::openai::communicator::DaemonManager;
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
                    block_size,
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
                    block_size,
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
    let devices: Vec<_> = default_pipelines
        .iter()
        .map(|pipeline| pipeline.device())
        .collect();
    let first_pipeline = default_pipelines
        .first()
        .expect("at least one pipeline must be loaded");
    let first_config = first_pipeline.get_model_config();
    let first_model_dtype = first_pipeline.dtype;
    let kv_fraction = args.kv_fraction.unwrap_or(0.6);
    let explicit_kv_fraction = args.kv_fraction.is_some();
    let prefill_chunk_size = args.prefill_chunk_size.unwrap_or(8192);

    let workspace_params = candle_vllm::WorkspaceBudgetParams::from_config(
        &first_config,
        first_model_dtype,
        num_shards,
        prefill_chunk_size,
    );
    let workspace_budget = candle_vllm::compute_workspace_budget(&workspace_params);
    info!(
        "Workspace memory reserve: {:.2} GB (flashinfer {:.0} MB, cutlass {:.0} MB, \
         moe_pool {:.0} MB, flash_splitk {:.0} MB, transient {:.0} MB)",
        workspace_budget.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        workspace_budget.flashinfer_bytes as f64 / 1024.0 / 1024.0,
        workspace_budget.cutlass_bytes as f64 / 1024.0 / 1024.0,
        workspace_budget.moe_pool_bytes as f64 / 1024.0 / 1024.0,
        workspace_budget.flash_splitk_bytes as f64 / 1024.0 / 1024.0,
        workspace_budget.transient_bytes as f64 / 1024.0 / 1024.0,
    );

    let (kvcache_mem_gpu, mamba_cache_budget_bytes, kvcache_budget_desc) =
        match candle_vllm::detect_kvcache_mem_gpu_mb_for_devices_with_workspace(
            &devices,
            kv_fraction,
            Some(&workspace_budget),
        ) {
            Ok(detected) => {
                let mut effective_kvcache_mem_gpu = detected;
                let mut mamba_cache_budget_bytes = 0usize;
                if let Some(estimate) = candle_vllm::estimate_hybrid_mamba_cache(
                    &first_config,
                    first_model_dtype,
                    num_shards,
                ) {
                    if let Some(plan) = candle_vllm::plan_hybrid_mamba_cache_with_fraction(
                        detected * 1024 * 1024,
                        estimate,
                        args.max_num_seqs,
                        !args.disable_prefix_cache,
                        args.mamba_fraction,
                    ) {
                        let reserved_mamba_mb = plan.budget_bytes.div_ceil(1024 * 1024);
                        if reserved_mamba_mb < detected {
                            effective_kvcache_mem_gpu = detected - reserved_mamba_mb;
                            mamba_cache_budget_bytes = plan.budget_bytes;
                            info!(
                            "Reserved {:.2} GB of the combined GPU cache budget for hybrid mamba/GDN ({} active slot target, {} prefix slot target); KV cache budget is now {:.2} GB",
                            plan.budget_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                            plan.active_slot_capacity,
                            plan.prefix_slot_capacity,
                            effective_kvcache_mem_gpu as f64 / 1024.0,
                        );
                        } else {
                            warn!(
                            "Hybrid mamba reservation {:.2} GB would consume the entire combined cache budget {:.2} GB; skipping upfront reservation.",
                            plan.budget_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                            detected as f64 / 1024.0,
                        );
                        }
                    }
                }
                info!(
                    "Using auto-detected KV cache budget of {} MB per rank (kv_fraction={} of post-load free GPU memory)",
                    effective_kvcache_mem_gpu, kv_fraction
                );
                (
                    effective_kvcache_mem_gpu,
                    mamba_cache_budget_bytes,
                    format!(
                        "--kv-fraction {} -> {} MB per rank",
                        kv_fraction, effective_kvcache_mem_gpu
                    ),
                )
            }
            Err(err) if !explicit_kv_fraction => {
                warn!(
                    "Auto KV cache sizing unavailable ({}), falling back to fixed --mem {} MB",
                    err, args.kvcache_mem_gpu
                );
                (
                    args.kvcache_mem_gpu,
                    0,
                    format!("--mem {} MB", args.kvcache_mem_gpu),
                )
            }
            Err(err) => return Err(err),
        };

    let pipelines = default_pipelines
        .into_iter()
        .map(|pipeline| {
            let cfg = pipeline.get_model_config();
            let mut cache_cfg = candle_vllm::get_cache_config(
                kvcache_mem_gpu,
                args.kvcache_mem_cpu,
                block_size,
                &cfg,
                kv_cache_dtype,
                num_shards,
                kvcache_dtype_enum,
            );
            cache_cfg.mamba_cache_budget_bytes = mamba_cache_budget_bytes;
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

    let total_gpu_blocks = cache_config.num_gpu_blocks.unwrap_or(0);
    let default_prefix_cache_blocks = if total_gpu_blocks > 0 {
        std::cmp::max(1, total_gpu_blocks / 2)
    } else {
        0
    };
    let prefix_cache_max_blocks = if !args.disable_prefix_cache {
        let max_blocks = args
            .prefix_cache_max_tokens
            .map(|tokens| tokens / cache_config.block_size)
            .unwrap_or(default_prefix_cache_blocks);
        std::cmp::min(max_blocks, total_gpu_blocks)
    } else {
        0
    };
    let prefix_cache_config = PrefixCacheConfig {
        enabled: !args.disable_prefix_cache,
        max_cached_blocks: prefix_cache_max_blocks,
    };

    let llm_engine = LLMEngine::new(
        pipelines,
        SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
            prefix_cache: prefix_cache_config,
            mamba_cache_capacity: None,
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
        args.disable_cuda_graph,
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

    pipeline_config.apply_kv_cache_limit(&cache_config);

    info!("Pipeline config {:?}", pipeline_config);

    let max_model_len = pipeline_config.max_model_len;
    let kvcached_tokens = kv_cache_capacity_tokens(&cache_config);

    let mcp_manager_config = if let Some(path) = &args.mcp_config {
        match candle_vllm::mcp::McpManagerConfig::from_file(path) {
            Ok(cfg) => Some(cfg),
            Err(err) => {
                tracing::error!("Failed to load MCP config file: {:?}", err);
                None
            }
        }
    } else if let Some(command) = args.mcp_command.clone() {
        Some(candle_vllm::mcp::McpManagerConfig::from_single(
            candle_vllm::mcp::manager::McpToolConfig::new(
                command,
                args.mcp_args.clone().unwrap_or_default(),
            ),
        ))
    } else {
        None
    };

    // Initialize MCP Manager
    let mcp_manager = if let Some(cfg) = mcp_manager_config {
        match candle_vllm::mcp::McpClientManager::new(cfg) {
            Ok(manager) => Some(Arc::new(manager)),
            Err(err) => {
                tracing::error!("Failed to start MCP client manager: {:?}", err);
                None
            }
        }
    } else {
        None
    };

    let server_data = OpenAIServerData {
        pipeline_config,
        model: llm_engine,
        record_conversation: args.record_conversation,
        device: Device::Cpu,
        mcp_manager: mcp_manager.clone(),
    };

    if let Some(manager) = &mcp_manager {
        info!("Waiting for MCP tools to be available...");
        if manager.wait_for_available(std::time::Duration::from_secs(30)) {
            info!("MCP tools available.");
        } else {
            warn!("MCP tools wait timed out.");
        }
    }

    if global_rank != 0 {
        info!("\nDaemon service started at rank {}.", global_rank);
    }

    #[cfg(feature = "nccl")]
    if multi_process {
        let e = server_data.model.read();
        let mut daemon_manager = e.daemon_manager.write();
        daemon_manager.as_mut().unwrap().mpi_sync();
    }

    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(Any) // same as "*"
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route(
            "/v1/models",
            get(|State(data): State<Arc<OpenAIServerData>>| async move {
                let (model_name, modalities) = {
                    let engine = data.model.read();
                    let (pipeline, _) = engine.get_pipeline(0).unwrap();
                    let modalities = if pipeline.image_config.is_some() {
                        vec!["text", "image"]
                    } else {
                        vec!["text", "embedding"]
                    };
                    (pipeline.name().to_string(), modalities)
                };
                Json(json!({
                    "object": "list",
                    "data": [
                        {
                            "id": model_name,
                            "object": "model",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as i64,
                            "owned_by": "candle-vllm",
                            "permission": [],
                            "modalities": modalities,
                            "max_model_len": data.pipeline_config.max_model_len,
                        }
                    ]
                }))
            }),
        )
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(create_embeddings))
        .layer(cors_layer)
        .with_state(Arc::new(server_data));

    let bind_addr = bind_addr_for_rank(&base_bind_addr, global_rank);
    let listener = bind_api_listener(&bind_addr).await?;

    if global_rank == 0 {
        warn!(
            "Maximum Model Length (affected by {} and the number of ranks):",
            kvcache_budget_desc
        );
        println!("-> Total KV cache tokens: {}", kvcached_tokens);
        for batch in [1, 2, 3, 4] {
            println!(
                "-> Batch {}: {}",
                batch,
                std::cmp::min(kvcached_tokens / batch, max_model_len)
            );
        }
        match bind_addr.clone() {
            ServerBindAddr::Tcp(sock_addr) => {
                let display_url = tcp_api_url(sock_addr);
                if sock_addr.ip().is_unspecified() {
                    let ip = local_ip().unwrap_or("127.0.0.1".parse().unwrap());
                    let local_url = format!("http://localhost:{}/v1/", sock_addr.port());
                    let lan_url = format!("http://{ip}:{}/v1/", sock_addr.port());

                    println!(
                        "\n🧠 API server running at:\n\t{} (Local Access) \n\t{} (Remote Access)\n",
                        local_url.cyan().bold(),
                        lan_url.cyan().bold(),
                    );
                } else {
                    println!(
                        "\n🧠 API server running at:\n\t{} (Bind Address)\n",
                        display_url.cyan().bold(),
                    );
                }
            }
            ServerBindAddr::Unix(path) => {
                println!(
                    "\n🧠 API server running at:\n\t{} (Unix Socket)\n",
                    format!("http+unix://{}", path.display()).cyan().bold(),
                );
            }
        }

        println!("");
        println!(
            "🛑 {}",
            format!("EXIT: Ctrl+C to quit. If unresponsive: Ctrl+P → Ctrl+Q (last resort).")
                .bold()
                .red()
        );
    }

    let mut tasks = Vec::new();
    tasks.push(tokio::spawn(async move {
        match listener {
            ApiListener::Tcp(listener) => axum::serve(listener, app).await.map_err(|e| {
                candle_core::Error::msg(format!("Chat API server error on TCP listener: {e}"))
            }),
            ApiListener::Unix(listener) => axum::serve(listener, app).await.map_err(|e| {
                candle_core::Error::msg(format!("Chat API server error on Unix listener: {e}"))
            }),
        }
    }));

    // Usage example: https://github.com/guoqingbao/rustchatui/blob/main/ReadMe.md
    if args.ui_server && global_rank == 0 {
        let ServerBindAddr::Tcp(sock_addr) = bind_addr else {
            candle_core::bail!("--ui-server is not supported with Unix sockets.");
        };
        let ui_port = sock_addr.port().checked_sub(1).ok_or_else(|| {
            candle_core::Error::msg(
                "Cannot start UI server because API port 0 has no preceding UI port.",
            )
        })?;
        let (api_port, api_url) = if sock_addr.ip().is_unspecified() {
            (Some(sock_addr.port()), None)
        } else {
            (None, Some(tcp_api_url(sock_addr)))
        };
        tasks.push(tokio::spawn(async move {
            match api_url {
                Some(api_url) => start_ui_server(ui_port, None, Some(api_url), None).await,
                None => start_ui_server(ui_port, api_port, None, None).await,
            }
            .map_err(candle_core::Error::wrap)
        }));
    }

    for task in tasks {
        task.await.map_err(candle_core::Error::wrap)??;
    }

    Ok(())
}
