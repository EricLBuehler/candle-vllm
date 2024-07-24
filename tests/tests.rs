use axum::{
    http::{self, Method},
    routing::post,
    Router,
};
use candle_core::{DType, Device};
use candle_vllm::{
    get_model_loader,
    openai::{
        openai_server::chat_completions, pipelines::llm_engine::LLMEngine, responses::APIError,
        OpenAIServerData,
    },
    scheduler::{cache_engine::CacheConfig, SchedulerConfig},
    ModelSelected,
};
use std::sync::Arc;
use tokio::sync::Notify;
use tower_http::cors::{AllowOrigin, CorsLayer};
#[tokio::main]
async fn test_llama() -> Result<(), APIError> {
    let (loader, model_id) = get_model_loader(
        ModelSelected::Llama {
            repeat_last_n: Some(64),
            penalty: Some(1.1),
            temperature: None,
            max_gen_tokens: Some(512),
        },
        Some("meta-llama/Llama-2-7b-chat-hf".to_string()),
    );
    let paths = loader.download_model(
        model_id,
        None,
        Some(std::env::var("TESTS_HF_TOKEN").unwrap()),
        None,
    )?;
    let model = loader.load_model(paths, DType::F16, Device::Cpu)?;
    let finish_notify = Arc::new(Notify::new());
    let llm_engine = LLMEngine::new(
        model.0,
        SchedulerConfig { max_num_seqs: 256 },
        CacheConfig {
            block_size: 16,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            fully_init: false,
            dtype: DType::F16,
        },
        Arc::new(Notify::new()),
        finish_notify.clone(),
    )?;

    let server_data = OpenAIServerData {
        pipeline_config: model.1,
        model: llm_engine,
        device: Device::Cpu,
        record_conversation: false,
        finish_notify: finish_notify.clone(),
    };

    let allow_origin = AllowOrigin::any();
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    let app = Router::new()
        .layer(cors_layer)
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(Arc::new(server_data));

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:2000"))
        .await
        .map_err(|e| APIError::new(e.to_string()))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| APIError::new(e.to_string()))?;

    Ok(())
}
