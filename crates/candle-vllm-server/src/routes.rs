use crate::models_config::ModelsState;
use crate::state::request_queue::{QueuedRequest, QueueError};
use axum::{
    extract::State,
    http::HeaderMap,
    routing::{get, post},
    Json, Router,
};
use candle_vllm_core::openai::requests::ChatCompletionRequest;
use candle_vllm_core::openai::responses::{ChatResponder, APIError};
use candle_vllm_core::openai::utils::get_created_time_secs;
use candle_vllm_core::openai::OpenAIServerData;
use candle_vllm_responses::session::ResponsesSession;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tracing::{info, error};

// Real MCP session using candle-vllm-responses

#[derive(Clone)]
pub struct AppState {
    pub models: ModelsState,
    pub data: Arc<OpenAIServerData>,
    pub mcp: Option<Arc<ResponsesSession>>,
    pub model_manager: Option<Arc<crate::state::model_manager::ModelManager>>,
}

/// Thread-pool based chat completions handler that avoids Send trait violations
/// by keeping model access within worker threads
async fn chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
) -> ChatResponder {
    // Extract conversation_id and resource_id from headers if not in request body
    if req.conversation_id.is_none() {
        if let Some(conv_id) = headers.get("x-conversation-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
        {
            req.conversation_id = Some(conv_id);
        }
    }
    if req.resource_id.is_none() {
        if let Some(res_id) = headers.get("x-resource-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
        {
            req.resource_id = Some(res_id);
        }
    }

    // Handle model switching if needed
    let model_name = &req.model;
    if let Some(alias) = state.models.resolve(model_name) {
        if let Some(ref manager) = state.model_manager {
            match manager.enqueue_switch(&alias.name) {
                Ok(crate::state::model_manager::SwitchResult::Enqueued) => {
                    // Model switch enqueued - queue this request
                    let queue = manager.get_or_create_queue(&alias.name);
                    let (response_tx, response_rx) = oneshot::channel();

                    let queued_request = QueuedRequest::new(
                        alias.name.clone(),
                        req,
                        Some(response_tx),
                    );

                    match queue.enqueue(queued_request) {
                        Ok(()) => {
                            info!("Request queued for model switch to {}", alias.name);
                            // Wait for the response from the queue processor
                            match tokio::time::timeout(
                                Duration::from_secs(60), // 60 second timeout
                                response_rx
                            ).await {
                                Ok(Ok(responder)) => return responder,
                                Ok(Err(_)) => return ChatResponder::ModelError(APIError::new("Queue processing failed".to_string())),
                                Err(_) => return ChatResponder::ModelError(APIError::new("Request timed out in queue".to_string())),
                            }
                        }
                        Err(QueueError::QueueFull) => {
                            return ChatResponder::ValidationError(APIError::new(
                                "Request queue is full. Please try again later.".to_string()
                            ));
                        }
                        Err(e) => {
                            return ChatResponder::ModelError(APIError::new(format!(
                                "Failed to queue request: {}", e
                            )));
                        }
                    }
                }
                Ok(crate::state::model_manager::SwitchResult::AlreadyActive) => {
                    // Model is already active, proceed with request
                }
                Err(e) => {
                    return ChatResponder::ValidationError(APIError::new(format!(
                        "Model switching failed: {}", e
                    )));
                }
            }
        }
    } else {
        return ChatResponder::ValidationError(APIError::new(format!(
            "Model '{}' not found in registry", model_name
        )));
    }

    // Execute the request in a dedicated thread to avoid Send trait violations
    let data = state.data.clone();
    let result = tokio::task::spawn_blocking(move || {
        // This runs in a dedicated thread where parking_lot guards don't cross async boundaries
        tokio::runtime::Handle::current().block_on(async move {
            process_chat_completion_in_thread(data, req).await
        })
    }).await;

    match result {
        Ok(responder) => responder,
        Err(e) => {
            error!("Thread pool task failed: {}", e);
            ChatResponder::ModelError(APIError::new(format!(
                "Internal processing error: {}", e
            )))
        }
    }
}

/// Process chat completion within a dedicated thread context
/// This avoids Send trait violations by keeping all model access within the same thread
async fn process_chat_completion_in_thread(
    data: Arc<OpenAIServerData>,
    request: ChatCompletionRequest,
) -> ChatResponder {
    // Import the core function that has the Send trait issues
    // We're calling it from within a dedicated thread context now
    use candle_vllm_core::openai::openai_server::chat_completions_with_data;

    chat_completions_with_data(data, request).await
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/mcp/tools", get(mcp_tools_handler))
        .route("/v1/models/status", get(models_status_handler))
        .route("/v1/models/select", post(models_select_handler))
        .with_state(state)
}

async fn models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let created = get_created_time_secs();
    let data: Vec<_> = state
        .models
        .list()
        .into_iter()
        .map(|m| {
            json!({
                "id": m.id,
                "object": m.object,
                "created": created,
                "owned_by": m.owned_by,
                "permission": []
            })
        })
        .collect();
    Json(json!({
        "object": "list",
        "data": data
    }))
}

async fn mcp_tools_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    if let Some(session) = &state.mcp {
        match session.list_openai_tools(None).await {
            Ok(tools) => {
                let serialized = serde_json::to_value(&tools).unwrap_or(json!([]));
                return Json(json!({
                    "object": "list",
                    "data": serialized
                }));
            }
            Err(err) => {
                return Json(json!({
                    "error": format!("MCP tools unavailable: {err}")
                }));
            }
        }
    }
    Json(json!({
        "object": "list",
        "data": [],
        "warning": "No MCP configuration loaded"
    }))
}

async fn models_status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    if let Some(ref _manager) = state.model_manager {
        // Placeholder: return basic status from manager
        Json(serde_json::json!({"status": "active", "manager": "enabled"}))
    } else {
        let status = state.models.status().await;
        Json(serde_json::to_value(status).unwrap_or(serde_json::json!({"status": "unknown"})))
    }
}

#[derive(Deserialize)]
struct SelectModelRequest {
    model_id: String,
}

async fn models_select_handler(
    State(state): State<AppState>,
    Json(req): Json<SelectModelRequest>,
) -> Json<serde_json::Value> {
    if let Some(alias) = state.models.resolve(&req.model_id) {
        if let Some(ref manager) = state.model_manager {
            match manager.enqueue_switch(&alias.name) {
                Ok(_) => Json(json!({
                    "status": "switching",
                    "model": alias.name,
                    "queue_length": manager.queue_length(&alias.name),
                })),
                Err(e) => Json(json!({
                    "error": format!("Failed to switch model: {e}")
                })),
            }
        } else {
            state.models.set_active(alias.name.clone()).await;
            Json(json!({
                "status": "switching",
                "model": alias.name,
            }))
        }
    } else {
        Json(json!({
            "error": format!("model '{}' not found in registry", req.model_id)
        }))
    }
}
