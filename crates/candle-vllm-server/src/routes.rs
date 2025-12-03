use crate::models_config::ModelsState;
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use candle_vllm_core::openai::openai_server::chat_completions_with_data;
use candle_vllm_core::openai::requests::ChatCompletionRequest;
use candle_vllm_core::openai::OpenAIServerData;
use candle_vllm_responses::session::ResponsesSession;
use candle_vllm_responses::status::ModelStatus;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, info, warn};

#[derive(Clone)]
pub struct AppState {
    pub models: ModelsState,
    pub data: Arc<OpenAIServerData>,
    pub mcp: Option<Arc<ResponsesSession>>,
    pub model_manager: Option<Arc<crate::state::model_manager::ModelManager>>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(models_handler))
        .route(
            "/v1/chat/completions",
            post({
                move |State(state): State<AppState>, Json(mut req): Json<ChatCompletionRequest>| async move {
                    // Auto-inject MCP tools if available and not already specified
                    if req.tools.is_none() || req.tools.as_ref().map(|t| t.is_empty()).unwrap_or(true) {
                        if let Some(ref mcp_session) = state.mcp {
                            match mcp_session.list_openai_tools(None).await {
                                Ok(tools) => {
                                    if !tools.is_empty() {
                                        info!("Auto-injecting {} MCP tools into request", tools.len());
                                        req.tools = Some(tools);
                                    } else {
                                        debug!("MCP session available but no tools found");
                                    }
                                }
                                Err(err) => {
                                    warn!("Failed to list MCP tools: {}", err);
                                }
                            }
                        } else {
                            debug!("No MCP session available for tool injection");
                        }
                    } else {
                        debug!("Request already specifies tools, skipping auto-injection");
                    }
                    chat_completions_with_data(state.data.clone(), req).await
                }
            }),
        )
        .route("/v1/mcp/tools", get(mcp_tools_handler))
        .route("/v1/models/status", get(models_status_handler))
        .route("/v1/models/select", post(models_select_handler))
        .with_state(state)
}

async fn models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
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

async fn models_status_handler(State(state): State<AppState>) -> Json<ModelStatus> {
    if let Some(ref manager) = state.model_manager {
        Json(manager.status())
    } else {
        let status = state.models.status().await;
        Json(status)
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
