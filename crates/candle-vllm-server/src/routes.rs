use crate::models_config::ModelsState;
use crate::state::mailbox_service::{now_secs as mailbox_now, MailboxRecord, MailboxService};
use crate::state::queue_backends::{MailboxBackend, QueueBackend};
use crate::state::queue_service::{now_secs as queue_now, QueueRecord, QueueService};
use crate::state::request_queue::{QueueError, QueuedRequest};
use crate::state::webhook_service::{RequestWebhookConfig, WebhookMode, WebhookService};
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use candle_vllm_core::openai::requests::ChatCompletionRequest;
use candle_vllm_core::openai::responses::{APIError, ChatResponder};
use candle_vllm_core::openai::utils::get_created_time_secs;
use candle_vllm_core::openai::OpenAIServerData;
use candle_vllm_responses::session::ResponsesSession;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tracing::{error, info, warn};

// Real MCP session using candle-vllm-responses

#[derive(Clone)]
pub struct AppState {
    pub models: ModelsState,
    pub data: Arc<OpenAIServerData>,
    pub mcp: Option<Arc<ResponsesSession>>,
    pub model_manager: Option<Arc<crate::state::model_manager::ModelManager>>,
    pub queue_backend: QueueBackend,
    pub mailbox_backend: MailboxBackend,
    pub queue_service: Arc<QueueService>,
    pub mailbox_service: Arc<MailboxService>,
    pub webhook_service: Arc<WebhookService>,
}

/// Extract per-request webhook configuration from headers.
fn extract_webhook_config(headers: &HeaderMap) -> Option<RequestWebhookConfig> {
    let url = headers
        .get("x-webhook-url")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    let mode = headers
        .get("x-webhook-mode")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<WebhookMode>().ok());

    let bearer_token = headers
        .get("x-webhook-bearer")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    // Only return config if at least one field is set
    if url.is_some() || mode.is_some() || bearer_token.is_some() {
        Some(RequestWebhookConfig {
            url,
            mode,
            bearer_token,
            headers: None,
        })
    } else {
        None
    }
}

/// Thread-pool based chat completions handler that avoids Send trait violations
/// by keeping model access within worker threads
async fn chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
) -> ChatResponder {
    let request_id = uuid::Uuid::new_v4().to_string();
    // Messages is exported from candle-vllm-openai, just log without count for now
    info!(
        "📨 HANDLER: Received chat completion request - model={}, stream={:?}",
        req.model, req.stream
    );

    // Extract per-request webhook configuration from headers
    let request_webhook_config = extract_webhook_config(&headers);

    // Extract conversation_id and resource_id from headers if not in request body
    if req.conversation_id.is_none() {
        if let Some(conv_id) = headers
            .get("x-conversation-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
        {
            req.conversation_id = Some(conv_id);
        }
    }
    if req.resource_id.is_none() {
        if let Some(res_id) = headers
            .get("x-resource-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
        {
            req.resource_id = Some(res_id);
        }
    }

    // Handle model switching if needed
    let model_name = &req.model;
    if let Some(alias) = state.models.resolve(model_name) {
        info!(
            "📋 HANDLER: Model resolved - requested='{}', alias='{}'",
            model_name, alias.name
        );
        if let Some(ref manager) = state.model_manager {
            match manager.enqueue_switch(&alias.name) {
                Ok(crate::state::model_manager::SwitchResult::Enqueued) => {
                    info!("🔄 HANDLER: Model switch enqueued for '{}'", alias.name);
                    // Model switch enqueued - queue this request
                    let queue = manager.get_or_create_queue(&alias.name);
                    let (response_tx, response_rx) = oneshot::channel();

                    let queued_request =
                        QueuedRequest::new(alias.name.clone(), req.clone(), Some(response_tx));

                    match queue.enqueue(queued_request) {
                        Ok(()) => {
                            // Record in configured queue backend for observability
                            let record = QueueRecord {
                                id: request_id.clone(),
                                model: alias.name.clone(),
                                queued_at: queue_now(),
                                request: serde_json::to_value(&req).unwrap_or_default(),
                            };
                            if let Err(err) = state.queue_service.enqueue(record) {
                                error!("Failed to record queued request: {}", err);
                            }
                            info!(
                                "⏳ HANDLER: Request queued for model switch to {}",
                                alias.name
                            );
                            // Wait for the response from the queue processor
                            // Use a longer timeout for initial model loads (up to 10 minutes for large model downloads)
                            // The queue timeout is configurable via --request-timeout, but we use a longer timeout here
                            // to account for model downloads which can take several minutes
                            let timeout_duration = Duration::from_secs(600); // 10 minutes for initial loads
                            match tokio::time::timeout(
                                timeout_duration,
                                response_rx
                            ).await {
                                Ok(Ok(responder)) => return responder,
                                Ok(Err(_)) => return ChatResponder::ModelError(APIError::new("Queue processing failed".to_string())),
                                Err(_) => return ChatResponder::ModelError(APIError::new(format!(
                                    "Request timed out in queue after {} seconds. Model may still be loading.",
                                    timeout_duration.as_secs()
                                ))),
                            }
                        }
                        Err(QueueError::QueueFull) => {
                            return ChatResponder::ValidationError(APIError::new(
                                "Request queue is full. Please try again later.".to_string(),
                            ));
                        }
                        Err(e) => {
                            return ChatResponder::ModelError(APIError::new(format!(
                                "Failed to queue request: {}",
                                e
                            )));
                        }
                    }
                }
                Ok(crate::state::model_manager::SwitchResult::AlreadyActive) => {
                    info!(
                        "✅ HANDLER: Model '{}' is already active, proceeding with request",
                        alias.name
                    );
                    // Model is already active, proceed with request
                }
                Err(e) => {
                    return ChatResponder::ValidationError(APIError::new(format!(
                        "Model switching failed: {}",
                        e
                    )));
                }
            }
        }
    } else {
        return ChatResponder::ValidationError(APIError::new(format!(
            "Model '{}' not found in registry",
            model_name
        )));
    }

    // Execute the request in a dedicated thread to avoid Send trait violations
    let data = state.data.clone();
    info!("🚀 HANDLER: Spawning blocking task for request processing");
    let req_for_processing = req.clone();
    let result = tokio::task::spawn_blocking(move || {
        info!("🔧 HANDLER: Inside blocking task, about to process request");
        // This runs in a dedicated thread where parking_lot guards don't cross async boundaries
        tokio::runtime::Handle::current().block_on(async move {
            process_chat_completion_in_thread(data, req_for_processing).await
        })
    })
    .await;

    let responder = match result {
        Ok(responder) => {
            info!("✅ HANDLER: Request processing completed successfully");
            responder
        }
        Err(e) => {
            error!("Thread pool task failed: {}", e);
            ChatResponder::ModelError(APIError::new(format!("Internal processing error: {}", e)))
        }
    };

    let status = match &responder {
        ChatResponder::Completion(_) => "completed",
        ChatResponder::Streamer(_) => "streaming",
        ChatResponder::ValidationError(_) => "validation_error",
        ChatResponder::ModelError(_) => "model_error",
        ChatResponder::InternalError(_) => "internal_error",
    };

    let response_value = match &responder {
        ChatResponder::Completion(c) => serde_json::to_value(c).ok(),
        _ => None,
    };

    let mailbox_record = MailboxRecord {
        request_id: request_id.clone(),
        model: req.model.clone(),
        created: mailbox_now(),
        status: status.to_string(),
        response: response_value.clone(),
    };

    if let Err(err) = state.mailbox_service.store(mailbox_record) {
        error!("Failed to store mailbox record: {}", err);
    }

    // Check if we should fire a webhook
    let should_fire_webhook = {
        // Check per-request mode first
        if let Some(ref config) = request_webhook_config {
            match config.mode {
                Some(WebhookMode::Always) => true,
                Some(WebhookMode::Never) => false,
                Some(WebhookMode::OnDisconnect) => false, // Only fires on disconnect, not here
                None => {
                    // Fall back to global config
                    state.webhook_service.is_enabled()
                        && state.webhook_service.default_mode() == WebhookMode::Always
                }
            }
        } else {
            // Use global config
            state.webhook_service.is_enabled()
                && state.webhook_service.default_mode() == WebhookMode::Always
        }
    };

    // Fire webhook if configured for on_complete
    if should_fire_webhook {
        let backend_record = crate::state::backend_traits::MailboxRecord {
            request_id: request_id.clone(),
            model: req.model.clone(),
            created: mailbox_now(),
            status: status.to_string(),
            response: response_value,
        };

        let webhook_service = state.webhook_service.clone();
        let webhook_config = request_webhook_config.clone();

        // Fire webhook in background to not block response
        tokio::spawn(async move {
            if let Err(e) = webhook_service
                .fire_with_retry(&backend_record, webhook_config.as_ref())
                .await
            {
                warn!(
                    request_id = %backend_record.request_id,
                    error = %e,
                    "Failed to deliver webhook on completion"
                );
            }
        });
    }

    responder
}

/// Process chat completion within a dedicated thread context
/// This avoids Send trait violations by keeping all model access within the same thread
async fn process_chat_completion_in_thread(
    data: Arc<OpenAIServerData>,
    request: ChatCompletionRequest,
) -> ChatResponder {
    info!(
        "🎯 THREAD: Starting chat_completions_with_data - model={}, stream={:?}",
        request.model, request.stream
    );
    // Import the core function that has the Send trait issues
    // We're calling it from within a dedicated thread context now
    use candle_vllm_core::openai::openai_server::chat_completions_with_data;

    let result = chat_completions_with_data(data, request).await;
    info!("🏁 THREAD: chat_completions_with_data returned");
    result
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/mcp/tools", get(mcp_tools_handler))
        .route("/v1/models/status", get(models_status_handler))
        .route("/v1/models/select", post(models_select_handler))
        .route("/v1/usage", get(usage_handler))
        .route("/v1/queues", get(queue_list_handler))
        .route("/v1/queues/{model}", get(queue_model_handler))
        .route("/v1/mailbox", get(mailbox_list_handler))
        .route(
            "/v1/mailbox/{request_id}",
            get(mailbox_get_handler).delete(mailbox_delete_handler),
        )
        .route(
            "/v1/mailbox/{request_id}/webhook",
            post(mailbox_trigger_webhook_handler),
        )
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

async fn queue_list_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let records = state.queue_service.list(None);
    let counts = records
        .iter()
        .fold(std::collections::HashMap::new(), |mut acc, r| {
            *acc.entry(r.model.clone()).or_insert(0usize) += 1;
            acc
        });
    Json(json!({
        "object": "list",
        "counts": counts,
        "data": records
    }))
}

async fn queue_model_handler(
    State(state): State<AppState>,
    axum::extract::Path(model): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    let records = state.queue_service.list(Some(&model));
    Json(json!({
        "object": "list",
        "model": model,
        "count": records.len(),
        "data": records
    }))
}

async fn mailbox_list_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let records = state.mailbox_service.list();
    Json(json!({
        "object": "list",
        "count": records.len(),
        "data": records
    }))
}

/// Query parameters for mailbox GET endpoint.
#[derive(Debug, Deserialize, Default)]
pub struct MailboxGetParams {
    /// If true, delete the record after retrieval (atomic get-and-delete).
    #[serde(default)]
    pub auto_delete: bool,
}

/// GET /v1/mailbox/:request_id
///
/// Retrieve a mailbox record by request_id.
///
/// Query parameters:
/// - `auto_delete`: If true, atomically retrieve and delete the record.
///   This is useful for one-time retrieval patterns where the client
///   wants to ensure the record is cleaned up after reading.
async fn mailbox_get_handler(
    State(state): State<AppState>,
    axum::extract::Path(request_id): axum::extract::Path<String>,
    Query(params): Query<MailboxGetParams>,
) -> axum::response::Response {
    if params.auto_delete {
        // Atomic get-and-delete
        if let Some(record) = state.mailbox_service.get(&request_id) {
            // Delete after successful retrieval
            let _ = state.mailbox_service.delete(&request_id);
            Json(json!({
                "object": "mailbox.record",
                "deleted": true,
                "data": record
            }))
            .into_response()
        } else {
            axum::http::StatusCode::NOT_FOUND.into_response()
        }
    } else {
        // Normal get
        if let Some(record) = state.mailbox_service.get(&request_id) {
            Json(record).into_response()
        } else {
            axum::http::StatusCode::NOT_FOUND.into_response()
        }
    }
}

async fn mailbox_delete_handler(
    State(state): State<AppState>,
    axum::extract::Path(request_id): axum::extract::Path<String>,
) -> axum::response::Response {
    if state.mailbox_service.delete(&request_id) {
        axum::http::StatusCode::NO_CONTENT.into_response()
    } else {
        axum::http::StatusCode::NOT_FOUND.into_response()
    }
}

/// Request body for manually triggering a webhook.
#[derive(Debug, Deserialize, Default)]
pub struct TriggerWebhookRequest {
    /// Override the webhook URL for this trigger.
    pub url: Option<String>,
    /// Override the bearer token for this trigger.
    pub bearer_token: Option<String>,
    /// Additional headers to include.
    pub headers: Option<std::collections::HashMap<String, String>>,
}

/// POST /v1/mailbox/:request_id/webhook
///
/// Manually trigger a webhook for a mailbox record.
///
/// This endpoint allows you to:
/// - Re-send a webhook that may have failed
/// - Send a webhook to a different URL than configured
/// - Test webhook integrations
///
/// The mailbox record must exist; this does not delete the record.
async fn mailbox_trigger_webhook_handler(
    State(state): State<AppState>,
    axum::extract::Path(request_id): axum::extract::Path<String>,
    Json(req): Json<TriggerWebhookRequest>,
) -> axum::response::Response {
    // Get the mailbox record
    let record = match state.mailbox_service.get(&request_id) {
        Some(r) => r,
        None => {
            return (
                axum::http::StatusCode::NOT_FOUND,
                Json(json!({
                    "error": "mailbox record not found",
                    "request_id": request_id
                })),
            )
                .into_response();
        }
    };

    // Check if webhooks are configured
    if !state.webhook_service.is_enabled() && req.url.is_none() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "webhooks not configured and no URL provided",
                "hint": "Either configure webhooks in models.yaml or provide a 'url' in the request body"
            })),
        )
            .into_response();
    }

    // Build per-request config from the request body
    let request_config = RequestWebhookConfig {
        url: req.url.clone(),
        mode: Some(WebhookMode::Always),
        bearer_token: req.bearer_token,
        headers: req.headers,
    };

    // Convert MailboxRecord from old service to new backend_traits type
    let backend_record = crate::state::backend_traits::MailboxRecord {
        request_id: record.request_id.clone(),
        model: record.model.clone(),
        created: record.created,
        status: record.status.clone(),
        response: record.response.clone(),
    };

    // Fire the webhook
    match state
        .webhook_service
        .fire_with_retry(&backend_record, Some(&request_config))
        .await
    {
        Ok(()) => Json(json!({
            "status": "delivered",
            "request_id": request_id,
            "url": req.url.as_deref().unwrap_or("default")
        }))
        .into_response(),
        Err(e) => {
            warn!(
                request_id = %request_id,
                error = %e,
                "Failed to deliver webhook"
            );
            (
                axum::http::StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": "webhook delivery failed",
                    "details": e.to_string(),
                    "request_id": request_id
                })),
            )
                .into_response()
        }
    }
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

async fn usage_handler() -> Json<serde_json::Value> {
    // Return usage statistics for the current session
    // This endpoint is expected by rustchatui but we don't track detailed usage stats yet
    // Return a basic response to prevent 404 errors
    Json(json!({
        "object": "usage",
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "message": "Usage tracking not yet implemented"
    }))
}
