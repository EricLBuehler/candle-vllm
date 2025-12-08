//! AI Gateway Example
//!
//! This example demonstrates how to build an AI gateway that routes requests
//! to multiple backend models. Key features demonstrated:
//!
//! - Multi-model routing based on request parameters
//! - Model aliasing and fallback
//! - Request queuing and load balancing concepts
//! - OpenAI-compatible API surface
//!
//! # Building
//!
//! For CUDA (NVIDIA):
//! ```bash
//! cargo build --release --features cuda
//! ```
//!
//! For Metal (Apple Silicon):
//! ```bash
//! cargo build --release --features metal
//! ```
//!
//! # Architecture
//!
//! The gateway receives OpenAI-compatible requests and routes them to the
//! appropriate backend model based on the `model` field in the request.
//! This allows a single endpoint to serve multiple models.

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use candle_vllm_openai::{ChatCompletionRequest, ChatCompletionResponse};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Model backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBackend {
    /// Display name for the model
    pub name: String,
    /// HuggingFace model ID or path
    pub model_id: String,
    /// Whether this backend is currently active
    pub active: bool,
    /// Priority for load balancing (higher = preferred)
    pub priority: u32,
}

/// Gateway state shared across request handlers
#[derive(Clone)]
pub struct GatewayState {
    /// Available model backends
    backends: Arc<RwLock<HashMap<String, ModelBackend>>>,
    /// Model aliases (e.g., "gpt-3.5-turbo" -> "mistral-7b")
    aliases: Arc<HashMap<String, String>>,
    /// Request counter for simple load tracking
    request_count: Arc<RwLock<u64>>,
}

impl GatewayState {
    pub fn new() -> Self {
        // Set up demo backends
        let mut backends = HashMap::new();
        
        backends.insert(
            "mistral-7b".to_string(),
            ModelBackend {
                name: "Mistral 7B Instruct".to_string(),
                model_id: "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
                active: true,
                priority: 10,
            },
        );
        
        backends.insert(
            "llama-3-8b".to_string(),
            ModelBackend {
                name: "Llama 3 8B Instruct".to_string(),
                model_id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                active: true,
                priority: 10,
            },
        );
        
        backends.insert(
            "qwen-7b".to_string(),
            ModelBackend {
                name: "Qwen2 7B Instruct".to_string(),
                model_id: "Qwen/Qwen2-7B-Instruct".to_string(),
                active: false, // Disabled by default
                priority: 5,
            },
        );

        // Set up aliases to map OpenAI model names to our backends
        let mut aliases = HashMap::new();
        aliases.insert("gpt-3.5-turbo".to_string(), "mistral-7b".to_string());
        aliases.insert("gpt-4".to_string(), "llama-3-8b".to_string());
        aliases.insert("gpt-4-turbo".to_string(), "llama-3-8b".to_string());
        aliases.insert("claude-3-opus".to_string(), "llama-3-8b".to_string());
        aliases.insert("claude-3-sonnet".to_string(), "mistral-7b".to_string());

        Self {
            backends: Arc::new(RwLock::new(backends)),
            aliases: Arc::new(aliases),
            request_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Resolve a model name to a backend, handling aliases
    pub async fn resolve_backend(&self, model_name: &str) -> Option<(String, ModelBackend)> {
        // Check if it's an alias
        let resolved_name = self
            .aliases
            .get(model_name)
            .cloned()
            .unwrap_or_else(|| model_name.to_string());

        // Look up the backend
        let backends = self.backends.read().await;
        backends
            .get(&resolved_name)
            .filter(|b| b.active)
            .map(|b| (resolved_name.clone(), b.clone()))
    }

    /// Get all available backends
    pub async fn list_backends(&self) -> Vec<(String, ModelBackend)> {
        let backends = self.backends.read().await;
        backends
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Increment request counter
    pub async fn increment_requests(&self) -> u64 {
        let mut count = self.request_count.write().await;
        *count += 1;
        *count
    }
}

impl Default for GatewayState {
    fn default() -> Self {
        Self::new()
    }
}

/// Response for model listing
#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
    backend: String,
    active: bool,
}

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    total_requests: u64,
    backends_available: usize,
}

/// Chat completion handler
async fn chat_completions(
    State(state): State<GatewayState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let request_num = state.increment_requests().await;
    info!("Processing request #{} for model: {}", request_num, request.model);

    // Resolve the model backend
    let (backend_name, backend) = state
        .resolve_backend(&request.model)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or not active", request.model),
            )
        })?;

    info!(
        "Routing to backend '{}' ({})",
        backend_name, backend.model_id
    );

    // In a real implementation, this would call the actual model inference
    // For this demo, we return a mock response
    let response = create_mock_response(&request, &backend_name);

    Ok(Json(response))
}

/// Create a mock response for demonstration
fn create_mock_response(request: &ChatCompletionRequest, backend_name: &str) -> ChatCompletionResponse {
    use candle_vllm_openai::{ChatChoice, ChatChoiceData, ChatCompletionUsageResponse};

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let response_content = format!(
        "This is a demo response from the AI Gateway.\n\n\
         Your request was routed to backend: {}\n\n\
         In a real deployment, this would be the actual model response.\n\n\
         To enable real inference:\n\
         1. Load the model using candle-vllm-core\n\
         2. Pass the request to OpenAIAdapter\n\
         3. Return the actual completion",
        backend_name
    );

    ChatCompletionResponse {
        id: uuid::Uuid::new_v4().to_string(),
        choices: vec![ChatChoice {
            message: ChatChoiceData::text(response_content),
            finish_reason: Some("stop".to_string()),
            index: 0,
            logprobs: None,
        }],
        created,
        model: request.model.clone(),
        object: "chat.completion",
        usage: ChatCompletionUsageResponse {
            request_id: uuid::Uuid::new_v4().to_string(),
            created,
            completion_tokens: 50,
            prompt_tokens: 10,
            total_tokens: 60,
            prompt_time_costs: 10,
            completion_time_costs: 100,
        },
        conversation_id: request.conversation_id.clone(),
        resource_id: request.resource_id.clone(),
    }
}

/// List available models
async fn list_models(State(state): State<GatewayState>) -> Json<ModelsResponse> {
    let backends = state.list_backends().await;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let data: Vec<ModelInfo> = backends
        .into_iter()
        .map(|(id, backend)| ModelInfo {
            id: id.clone(),
            object: "model",
            created,
            owned_by: "candle-vllm".to_string(),
            backend: backend.model_id,
            active: backend.active,
        })
        .collect();

    Json(ModelsResponse {
        object: "list",
        data,
    })
}

/// Health check endpoint
async fn health_check(State(state): State<GatewayState>) -> Json<HealthResponse> {
    let total_requests = *state.request_count.read().await;
    let backends = state.backends.read().await;
    let active_count = backends.values().filter(|b| b.active).count();

    Json(HealthResponse {
        status: "healthy",
        total_requests,
        backends_available: active_count,
    })
}

/// Build the router
fn build_router(state: GatewayState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("candle_vllm_ai_gateway_example=info".parse().unwrap())
                .add_directive("tower_http=info".parse().unwrap()),
        )
        .init();

    info!("candle-vllm AI Gateway Example");
    info!("===============================");
    info!("");
    info!("This example demonstrates multi-model routing.");
    info!("");

    // Create gateway state
    let state = GatewayState::new();

    // List available backends
    let backends = state.list_backends().await;
    info!("Available backends:");
    for (name, backend) in &backends {
        let status = if backend.active { "active" } else { "inactive" };
        info!("  - {} ({}) [{}]", name, backend.model_id, status);
    }
    info!("");

    // Build router with CORS support
    let app = build_router(state).layer(
        tower_http::cors::CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any),
    );

    // Start server
    let addr = "0.0.0.0:8080";
    info!("Starting gateway server on http://{}", addr);
    info!("");
    info!("Endpoints:");
    info!("  POST /v1/chat/completions - Chat completion (OpenAI-compatible)");
    info!("  GET  /v1/models          - List available models");
    info!("  GET  /health             - Health check");
    info!("");
    info!("Model aliases:");
    info!("  gpt-3.5-turbo  -> mistral-7b");
    info!("  gpt-4          -> llama-3-8b");
    info!("  claude-3-opus  -> llama-3-8b");
    info!("");
    info!("Example request:");
    info!(r#"  curl -X POST http://localhost:8080/v1/chat/completions \"#);
    info!(r#"    -H "Content-Type: application/json" \"#);
    info!(r#"    -d '{{"model": "gpt-3.5-turbo", "messages": [{{"role": "user", "content": "Hello!"}}]}}'"#);
    info!("");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resolve_backend_direct() {
        let state = GatewayState::new();
        let result = state.resolve_backend("mistral-7b").await;
        assert!(result.is_some());
        let (name, _) = result.unwrap();
        assert_eq!(name, "mistral-7b");
    }

    #[tokio::test]
    async fn test_resolve_backend_alias() {
        let state = GatewayState::new();
        let result = state.resolve_backend("gpt-3.5-turbo").await;
        assert!(result.is_some());
        let (name, _) = result.unwrap();
        assert_eq!(name, "mistral-7b");
    }

    #[tokio::test]
    async fn test_resolve_backend_unknown() {
        let state = GatewayState::new();
        let result = state.resolve_backend("unknown-model").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_resolve_backend_inactive() {
        let state = GatewayState::new();
        // qwen-7b is inactive by default
        let result = state.resolve_backend("qwen-7b").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_request_counter() {
        let state = GatewayState::new();
        assert_eq!(state.increment_requests().await, 1);
        assert_eq!(state.increment_requests().await, 2);
        assert_eq!(state.increment_requests().await, 3);
    }
}

