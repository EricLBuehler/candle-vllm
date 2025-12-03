use crate::models_config::ModelsState;
use axum::{
    extract::State,
    http::HeaderMap,
    response::IntoResponse,
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
use std::collections::HashSet;
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
                move |State(state): State<AppState>, headers: HeaderMap, Json(mut req): Json<ChatCompletionRequest>| async move {
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
                    
                    // Always try to add MCP tools if available
                    // If client provided tools, merge MCP tools with them
                    // Otherwise, use only MCP tools
                    if let Some(ref mcp_session) = state.mcp {
                        match mcp_session.list_openai_tools(None).await {
                            Ok(mcp_tools) => {
                                if !mcp_tools.is_empty() {
                                    match req.tools {
                                        Some(ref mut client_tools) => {
                                            // Client provided tools: merge MCP tools with client tools
                                            // Deduplicate by function name to avoid conflicts
                                            let existing_names: HashSet<String> = client_tools
                                                .iter()
                                                .map(|t| t.function.name.clone())
                                                .collect();
                                            
                                            let mcp_tools_len = mcp_tools.len();
                                            let new_mcp_tools: Vec<_> = mcp_tools
                                                .into_iter()
                                                .filter(|t| !existing_names.contains(&t.function.name))
                                                .collect();
                                            
                                            if !new_mcp_tools.is_empty() {
                                                info!(
                                                    "Merging {} MCP tools with {} client-provided tools ({} duplicates skipped)",
                                                    new_mcp_tools.len(),
                                                    client_tools.len(),
                                                    mcp_tools_len - new_mcp_tools.len()
                                                );
                                                client_tools.extend(new_mcp_tools);
                                            } else {
                                                debug!("All {} MCP tools already present in client-provided tools", mcp_tools_len);
                                            }
                                        }
                                        None => {
                                            // No client tools: use only MCP tools
                                            info!("Auto-injecting {} MCP tools into request", mcp_tools.len());
                                            req.tools = Some(mcp_tools);
                                        }
                                    }
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
                    
                    // Track which tools are MCP tools (have server:: prefix)
                    let mcp_tool_names: HashSet<String> = if let Some(ref mcp_session) = state.mcp {
                        match mcp_session.list_openai_tools(None).await {
                            Ok(tools) => tools.iter()
                                .map(|t| t.function.name.clone())
                                .collect(),
                            Err(_) => HashSet::new(),
                        }
                    } else {
                        HashSet::new()
                    };
                    
                    // Extract original messages and clone req before it's moved
                    let original_messages = match &req.messages {
                        candle_vllm_core::openai::requests::Messages::Chat(msgs) => msgs.clone(),
                        _ => vec![],
                    };
                    let req_clone = req.clone();
                    
                    // Get the initial response
                    let responder = chat_completions_with_data(state.data.clone(), req).await;
                    
                    // Handle tool calls: execute MCP tools automatically, forward client tools
                    match responder {
                        candle_vllm_core::openai::responses::ChatResponder::Completion(mut response) => {
                            // Check if there are tool calls in the response
                            if let Some(choice) = response.choices.first_mut() {
                                if let Some(ref tool_calls) = choice.message.tool_calls {
                                    // Separate MCP tool calls from client tool calls (clone to get owned values)
                                    let (mcp_tool_calls, client_tool_calls): (Vec<_>, Vec<_>) = tool_calls
                                        .iter()
                                        .map(|tc| (tc.clone(), mcp_tool_names.contains(&tc.function.name)))
                                        .partition(|(_, is_mcp)| *is_mcp);
                                    let mcp_tool_calls: Vec<_> = mcp_tool_calls.into_iter().map(|(tc, _)| tc).collect();
                                    let client_tool_calls: Vec<_> = client_tool_calls.into_iter().map(|(tc, _)| tc).collect();
                                    
                                    if !mcp_tool_calls.is_empty() {
                                        info!("Executing {} MCP tool call(s) automatically", mcp_tool_calls.len());
                                        
                                        // Execute MCP tool calls
                                        if let Some(ref mcp_session) = state.mcp {
                                            match mcp_session.execute_mcp_tool_calls(&mcp_tool_calls).await {
                                                Ok(tool_responses) => {
                                                    info!("Successfully executed {} MCP tool call(s)", mcp_tool_calls.len());
                                                    
                                                    // Build conversation with tool results
                                                    let mut messages = original_messages;
                                                    messages.push(candle_vllm_core::openai::requests::ChatMessage {
                                                        role: "assistant".to_string(),
                                                        content: choice.message.content.clone(),
                                                        tool_calls: Some(mcp_tool_calls.clone()),
                                                        tool_call_id: None,
                                                        name: None,
                                                    });
                                                    messages.extend(tool_responses);
                                                    
                                                    // Continue conversation with tool results
                                                    let mut follow_up_req = req_clone.clone();
                                                    follow_up_req.messages = candle_vllm_core::openai::requests::Messages::Chat(messages);
                                                    // Keep tools in case there are more tool calls needed
                                                    
                                                    // Get follow-up response
                                                    let follow_up_responder = chat_completions_with_data(state.data.clone(), follow_up_req).await;
                                                    
                                                    match follow_up_responder {
                                                        candle_vllm_core::openai::responses::ChatResponder::Completion(mut follow_up_response) => {
                                                            // Merge client tool calls into the follow-up response if any
                                                            if !client_tool_calls.is_empty() {
                                                                info!("Returning {} client tool call(s) to client after MCP execution", client_tool_calls.len());
                                                                if let Some(follow_up_choice) = follow_up_response.choices.first_mut() {
                                                                    if let Some(ref mut existing_tool_calls) = follow_up_choice.message.tool_calls {
                                                                        existing_tool_calls.extend(client_tool_calls);
                                                                    } else {
                                                                        follow_up_choice.message.tool_calls = Some(client_tool_calls);
                                                                    }
                                                                }
                                                            }
                                                            Json(follow_up_response).into_response()
                                                        }
                                                        _ => Json(response).into_response(),
                                                    }
                                                }
                                                Err(e) => {
                                                    warn!("Failed to execute MCP tool calls: {}", e);
                                                    // Return original response with client tool calls
                                                    if !client_tool_calls.is_empty() {
                                                        choice.message.tool_calls = Some(client_tool_calls);
                                                    } else {
                                                        choice.message.tool_calls = None;
                                                    }
                                                    Json(response).into_response()
                                                }
                                            }
                                        } else {
                                            // No MCP session but MCP tools were called - shouldn't happen
                                            warn!("MCP tool calls detected but no MCP session available");
                                            Json(response).into_response()
                                        }
                                    } else if !client_tool_calls.is_empty() {
                                        // Only client tool calls - return them to client
                                        info!("Returning {} client tool call(s) to client for execution", client_tool_calls.len());
                                        choice.message.tool_calls = Some(client_tool_calls);
                                        Json(response).into_response()
                                    } else {
                                        // No tool calls
                                        Json(response).into_response()
                                    }
                                } else {
                                    // No tool calls in response
                                    Json(response).into_response()
                                }
                            } else {
                                Json(response).into_response()
                            }
                        }
                        _ => responder.into_response(),
                    }
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
