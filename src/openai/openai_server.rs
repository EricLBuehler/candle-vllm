use super::requests::Messages;
use super::requests::{ChatCompletionRequest, EmbeddingRequest, EmbeddingType, EncodingFormat};
use super::responses::{APIError, ChatCompletionResponse, ChatResponder};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::{ChatResponse, Streamer, StreamingStatus};
use super::OpenAIServerData;
use crate::tools::{Tool, ToolChoice, ToolFormat};
use axum::response::sse::KeepAlive;
use axum::{
    extract::{Json, State},
    response::Sse,
};
use flume;
use std::env;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Notify;
use tokio::time::Duration;
use tracing::debug;
use uuid::Uuid;

#[derive(Debug, Clone)]
enum ToolChoiceKind {
    Auto,
    None,
    Function(String),
}

#[derive(Debug, Clone)]
struct ResolvedToolConfig {
    tools: Vec<Tool>,
    choice: ToolChoiceKind,
}

fn normalize_tool_choice(choice: &Option<ToolChoice>) -> ToolChoiceKind {
    match choice {
        None => ToolChoiceKind::Auto,
        Some(ToolChoice::Function { function, .. }) => {
            ToolChoiceKind::Function(function.name.clone())
        }
        Some(ToolChoice::Auto(value)) | Some(ToolChoice::None(value)) => match value.as_str() {
            "none" => ToolChoiceKind::None,
            "auto" => ToolChoiceKind::Auto,
            _ => ToolChoiceKind::Auto,
        },
    }
}

fn resolve_tools_for_request(
    request_tools: &Option<Vec<Tool>>,
    tool_choice: &Option<ToolChoice>,
    mcp_manager: Option<&Arc<crate::mcp::McpClientManager>>,
) -> Result<ResolvedToolConfig, APIError> {
    let choice = normalize_tool_choice(tool_choice);
    let mut tools = if let Some(req_tools) = request_tools {
        if req_tools.is_empty() {
            Vec::new()
        } else {
            req_tools.clone()
        }
    } else if let Some(manager) = mcp_manager {
        manager.cached_tools()
    } else {
        Vec::new()
    };

    if matches!(choice, ToolChoiceKind::None) {
        tools.clear();
        return Ok(ResolvedToolConfig { tools, choice });
    }

    if let ToolChoiceKind::Function(name) = &choice {
        if tools.is_empty() {
            return Err(APIError::new(format!(
                "tool_choice '{}' requires tools to be provided.",
                name
            )));
        }
        tools.retain(|tool| tool.function.name == *name);
        if tools.is_empty() {
            return Err(APIError::new(format!(
                "tool_choice '{}' not found in tools.",
                name
            )));
        }
    }

    Ok(ResolvedToolConfig { tools, choice })
}

// Get prompt, roles
async fn get_gen_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
    tool_config: &ResolvedToolConfig,
) -> Result<String, APIError> {
    let mut model = data.model.write();
    let pipeline = model
        .get_mut_pipeline(0)
        .ok_or(APIError::new("Missing pipeline".to_string()))?;
    let mut conversation = pipeline.0.get_conversation().clone();

    match &request.messages {
        Messages::Literal(msg) => {
            return Ok(msg.clone());
        }
        Messages::Chat(messages) => {
            for message in messages {
                let role = message.role.as_str();
                if role == "system" {
                    if let Some(content) = &message.content {
                        tracing::info!("system prompt found: {}", content);
                        conversation.set_system_message(Some(content.clone()));
                    }
                    continue;
                }

                if role == "tool" {
                    let tool_call_id = message.tool_call_id.as_deref().unwrap_or("unknown");
                    let content = message.content.clone().unwrap_or_default();
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        let prompt = format!("[Tool Result for {}]: {}", tool_call_id, trimmed);
                        conversation.append_message(role.to_string(), prompt);
                    }
                    continue;
                }

                if let Some(tool_calls) = &message.tool_calls {
                    let mut tool_text = String::new();
                    for tc in tool_calls {
                        tool_text.push_str(&format!(
                            "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>\n",
                            tc.function.name, tc.function.arguments
                        ));
                    }
                    if !tool_text.trim().is_empty() {
                        conversation.append_message(role.to_string(), tool_text.trim().to_string());
                    }
                    continue;
                }

                if let Some(content) = &message.content {
                    conversation.append_message(role.to_string(), content.clone());
                }
            }
        }
        Messages::Map(messages) => {
            for message in messages {
                let role = message
                    .get("role")
                    .ok_or(APIError::new("Message key `role` not found.".to_string()))?;
                let content = message
                    .get("content")
                    .ok_or(APIError::new(
                        "Message key `content` not found.".to_string(),
                    ))?
                    .clone();

                if role == "system" {
                    tracing::info!("system prompt found: {}", content);
                    conversation.set_system_message(Some(content.clone()));
                } else {
                    conversation.append_message(role.to_string(), content)
                }
            }
        }
    }

    if !tool_config.tools.is_empty() {
        let mut tools_prompt = ToolFormat::format_tools(&tool_config.tools);
        if let ToolChoiceKind::Function(name) = &tool_config.choice {
            tools_prompt = format!(
                "IMPORTANT: You must call the tool \"{}\".\n\n{}",
                name, tools_prompt
            );
        }
        let current_system = conversation.get_system_message().unwrap_or_default();
        let new_system = if current_system.is_empty() {
            tools_prompt
        } else {
            format!("{}\n\n{}", current_system, tools_prompt)
        };
        conversation.set_system_message(Some(new_system));
    }

    Ok(conversation.get_prompt(request.thinking.unwrap_or(false)))
}

async fn check_length(
    request: &ChatCompletionRequest,
    prompt: String,
    data: &OpenAIServerData,
) -> Result<(Vec<u32>, usize), APIError> {
    let (token_ids, available_kv_tokens) = {
        let model = data.model.read();
        let available_kv_tokens = model.get_available_kv_tokens();
        let pipeline = model
            .get_pipeline(0)
            .ok_or(APIError::new("Missing pipeline".to_string()))?;
        (
            pipeline
                .0
                .tokenizer()
                .encode_fast(prompt, false)
                .map_err(APIError::from)?
                .get_ids()
                .to_vec(),
            available_kv_tokens,
        )
    };

    let max_gen_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    if token_ids.len() >= data.pipeline_config.max_model_len {
        Err(APIError::new(format!(
            "This model's maximum context length is {} tokens. \
            However, you requested {} tokens ({} in the messages, \
            {} in the completion). \nPlease clear the chat history or reduce the length of the \
            messages.",
            data.pipeline_config.max_model_len,
            max_gen_tokens + token_ids.len(),
            token_ids.len(),
            max_gen_tokens
        )))
    } else if token_ids.len() >= available_kv_tokens {
        Err(APIError::new(format!(
            "Requested prompt({} tokens) is  \
            larger than available kvcache (maximum {} tokens).\n \
            You can increase kvcache by setting `--mem` to a larger value!",
            token_ids.len(),
            available_kv_tokens
        )))
    } else {
        let max_valid_request_tokens =
            std::cmp::min(available_kv_tokens, data.pipeline_config.max_model_len) - 10;
        Ok((token_ids, max_valid_request_tokens))
    }
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((status = 200, description = "Chat completions"))
)]
pub async fn chat_completions(
    State(data): State<Arc<OpenAIServerData>>,
    request: Json<ChatCompletionRequest>,
) -> ChatResponder {
    #[cfg(feature = "nccl")]
    use crate::openai::communicator::DaemonManager;
    #[cfg(feature = "nccl")]
    if !DaemonManager::is_master_rank() {
        return ChatResponder::ModelError(APIError::from(
            "Daemon process unable to generate response, please request server port of the main process!",
        ));
    }

    if request.logit_bias.as_ref().is_some()
        && request.logit_bias.as_ref().is_some_and(|x| !x.is_empty())
    {
        return ChatResponder::ValidationError(APIError::new_str(
            "`logit_bias` is not currently supported.",
        ));
    }

    let tool_config = match resolve_tools_for_request(
        &request.tools,
        &request.tool_choice,
        data.mcp_manager.as_ref(),
    ) {
        Ok(config) => config,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let prompt = match get_gen_prompt(&data, &request, &tool_config).await {
        Ok(p) => p,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let (token_ids, available_tokens): (Vec<u32>, usize) =
        match check_length(&request, prompt.clone(), &data).await {
            Ok(ids) => ids,
            Err(e) => return ChatResponder::ValidationError(e),
        };

    debug!("\n\n\nPrompt {:?}", prompt);

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let mut max_request_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    if max_request_tokens + token_ids.len() > available_tokens {
        tracing::warn!(
            "Requested max tokens + prompt length {} larger than available tokens {}, \
        max_tokens changed to {} ({} tokens reserved for prompt)!",
            max_request_tokens + token_ids.len(),
            available_tokens,
            available_tokens - token_ids.len(),
            token_ids.len()
        );
        max_request_tokens = if available_tokens > token_ids.len() {
            available_tokens - token_ids.len()
        } else {
            return ChatResponder::ValidationError(APIError::new(format!(
                "Requested prompt({} tokens) is  \
                larger than available kvcache (maximum {} tokens).\n \
                You can increase kvcache by setting `--mem` to a larger value!",
                token_ids.len(),
                available_tokens
            )));
        }
    }

    let generation_cfg = data.pipeline_config.generation_cfg.as_ref().unwrap();
    let mut sampling_params = match SamplingParams::new(
        request.n.unwrap_or(1),
        request.best_of,
        request
            .presence_penalty
            .unwrap_or(generation_cfg.presence_penalty.unwrap_or(0.0)),
        request
            .frequency_penalty
            .unwrap_or(generation_cfg.frequency_penalty.unwrap_or(0.0)),
        request.repeat_last_n,
        request.temperature.or(generation_cfg.temperature),
        request.top_p.or(generation_cfg.top_p),
        request.min_p.or(generation_cfg.min_p),
        request.top_k.or(generation_cfg.top_k),
        request.use_beam_search.unwrap_or(false),
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        request.stop.clone(),
        request.stop_token_ids.clone().unwrap_or_default(),
        request.ignore_eos.unwrap_or(false),
        max_request_tokens,
        None,
        None,
        request.skip_special_tokens.unwrap_or(true),
        request.thinking,
    ) {
        Ok(params) => params,
        Err(e) => return ChatResponder::ValidationError(e),
    };
    let has_tools = !tool_config.tools.is_empty();
    sampling_params.mcp_mode = if has_tools { Some(true) } else { None };

    let (response_tx, rx) = flume::unbounded();
    tracing::info!("{:?}", sampling_params);

    let data_clone = data.clone();
    let request_id_clone = request_id.clone();
    let stream_request = request.stream.is_some_and(|x| x);
    let model_name = request.model.clone().unwrap_or("default".to_string());
    let sync_notify = Arc::new(Notify::new());
    let sync_completion_notify = if stream_request {
        None
    } else {
        Some(Arc::clone(&sync_notify))
    };

    let _ = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(async move {
            {
                //send completion request to inference engine
                let mut model = data.model.write();
                model.add_request(
                    token_ids,
                    request_id.clone(),
                    SystemTime::now(),
                    sampling_params,
                    request.logprobs.unwrap_or(false),
                    false,
                    EncodingFormat::default(),
                    EmbeddingType::default(),
                    if stream_request {
                        Some(Arc::new(response_tx))
                    } else {
                        None
                    },
                    sync_completion_notify,
                );
                model.notify.notify_one();
            }
        });
    });

    if stream_request {
        ChatResponder::Streamer(
            Sse::new(Streamer {
                rx,
                status: StreamingStatus::Uninitialized,
            })
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(100))
                            .unwrap_or(100),
                    ))
                    .text("keep-alive-text"),
            ),
        )
    } else {
        // wait until current response finished
        tracing::warn!("waiting response for sync request {}", request_id_clone);
        sync_notify.as_ref().notified().await;
        // Re-acquire read lock to get the response
        // Note: we need to drop the lock later
        let (choices, usage) = {
            let model = data_clone.model.read();
            if !model.completion_records.contains_key(&request_id_clone) {
                return ChatResponder::ModelError(APIError::from(format!(
                    "Unable to generate response for request {request_id_clone}"
                )));
            }
            let record = &model.completion_records[&request_id_clone];
            (record.0.clone(), record.1.clone())
        };

        // Check for tool calls in the output
        let mut final_choices = choices.clone();
        if has_tools {
            let parser = crate::tools::parser::ToolParser::new();
            for choice in &mut final_choices {
                if let Some(content) = &choice.message.content {
                    let calls = parser.parse(content);
                    if !calls.is_empty() {
                        choice.message.tool_calls = Some(calls);
                        choice.message.content = None;
                        choice.finish_reason = Some("tool_calls".to_string());
                    }
                }
            }
        }

        ChatResponder::Completion(ChatCompletionResponse {
            id: request_id_clone,
            choices: final_choices,
            created: usage.created,
            model: model_name,
            object: "chat.completion",
            usage: usage,
        })
    }
}

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/embeddings",
    request_body = EmbeddingRequest,
    responses((status = 200, description = "Embeddings"))
)]
pub async fn create_embeddings(
    State(data): State<Arc<OpenAIServerData>>,
    request: Json<EmbeddingRequest>,
) -> ChatResponder {
    let input = request.input.clone();
    let prompts = input.into_vec();

    //For now only support single prompt for simplicity, loop if multiple
    if prompts.len() != 1 {
        return ChatResponder::ValidationError(APIError::new_str(
            "Currently only support single string or token array input.",
        ));
    }

    let prompt_str = prompts[0].clone();

    //TODO: Reuse check_length or similar logic. For now simplified.
    let (token_ids, available_tokens) = {
        let model = data.model.read();
        let available_kv_tokens = model.get_available_kv_tokens();
        let pipeline = model
            .get_pipeline(0)
            .ok_or(APIError::new("Missing pipeline".to_string()));

        match pipeline {
            Ok(pipeline) => match pipeline.0.tokenizer().encode_fast(prompt_str, false) {
                Ok(encoding) => (encoding.get_ids().to_vec(), available_kv_tokens),
                Err(e) => return ChatResponder::ValidationError(APIError::from(e)),
            },
            Err(e) => return ChatResponder::ModelError(e),
        }
    };

    if token_ids.len() >= available_tokens {
        return ChatResponder::ValidationError(APIError::new_str("Prompt too long."));
    }

    let request_id = format!("embd-{}", Uuid::new_v4());

    // Create sampling params for embedding (max_tokens=0, etc)
    // We reuse SamplingParams but most fields irrelevant.
    let _generation_cfg = data.pipeline_config.generation_cfg.as_ref().unwrap();
    let sampling_params = match SamplingParams::new(
        1,
        None,
        0.0,
        0.0,
        None,
        None,
        None,
        None,
        None,
        false,
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        None,
        Vec::new(),
        false,
        1,
        None,
        None,
        true,
        None,
    ) {
        Ok(params) => params,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let (response_tx, rx) = flume::unbounded();

    let request_id_clone = request_id.clone();

    let _ = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(async move {
            {
                let mut model = data.model.write();
                model.add_request(
                    token_ids,
                    request_id_clone,
                    SystemTime::now(),
                    sampling_params,
                    false,
                    true, //is_embedding
                    request.encoding_format.clone(),
                    request.embedding_type.clone(),
                    Some(Arc::new(response_tx)),
                    None,
                );
                model.notify.notify_one();
            }
        });
    });

    // Wait for response from channel
    // Embedding is strictly one response.
    match rx.recv_async().await {
        Ok(ChatResponse::Embedding(resp)) => ChatResponder::Embedding(resp),
        Ok(ChatResponse::ModelError(e)) => ChatResponder::ModelError(APIError::new_str(&e)),
        Ok(_) => ChatResponder::InternalError(APIError::new(format!("Unexpected response type"))),
        Err(_) => ChatResponder::InternalError(APIError::new("Channel closed".to_string())),
    }
}
