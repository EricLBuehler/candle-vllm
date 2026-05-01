use super::logger::ChatCompletionLogger;
use super::requests::Messages;
use super::requests::{
    normalize_empty_openai_tool_results, validate_openai_tool_messages, ChatCompletionRequest,
    EmbeddingRequest, EmbeddingType, EncodingFormat,
};
use super::responses::{APIError, ChatCompletionResponse, ChatResponder};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::{ChatResponse, Streamer, StreamingStatus};
use super::OpenAIServerData;
use crate::openai::multimodal::{build_messages_and_images, ImageData};
use crate::openai::{resolve_tools_for_request, ResolvedToolConfig};
use crate::tools::helpers::{
    build_invalid_tool_call_feedback, build_tool_schema_map, filter_tool_calls,
};
use crate::tools::stream_parser::{
    detect_prefilled_reasoning_end_marker, extract_reasoning_content,
};
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

fn current_model_name(data: &OpenAIServerData) -> Result<String, APIError> {
    let model = data.model.read();
    let (pipeline, _) = model
        .get_pipeline(0)
        .ok_or(APIError::new("Missing pipeline".to_string()))?;
    Ok(pipeline.name().to_string())
}

fn resolve_response_model_name(requested: Option<&str>, current: &str) -> String {
    match requested.map(str::trim) {
        None | Some("") | Some("default") => current.to_string(),
        Some(name) => name.to_string(),
    }
}

// Get prompt, roles
async fn get_gen_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
    tool_config: &ResolvedToolConfig,
) -> Result<(String, Option<ImageData>), APIError> {
    let mut model = data.model.write();
    let pipeline = model
        .get_mut_pipeline(0)
        .ok_or(APIError::new("Missing pipeline".to_string()))?;
    let mut conversation = pipeline.0.get_conversation().clone();
    let mut image_data = None;

    match &request.messages {
        Messages::Literal(msg) => {
            conversation.append_message("user".to_string(), msg.clone());
        }
        Messages::Chat(messages) => {
            let (render_messages, images) =
                build_messages_and_images(messages, pipeline.0.image_config.as_ref())
                    .map_err(APIError::from)?;
            image_data = images;
            for message in render_messages {
                let role = message.role.as_str();
                if role == "system" {
                    conversation.set_system_message(Some(message.content.clone()));
                    continue;
                }
                conversation.append_template_message(message);
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
                    conversation.set_system_message(Some(content.clone()));
                } else {
                    use crate::openai::conversation::Message;
                    conversation.append_template_message(Message {
                        role: role.to_string(),
                        content,
                        num_images: 0,
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
        }
    }

    let enable_thinking = request.thinking.unwrap_or(true);
    let prompt = conversation.get_prompt(enable_thinking, &tool_config.tools);

    Ok((prompt, image_data))
}

async fn check_length(
    request: &ChatCompletionRequest,
    prompt: String,
    data: &OpenAIServerData,
) -> Result<(Vec<u32>, usize), APIError> {
    let token_ids = {
        let model = data.model.read();
        let pipeline = model
            .get_pipeline(0)
            .ok_or(APIError::new("Missing pipeline".to_string()))?;
        pipeline
            .0
            .tokenizer()
            .encode_fast(prompt, true)
            .map_err(APIError::from)?
            .get_ids()
            .to_vec()
    };

    let available_kv_tokens = {
        let mut model = data.model.write();
        let (available_kv_tokens, evicted) = model.ensure_available_kv_tokens(token_ids.len());
        if evicted > 0 {
            tracing::warn!(
                "Evicted {} prefix cache block(s) before request length check.",
                evicted
            );
        }
        available_kv_tokens
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
            You can increase kvcache by setting `--gpu-memory-fraction` (default 0.5) to a larger value!",
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
    let mut request = request.0;
    let logger = ChatCompletionLogger::new();
    if let Some(ref l) = logger {
        l.log_request(&request);
    }
    if let Messages::Chat(messages) = &mut request.messages {
        normalize_empty_openai_tool_results(messages);
        if let Err(err) = validate_openai_tool_messages(messages) {
            return ChatResponder::ValidationError(APIError::new(err));
        }
    }

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

    let (prompt, image_data) = match get_gen_prompt(&data, &request, &tool_config).await {
        Ok(p) => p,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    let (token_ids, available_tokens): (Vec<u32>, usize) =
        match check_length(&request, prompt.clone(), &data).await {
            Ok(ids) => ids,
            Err(e) => return ChatResponder::ValidationError(e),
        };

    debug!("\n\n\nPrompt {:?}", prompt);
    if let Some(ref l) = logger {
        l.log_prompt(&prompt);
    }

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let mut max_request_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    // Query prefix cache to determine how many prompt tokens are already cached
    let cached_tokens = {
        let mut model = data.model.write();
        model.query_prefix_cache_match_tokens(&token_ids)
    };
    let new_tokens = token_ids.len().saturating_sub(cached_tokens);

    if max_request_tokens + new_tokens > available_tokens {
        let mut adjusted_max = if available_tokens > new_tokens {
            available_tokens - new_tokens
        } else {
            return ChatResponder::ValidationError(APIError::new(format!(
                "Requested prompt({} tokens, {} new after prefix cache) is  \
                larger than available kvcache (maximum {} tokens).\n \
                You can increase kvcache by setting `--gpu-memory-fraction` (default 0.5) to a larger value!",
                token_ids.len(),
                new_tokens,
                available_tokens
            )));
        };
        // Ensure max_tokens is at least 4096
        if adjusted_max < 4096 {
            adjusted_max = 4096;
        }
        if adjusted_max != max_request_tokens {
            tracing::warn!(
                "Requested max tokens + new prompt tokens {} larger than available tokens {}, \
                max_tokens changed to {} ({} new tokens after prefix cache hit of {} cached tokens)!",
                max_request_tokens + new_tokens,
                available_tokens,
                adjusted_max,
                new_tokens,
                cached_tokens
            );
        }
        max_request_tokens = adjusted_max;
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

    let prefilled_reasoning_end = detect_prefilled_reasoning_end_marker(&prompt);

    let (response_tx, rx) = flume::unbounded();
    tracing::info!("{:?}", sampling_params);

    let data_clone = data.clone();
    let request_id_clone = request_id.clone();
    let stream_request = request.stream.is_some_and(|x| x);
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let model_name = match current_model_name(&data) {
        Ok(current) => resolve_response_model_name(request.model.as_deref(), &current),
        Err(e) => return ChatResponder::ModelError(e),
    };
    let sync_notify = Arc::new(Notify::new());
    let sync_completion_notify = if stream_request {
        None
    } else {
        Some(Arc::clone(&sync_notify))
    };
    let request_tools_for_engine = tool_config.tools.clone();
    let response_tools = tool_config.tools.clone();

    let _ = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(async move {
            {
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
                    request_tools_for_engine.clone(),
                    image_data,
                    if stream_request {
                        Some(Arc::new(response_tx))
                    } else {
                        None
                    },
                    sync_completion_notify,
                    include_usage,
                    prefilled_reasoning_end,
                );
                model.notify.notify_one();
            }
        });
    });

    if stream_request {
        if let Some(ref l) = logger {
            l.log_start_response();
        }
        ChatResponder::Streamer(
            Sse::new(Streamer {
                rx,
                status: StreamingStatus::Uninitialized,
                logger,
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

        let mut final_choices = choices.clone();

        // Extract reasoning content BEFORE tool parsing so that reasoning
        // blocks are preserved even when tool calls consume the remaining
        // content.  Without this, tool parsing sets content=None and the
        // subsequent reasoning extraction finds nothing.
        if crate::stream_as_reasoning_content() && has_tools {
            for choice in &mut final_choices {
                if let Some(text) = choice.message.content.take() {
                    match extract_reasoning_content(&text) {
                        Some((reasoning, remaining)) => {
                            choice.message.content = if remaining.is_empty() {
                                None
                            } else {
                                Some(remaining)
                            };
                            choice.message.reasoning_content = Some(reasoning);
                        }
                        None => {
                            choice.message.content = Some(text);
                        }
                    }
                }
            }
        }

        if has_tools {
            let parser = crate::tools::parser::ToolParser::new();
            let tool_schemas = build_tool_schema_map(&response_tools);
            for choice in &mut final_choices {
                let parsed_calls = if let Some(calls) = choice.message.tool_calls.take() {
                    calls
                } else if let Some(content) = &choice.message.content {
                    parser.parse(content)
                } else {
                    Vec::new()
                };

                if parsed_calls.is_empty() {
                    continue;
                }

                let (valid_calls, invalid_calls) = filter_tool_calls(&parsed_calls, &tool_schemas);
                if !invalid_calls.is_empty() {
                    tracing::warn!(
                        "Dropped {} invalid tool call(s) before response",
                        invalid_calls.len()
                    );
                }
                if valid_calls.is_empty() {
                    if let Some(feedback) =
                        build_invalid_tool_call_feedback(&invalid_calls, &tool_schemas, None)
                    {
                        choice.message.content = Some(feedback);
                    }
                    choice.finish_reason = Some("stop".to_string());
                    continue;
                }

                choice.message.tool_calls = Some(valid_calls);
                choice.message.content = None;
                choice.finish_reason = Some("tool_calls".to_string());
            }
        }

        let response = ChatCompletionResponse {
            id: request_id_clone,
            choices: final_choices,
            created: usage.created,
            model: model_name,
            object: "chat.completion",
            usage: usage,
        };
        if let Some(ref l) = logger {
            l.log_response(&response);
        }
        ChatResponder::Completion(response)
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
    let token_ids = {
        let model = data.model.read();
        let pipeline = model
            .get_pipeline(0)
            .ok_or(APIError::new("Missing pipeline".to_string()));

        match pipeline {
            Ok(pipeline) => match pipeline.0.tokenizer().encode_fast(prompt_str, true) {
                Ok(encoding) => encoding.get_ids().to_vec(),
                Err(e) => return ChatResponder::ValidationError(APIError::from(e)),
            },
            Err(e) => return ChatResponder::ModelError(e),
        }
    };

    let available_tokens = {
        let mut model = data.model.write();
        let (available_tokens, evicted) = model.ensure_available_kv_tokens(token_ids.len());
        if evicted > 0 {
            tracing::warn!(
                "Evicted {} prefix cache block(s) before embedding length check.",
                evicted
            );
        }
        available_tokens
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
                    Vec::new(),
                    None,
                    Some(Arc::new(response_tx)),
                    None,
                    false,
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
