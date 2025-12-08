use super::requests::{ChatCompletionRequest, Messages, ToolCall};
use super::responses::{APIError, ChatChoiceData, ChatCompletionResponse, ChatResponder};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::{ChatResponse, Streamer, StreamingStatus};
use super::tool_parser::{get_tool_parser, ParsedOutput};
use super::utils::get_created_time_secs;
use super::OpenAIServerData;
use axum::extract::State;
use axum::response::sse::KeepAlive;
use axum::{response::Sse, Json};
use flume;
use std::env;
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::Duration;
use tracing::{error, info};
use uuid::Uuid;

#[utoipa::path(
    post,
    tag = "candle-vllm",
    path = "/v1/chat/completions",
    request_body(content = inline(serde_json::Value), description = "Chat completion request"),
    responses((status = 200, description = "Chat completions"))
)]
pub async fn chat_completions(
    State(data): State<Arc<OpenAIServerData>>,
    request: Json<ChatCompletionRequest>,
) -> ChatResponder {
    info!("➡️ CHAT: received chat completion request");
    chat_completions_with_data(data, request.0).await
}

// Process images in multimodal content
#[allow(dead_code)]
async fn process_multimodal_content(
    data: &OpenAIServerData,
    content: &crate::openai::requests::MessageContent,
) -> Result<String, APIError> {
    use crate::openai::image_tool::ImageDescriptionTool;
    use crate::openai::requests::{ContentPart, MessageContent};

    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::Parts(parts) => {
            let mut processed_content = String::new();

            for part in parts {
                match part {
                    ContentPart::Text { text } => {
                        processed_content.push_str(text);
                        processed_content.push('\n');
                    }
                    ContentPart::ImageUrl { image_url } => {
                        if let Some(ref vision_tool) = data.vision_tool {
                            // Process the image through vision tool
                            let description = vision_tool
                                .describe_image(image_url, None)
                                .await
                                .map_err(|e| {
                                    APIError::new(format!("Vision processing failed: {}", e))
                                })?;

                            processed_content
                                .push_str(&format!("[Image: {}]", description.description));
                            processed_content.push('\n');
                        } else {
                            // No vision tool available, add placeholder
                            processed_content.push_str("[Image: vision processing not available]");
                            processed_content.push('\n');
                        }
                    }
                }
            }

            Ok(processed_content)
        }
    }
}

/// Build prompt from messages using the lock-free LLMEngine.
///
/// Uses `model.build_prompt()` which handles conversation formatting
/// without requiring pipeline access.
fn get_gen_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
) -> Result<String, APIError> {
    info!(
        model = %request.model,
        message_count = messages_len(&request.messages),
        tools = request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        "✏️ CHAT: building prompt"
    );

    // Use the engine's stateless prompt builder
    data.model
        .build_prompt(
            &request.messages,
            request.tools.as_ref(),
            request.thinking.unwrap_or(false),
        )
        .map_err(|e| APIError::new(e))
        .map(|prompt| {
            info!(
                model = %request.model,
                prompt_chars = prompt.chars().count(),
                "✅ CHAT: prompt built"
            );
            prompt
        })
}

fn messages_len(messages: &Messages) -> usize {
    match messages {
        Messages::Literal(_) => 1,
        Messages::Chat(list) => list.len(),
        Messages::Map(list) => list.len(),
    }
}

/// Parse the model output for tool calls
#[allow(dead_code)]
fn parse_tool_calls(output: &str, model_name: &str) -> ParsedOutput {
    let parser = get_tool_parser(model_name);
    parser.parse(output)
}

/// Build a ChatChoiceData from parsed output
#[allow(dead_code)]
fn build_choice_data(parsed: ParsedOutput) -> ChatChoiceData {
    match parsed {
        ParsedOutput::Text(text) => ChatChoiceData::text(text),
        ParsedOutput::ToolCalls(tool_calls) => {
            let api_calls: Vec<ToolCall> =
                tool_calls.into_iter().map(|tc| tc.to_tool_call()).collect();
            ChatChoiceData::with_tool_calls(api_calls)
        }
        ParsedOutput::Mixed { text, tool_calls } => {
            let api_calls: Vec<ToolCall> =
                tool_calls.into_iter().map(|tc| tc.to_tool_call()).collect();
            ChatChoiceData::with_content_and_tool_calls(text, api_calls)
        }
    }
}

/// Check if a model supports reasoning/thinking output
///
/// This function checks if a model is known to emit reasoning tokens
/// when thinking mode is enabled. Detection is based on model name patterns.
fn is_reasoning_model(model_name: &str) -> bool {
    let name_lower = model_name.to_lowercase();
    // Known reasoning model patterns
    name_lower.contains("reasoning")
        || name_lower.contains("thinking")
        || name_lower.contains("cot")  // Chain-of-thought
        || name_lower.contains("ministral") && name_lower.contains("reasoning")
        || name_lower.contains("deepseek") && name_lower.contains("r1")
        || name_lower.contains("qwq")
}

/// Check if a token is part of reasoning/thinking output
///
/// Reasoning tokens are detected by:
/// 1. The thinking_enabled flag from the request
/// 2. Model type (must be a reasoning model)
/// 3. Token content patterns (e.g., <think>, </think> tags)
/// 4. The is_reasoning flag on the StreamingToken (if set by the worker)
fn is_reasoning_token(
    token_text: &str,
    _token_id: u32,
    model_name: &str,
    thinking_enabled: bool,
    token_is_reasoning: bool,
) -> bool {
    // If the worker already marked this as a reasoning token, trust it
    if token_is_reasoning {
        return true;
    }

    // Must have thinking enabled and be a reasoning model
    if !thinking_enabled || !is_reasoning_model(model_name) {
        return false;
    }

    // Check for common reasoning token patterns
    let text = token_text.to_lowercase();
    text.contains("<think>")
        || text.contains("</think>")
        || text.contains("<reasoning>")
        || text.contains("</reasoning>")
        || text.contains("<thought>")
        || text.contains("</thought>")
}

/// Determine the finish reason based on output
#[allow(dead_code)]
fn determine_finish_reason(choice_data: &ChatChoiceData, original_reason: Option<&str>) -> String {
    if choice_data.has_tool_calls() {
        "tool_calls".to_string()
    } else {
        original_reason.unwrap_or("stop").to_string()
    }
}

/// Check token length using lock-free engine accessors.
///
/// Uses `model.tokenizer()` and `model.get_available_kv_tokens()`
/// without requiring lock acquisition.
fn check_length(
    request: &ChatCompletionRequest,
    prompt: String,
    data: &OpenAIServerData,
) -> Result<(Vec<u32>, usize), APIError> {
    info!(
        model = %request.model,
        prompt_chars = prompt.chars().count(),
        "📏 CHAT: checking prompt length"
    );

    // Get available KV tokens from cache config (no lock needed)
    let available_kv_tokens = data.model.get_available_kv_tokens();

    // Tokenize using the shared tokenizer (no lock needed)
    let token_ids = data
        .model
        .tokenizer()
        .encode(prompt.as_str(), false)
        .map_err(|e| APIError::new(format!("Tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();

    info!(
        model = %request.model,
        prompt_tokens = token_ids.len(),
        available_kv_tokens,
        max_model_len = data.pipeline_config.max_model_len,
        "🔢 CHAT: tokenization complete"
    );

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
        info!(
            model = %request.model,
            max_request_tokens = max_valid_request_tokens,
            "✅ CHAT: prompt length accepted"
        );
        Ok((token_ids, max_valid_request_tokens))
    }
}

pub async fn chat_completions_with_data(
    data: Arc<OpenAIServerData>,
    request: ChatCompletionRequest,
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

    let request_id = if let Some(conv_id) = request.conversation_id.clone() {
        conv_id
    } else {
        Uuid::new_v4().to_string()
    };

    info!(
        %request_id,
        model = %request.model,
        stream = request.stream.unwrap_or(false),
        logprobs = ?request.logprobs,
        "🚦 CHAT: begin prompt processing"
    );

    // Build prompt using stateless conversation handler (no lock needed)
    let prompt = match get_gen_prompt(&data, &request) {
        Ok(p) => p,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    // Check length using shared tokenizer (no lock needed)
    let (token_ids, available_tokens): (Vec<u32>, usize) =
        match check_length(&request, prompt, &data) {
            Ok(v) => v,
            Err(e) => return ChatResponder::ValidationError(e),
        };

    let created = get_created_time_secs();

    info!(
        %request_id,
        model = %request.model,
        prompt_tokens = token_ids.len(),
        max_request_tokens = available_tokens,
        "🧮 CHAT: prompt ready with token budget"
    );

    let max_request_tokens = available_tokens;

    // Convert logprobs from Option<bool> to Option<usize>
    let logprobs_count: Option<usize> = request.logprobs.and_then(|enabled| {
        if enabled {
            Some(5) // Default to 5 logprobs when enabled
        } else {
            None
        }
    });

    // Build SamplingParams using the new constructor
    let sampling_params = match SamplingParams::new(
        request.n.unwrap_or(1),
        request.best_of,
        request.presence_penalty.unwrap_or(0.0),
        request.frequency_penalty.unwrap_or(0.0),
        None, // repeat_last_n
        request.temperature,
        request.top_p,
        request.min_p,
        request.top_k,
        request.use_beam_search.unwrap_or(false),
        1.0, // length_penalty
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        request.stop.clone(),
        request
            .stop_token_ids
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|id| id as usize)
            .collect(),
        request.ignore_eos.unwrap_or(false),
        max_request_tokens,
        logprobs_count,
        None, // prompt_logprobs
        request.skip_special_tokens.unwrap_or(true),
        request.thinking,
    ) {
        Ok(params) => params,
        Err(e) => return ChatResponder::ValidationError(e),
    };

    // Create flume channel for ChatResponse (matches Streamer expectation)
    let (response_tx, rx) = flume::unbounded::<ChatResponse>();
    info!(
        "🎯 CORE: sampling_params prepared for request {} - stream={}, max_tokens={}, prompt_tokens={}",
        request_id,
        request.stream.is_some_and(|x| x),
        sampling_params.max_tokens,
        token_ids.len()
    );

    let data_clone = data.clone();
    let request_id_clone = request_id.clone();
    let stream_request = request.stream.is_some_and(|x| x);
    let model_name = request.model.clone();
    let model_name_for_response = model_name.clone(); // Clone for use outside the spawned task
    let thinking_enabled = request.thinking.unwrap_or(false);
    let sync_notify = Arc::new(Notify::new());
    let sync_notify_clone = Arc::clone(&sync_notify);

    // Spawn blocking task to submit work to the engine
    let sampling_params_clone = sampling_params.clone();
    info!(
        "🚀 CORE: Spawning blocking task to submit work to engine - request_id={}",
        request_id_clone
    );
    let _ = tokio::task::spawn_blocking(move || {
        let positions: Vec<usize> = (0..token_ids.len()).collect();
        let max_context_len = data.pipeline_config.max_model_len;

        if stream_request {
            info!(
                "📡 CORE: Submitting STREAMING request to engine - request_id={}",
                request_id
            );
            // Submit streaming request (engine is async, use block_on)
            let rt = tokio::runtime::Handle::current();
            let stream_rx = match rt.block_on(data.model.add_streaming_request(
                request_id.clone(),
                token_ids.clone(),
                positions.clone(),
                sampling_params_clone,
                created,
                max_context_len,
            )) {
                Ok(rx) => {
                    info!(
                        "✅ CORE: Streaming request accepted by engine - request_id={}",
                        request_id
                    );
                    rx
                }
                Err(e) => {
                    error!(
                        "❌ CORE: Streaming request rejected by engine - request_id={}, error={}",
                        request_id, e
                    );
                    let _ = response_tx.send(ChatResponse::ModelError(e.to_string()));
                    return;
                }
            };

            // Bridge streaming tokens to ChatResponse chunks
            // Clone model_name for use in the spawned thread
            let model_name_for_stream = model_name.clone();
            info!(
                "🔗 CORE: Spawning thread to bridge streaming tokens - request_id={}",
                request_id
            );
            std::thread::spawn(move || {
                let mut full_content = String::new();
                let mut full_reasoning = String::new();
                let mut is_first_chunk = true;
                let mut token_count = 0;

                info!(
                    "👂 CORE: Streaming bridge thread started, waiting for tokens - request_id={}",
                    request_id
                );
                while let Ok(result) = stream_rx.recv() {
                    token_count += 1;
                    match result {
                        Ok(token) => {
                            if token_count == 1 {
                                info!(
                                    "🎉 CORE: Received FIRST streaming token - request_id={}",
                                    request_id
                                );
                            }
                            if token_count % 10 == 0 {
                                info!("📊 CORE: Streaming progress - request_id={}, tokens_received={}", request_id, token_count);
                            }
                            // Detect if this token is a reasoning token
                            let token_is_reasoning = is_reasoning_token(
                                &token.text,
                                token.token_id,
                                &model_name_for_stream,
                                thinking_enabled,
                                token.is_reasoning,
                            );

                            // Track content separately for reasoning vs regular content
                            if token_is_reasoning {
                                full_reasoning.push_str(&token.text);
                            } else {
                                full_content.push_str(&token.text);
                            }

                            // Create streaming chunk with appropriate delta type
                            let delta = if token_is_reasoning {
                                // Reasoning token - use reasoning field
                                crate::openai::responses::ChoiceData {
                                    role: if is_first_chunk {
                                        Some("assistant".to_string())
                                    } else {
                                        None
                                    },
                                    content: None,
                                    tool_calls: None,
                                    reasoning: Some(token.text.clone()),
                                }
                            } else {
                                // Regular content token
                                crate::openai::responses::ChoiceData {
                                    role: if is_first_chunk {
                                        Some("assistant".to_string())
                                    } else {
                                        None
                                    },
                                    content: Some(token.text.clone()),
                                    tool_calls: None,
                                    reasoning: None,
                                }
                            };

                            is_first_chunk = false;

                            let chunk = crate::openai::responses::ChatCompletionChunk {
                                id: request_id.clone(),
                                choices: vec![crate::openai::responses::Choice {
                                    delta,
                                    finish_reason: if token.is_finished {
                                        token.finish_reason.clone()
                                    } else {
                                        None
                                    },
                                    index: 0,
                                }],
                                created,
                                model: model_name_for_stream.to_string(),
                                object: "chat.completion.chunk",
                                system_fingerprint: None,
                                conversation_id: None,
                                resource_id: None,
                            };

                            if response_tx.send(ChatResponse::Chunk(chunk)).is_err() {
                                error!("❌ CORE: Failed to send chunk, client disconnected - request_id={}", request_id);
                                break;
                            }

                            if token.is_finished {
                                info!("🏁 CORE: Streaming finished - request_id={}, total_tokens={}, finish_reason={:?}", 
                                    request_id, token_count, token.finish_reason);
                                let _ = response_tx.send(ChatResponse::Done);
                                sync_notify_clone.notify_one();
                                break;
                            }
                        }
                        Err(err) => {
                            error!(
                                "❌ CORE: Streaming error - request_id={}, error={}",
                                request_id, err
                            );
                            let _ = response_tx.send(ChatResponse::ModelError(err));
                            sync_notify_clone.notify_one();
                            break;
                        }
                    }
                }
                info!(
                    "🔚 CORE: Streaming bridge thread exiting - request_id={}, total_tokens={}",
                    request_id, token_count
                );
            });
        } else {
            info!(
                "📝 CORE: Submitting NON-STREAMING (completion) request to engine - request_id={}",
                request_id
            );
            // Submit completion request (engine is async, use block_on)
            let rt = tokio::runtime::Handle::current();
            let response_rx = match rt.block_on(data.model.add_request(
                request_id.clone(),
                token_ids.clone(),
                positions.clone(),
                sampling_params_clone,
                max_context_len,
            )) {
                Ok(rx) => {
                    info!(
                        "✅ CORE: Completion request accepted by engine - request_id={}",
                        request_id
                    );
                    rx
                }
                Err(e) => {
                    error!(
                        "❌ CORE: Completion request rejected by engine - request_id={}, error={}",
                        request_id, e
                    );
                    let _ = response_tx.send(ChatResponse::ModelError(e.to_string()));
                    sync_notify_clone.notify_one();
                    return;
                }
            };

            // Bridge the engine's response channel to ChatResponse
            info!(
                "🔗 CORE: Spawning thread to wait for completion response - request_id={}",
                request_id
            );
            std::thread::spawn(move || {
                use crate::parking_lot::InferenceResult;
                info!(
                    "👂 CORE: Waiting for completion result from engine - request_id={}",
                    request_id
                );
                match response_rx.blocking_recv() {
                    Ok(InferenceResult::Completion { choices, mut usage }) => {
                        info!("✅ CORE: Received completion result - request_id={}, choices={}, tokens={}", 
                            request_id, choices.len(), usage.total_tokens);
                        
                        // Update usage with cached token information
                        data.model.update_usage_with_cache(&mut usage, &request_id);
                        
                        // For non-streaming, we still need to send via the response channel
                        // but we'll handle the final response differently
                        let response = ChatCompletionResponse {
                            id: request_id.clone(),
                            object: "chat.completion",
                            created: usage.created,
                            model: model_name.to_string(),
                            choices,
                            usage,
                            conversation_id: None,
                            resource_id: None,
                            system_fingerprint: Some(data.model.config.system_fingerprint()),
                        };
                        // Store in completion_records for non-streaming retrieval
                        // Always persist completion results for sync retrieval
                        let mut records = data.model.completion_records.write();

                        records.insert(
                            request_id.clone(),
                            (response.choices.clone(), response.usage.clone()),
                        );
                        // Also send as chunk for consistency
                        let chunk = crate::openai::responses::ChatCompletionChunk {
                            id: request_id.clone(),
                            choices: response
                                .choices
                                .iter()
                                .map(|c| crate::openai::responses::Choice {
                                    delta: crate::openai::responses::ChoiceData {
                                        role: Some(c.message.role.clone()),
                                        content: c.message.content.clone(),
                                        tool_calls: None,
                                        reasoning: None,
                                    },
                                    finish_reason: c.finish_reason.clone(),
                                    index: c.index,
                                })
                                .collect(),
                            created: response.usage.created,
                            model: model_name.to_string(),
                            object: "chat.completion.chunk",
                            system_fingerprint: None,
                            conversation_id: None,
                            resource_id: None,
                        };
                        if response_tx.send(ChatResponse::Chunk(chunk)).is_err() {
                            error!(
                                "❌ CORE: Failed to send completion chunk - request_id={}",
                                request_id
                            );
                        }
                        if response_tx.send(ChatResponse::Done).is_err() {
                            error!(
                                "❌ CORE: Failed to send completion done signal - request_id={}",
                                request_id
                            );
                        }
                        info!(
                            "🏁 CORE: Completion response sent successfully - request_id={}",
                            request_id
                        );
                    }
                    Ok(InferenceResult::Error { message }) => {
                        error!(
                            "❌ CORE: Received error result - request_id={}, error={}",
                            request_id, message
                        );
                        let _ = response_tx.send(ChatResponse::ModelError(message));
                    }
                    Ok(InferenceResult::Streaming { .. }) => {
                        error!("❌ CORE: Unexpected streaming result for completion request - request_id={}", request_id);
                        let _ = response_tx.send(ChatResponse::ModelError(
                            "Unexpected streaming result for completion request".to_string(),
                        ));
                    }
                    Err(e) => {
                        error!(
                            "❌ CORE: Failed to receive from engine - request_id={}, error={}",
                            request_id, e
                        );
                        let _ = response_tx.send(ChatResponse::ModelError(e.to_string()));
                    }
                }
                info!("📢 CORE: Notifying sync waiter - request_id={}", request_id);
                sync_notify_clone.notify_one();
            });
        }
    });

    if stream_request {
        info!(
            "🌊 CORE: Returning SSE streamer for request {}",
            request_id_clone
        );
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
        // Wait until current response finished
        info!(
            "⏳ CORE: Waiting for sync response for request {}",
            request_id_clone
        );
        sync_notify.notified().await;
        info!(
            "✅ CORE: Sync notify received for request {}",
            request_id_clone
        );

        // Get response from completion_records
        let records = data_clone.model.completion_records.read();
        if let Some((choices, usage)) = records.get(&request_id_clone) {
            ChatResponder::Completion(ChatCompletionResponse {
                id: request_id_clone,
                object: "chat.completion",
                created,
                model: model_name_for_response,
                choices: choices.clone(),
                usage: usage.clone(),
                conversation_id: None,
                resource_id: None,
                system_fingerprint: Some(data_clone.model.config.system_fingerprint()),
            })
        } else {
            ChatResponder::ModelError(APIError::new(format!(
                "Unable to generate response for request {}",
                request_id_clone
            )))
        }
    }
}
