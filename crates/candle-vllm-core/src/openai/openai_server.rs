use super::requests::{ChatCompletionRequest, Messages, ToolCall};
use super::responses::{APIError, ChatChoiceData, ChatCompletionResponse, ChatResponder};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::{Streamer, StreamingStatus};
use super::tool_parser::{get_tool_parser, ParsedOutput};
use super::OpenAIServerData;
use axum::extract::State;
use axum::response::sse::KeepAlive;
use axum::{response::Sse, Json};
use flume;
use std::env;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Notify;
use tokio::time::Duration;
use tracing::debug;
use uuid::Uuid;
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
    chat_completions_with_data(data, request.0).await
}

// Get prompt, roles - now with tool support
async fn get_gen_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
) -> Result<String, APIError> {
    let mut model = data.model.write();
    let pipeline = model
        .get_mut_pipeline(0)
        .ok_or(APIError::new("Missing pipeline".to_string()))?;
    let conversation = pipeline.0.get_conversation(data.record_conversation);

    // Set tools if provided
    if let Some(ref tools) = request.tools {
        conversation.set_tools(Some(tools.clone()));
    } else {
        conversation.set_tools(None);
    }

    match &request.messages {
        Messages::Literal(msg) => {
            return Ok(msg.clone());
        }
        Messages::Chat(messages) => {
            // New Chat format with full tool support
            for message in messages {
                if message.role == "system" {
                    if let Some(ref content) = message.content {
                        tracing::info!("system prompt found: {}", content);
                        conversation.set_system_message(Some(content.clone()));
                    }
                }
                conversation.append_message_ext(
                    message.role.clone(),
                    message.content.clone(),
                    message.tool_calls.clone(),
                    message.tool_call_id.clone(),
                    message.name.clone(),
                );
            }
        }
        Messages::Map(messages) => {
            // Legacy format - convert to simple messages
            for message in messages {
                let role = message
                    .get("role")
                    .ok_or(APIError::new("Message key `role` not found.".to_string()))?;
                let content = message.get("content").cloned().unwrap_or_default();

                if role == "system" {
                    tracing::info!("system prompt found: {}", content);
                    conversation.set_system_message(Some(content.clone()));
                }
                conversation.append_message(role.to_string(), content)
            }
        }
    }

    Ok(conversation.get_prompt(request.thinking.unwrap_or(false)))
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

/// Determine the finish reason based on output
#[allow(dead_code)]
fn determine_finish_reason(choice_data: &ChatChoiceData, original_reason: Option<&str>) -> String {
    if choice_data.has_tool_calls() {
        "tool_calls".to_string()
    } else {
        original_reason.unwrap_or("stop").to_string()
    }
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

    let prompt = match get_gen_prompt(&data, &request).await {
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

    // Store conversation_id and resource_id for this request
    {
        let model = data.model.write();
        model.request_metadata.write().insert(
            request_id.clone(),
            (request.conversation_id.clone(), request.resource_id.clone()),
        );
    }

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
    let sampling_params = match SamplingParams::new(
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

    let (response_tx, rx) = flume::unbounded();
    tracing::info!("{:?}", sampling_params);

    let data_clone = data.clone();
    let request_id_clone = request_id.clone();
    let stream_request = request.stream.is_some_and(|x| x);
    let model_name = request.model.clone();
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
        let model = data_clone.model.read();
        if !model.completion_records.contains_key(&request_id_clone) {
            return ChatResponder::ModelError(APIError::from(format!(
                "Unable to generate response for request {request_id_clone}"
            )));
        }

        let choices = &model.completion_records[&request_id_clone].0;
        let usage = &model.completion_records[&request_id_clone].1;
        
        // Retrieve conversation_id and resource_id from metadata
        let (conversation_id, resource_id) = model
            .request_metadata
            .read()
            .get(&request_id_clone)
            .cloned()
            .unwrap_or((None, None));

        ChatResponder::Completion(ChatCompletionResponse {
            id: request_id_clone,
            choices: choices.to_vec(),
            created: usage.created,
            model: model_name,
            object: "chat.completion",
            usage: usage.clone(),
            conversation_id,
            resource_id,
        })
    }
}
