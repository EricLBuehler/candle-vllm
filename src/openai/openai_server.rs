use super::requests::ChatCompletionRequest;
use super::requests::Messages;
use super::responses::{APIError, ChatCompletionResponse, ChatResponder};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::{Streamer, StreamingStatus};
use super::OpenAIServerData;
use axum::response::sse::KeepAlive;
use axum::{
    extract::{Json, State},
    response::Sse,
};
use flume;
use std::env;
use std::sync::Arc;
use std::time::SystemTime;
use tokenizers::Encoding;
use tokio::time::Duration;
use uuid::Uuid;
// fn verify_model(data: &OpenAIServerData<'_>, model_name: &String) -> Result<(), APIError> {
//     let current_name = {
//         let model = data.model.lock().unwrap();
//         model.get_pipeline().name().to_string()
//     };
//     if &current_name != model_name {
//         Err(APIError::new(format!(
//             "Model name `{model_name}` is invalid."
//         )))
//     } else {
//         Ok(())
//     }
// }

// Get prompt, roles
async fn get_gen_prompt(
    data: &OpenAIServerData,
    request: &ChatCompletionRequest,
) -> Result<String, APIError> {
    let mut model = data.model.lock().await;
    let conversation = model
        .get_mut_pipeline()
        .get_conversation(data.record_conversation);

    match &request.messages {
        Messages::Literal(msg) => {
            return Ok(msg.clone());
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
                    conversation.set_system_message(content);
                } else {
                    conversation.append_message(role.to_string(), content)
                }
            }
        }
    }

    Ok(conversation.get_prompt())
}

async fn check_length(
    request: &ChatCompletionRequest,
    prompt: String,
    data: &OpenAIServerData,
) -> Result<Encoding, APIError> {
    let token_ids = {
        let model = data.model.lock().await;
        model
            .get_pipeline()
            .tokenizer()
            .tokenizer()
            .encode(prompt, false)
            .map_err(APIError::from)?
    };

    let max_gen_tokens = request
        .max_tokens
        .unwrap_or(data.pipeline_config.default_max_tokens);

    if token_ids.len() + max_gen_tokens > data.pipeline_config.max_model_len {
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
    } else {
        Ok(token_ids)
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
    // let model_name = &request.model;
    // let res = verify_model(&data, model_name);
    // if res.is_err() {
    //     return Either::Left(Err(res.err().unwrap()));
    // }

    if request.logit_bias.as_ref().is_some()
        && request.logit_bias.as_ref().is_some_and(|x| !x.is_empty())
    {
        return ChatResponder::ValidationError(APIError::new_str(
            "`logit_bias` is not currently supported.",
        ));
    }

    let prompt = get_gen_prompt(&data, &request).await;
    if prompt.is_err() {
        return ChatResponder::ValidationError(prompt.err().unwrap());
    }
    let prompt = prompt.unwrap();

    let token_ids = check_length(&request, prompt.clone(), &data).await;
    if token_ids.is_err() {
        return ChatResponder::ValidationError(token_ids.err().unwrap());
    }
    let token_ids: Encoding = token_ids.unwrap();

    println!("\n\n\nPrompt {:?}", prompt);

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let sampling_params = SamplingParams::new(
        request.n.unwrap_or(1),
        request.best_of,
        request.presence_penalty.unwrap_or(0.0),
        request.frequency_penalty.unwrap_or(0.0),
        request
            .repetition_penalty
            .unwrap_or(data.pipeline_config.penalty),
        request
            .temperature
            .unwrap_or(data.pipeline_config.temperature),
        request.top_p.unwrap_or(1.0),
        request.top_k.unwrap_or(-1),
        request.use_beam_search.unwrap_or(false),
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        request.stop.clone(),
        request.stop_token_ids.clone().unwrap_or_default(),
        request.ignore_eos.unwrap_or(false),
        request
            .max_tokens
            .unwrap_or(data.pipeline_config.default_max_tokens),
        None,
        None,
        request.skip_special_tokens.unwrap_or(true),
    );
    if sampling_params.is_err() {
        return ChatResponder::ValidationError(sampling_params.err().unwrap());
    }
    let sampling_params = sampling_params.unwrap();

    let (response_tx, rx) = flume::unbounded();
    // println!("{:?}", sampling_params);

    if request.stream.is_some_and(|x| x) {
        let _ = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                {
                    //send completion request to inference engine
                    let mut model = data.model.lock().await;
                    model.add_request(
                        token_ids,
                        request_id.clone(),
                        SystemTime::now(),
                        sampling_params,
                        request.logprobs.unwrap_or(false),
                        Some(response_tx),
                    );
                    model.notify.notify_one();
                }
            });
        });
        ChatResponder::Streamer(
            Sse::new(Streamer {
                rx,
                status: StreamingStatus::Uninitilized,
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
        //send completion request to inference engine
        let mut model = data.model.lock().await;
        model.add_request(
            token_ids,
            request_id.clone(),
            SystemTime::now(),
            sampling_params,
            request.logprobs.unwrap_or(false),
            Some(response_tx),
        );
        model.notify.notify_one();
        // wait until current response finished
        data.finish_notify.notified().await;
        let model = data.model.lock().await;
        if !model.completion_records.contains_key(&request_id) {
            return ChatResponder::ModelError(APIError::from(format!(
                "Unable to generate response for request {}",
                request_id
            )));
        }

        let choices = &model.completion_records[&request_id].0;
        let usage = &model.completion_records[&request_id].1;

        ChatResponder::Completion(ChatCompletionResponse {
            id: request_id,
            choices: choices.to_vec(),
            created: usage.created,
            model: request.model.clone(),
            object: "chat.completion",
            usage: usage.clone(),
        })
    }
}
