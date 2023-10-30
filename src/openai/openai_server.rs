use super::conversation::SeperatorStyle;
use super::responses::APIError;
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::OpenAIServerData;
use super::{conversation::Conversation, requests::ChatCompletionRequest};
use actix_web::{get, web};
use tokenizers::Encoding;
use uuid::Uuid;

fn verify_model(model_name: &String) -> Result<(), APIError> {
    match model_name.as_str() {
        "llama" => Ok(()),
        _ => Err(APIError::new(format!(
            "Model name `{model_name}` is invalid."
        ))),
    }
}

fn get_conversational_template(model_name: &str) -> Conversation {
    match model_name {
        "llama" => Conversation::new(
            "llama-2".to_string(),
            "[INST] <<SYS>>\n{}\n<</SYS>>\n\n".to_string(),
            Vec::default(),
            0,
            SeperatorStyle::Llama2,
            "".to_string(),
            Vec::default(),
            ("[INST]".to_string(), "[/INST]".to_string()),
            " ".to_string(),
            Some(" </s></s>".to_string()),
        ),
        _ => unreachable!(),
    }
}

async fn get_gen_prompt(request: &web::Json<ChatCompletionRequest>) -> Result<String, APIError> {
    let mut conversation = get_conversational_template(&request.model);

    //assume messages are not String
    for message in &request.messages {
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
        } else if role == "user" {
            conversation.append_message(conversation.get_roles().0.clone(), content)
        } else if role == "assistant" {
            conversation.append_message(conversation.get_roles().1.clone(), content)
        } else {
            return Err(APIError::new(format!("Unknown role: {role}")));
        }
    }

    conversation.append_none_message(conversation.get_roles().1.clone());

    Ok(conversation.get_prompt())
}

fn check_length(
    request: &web::Json<ChatCompletionRequest>,
    prompt: String,
    data: &OpenAIServerData<'_>,
) -> Result<Encoding, APIError> {
    let token_ids = data.tokenizer.tokenize(prompt)?;

    let max_tokens = if let Some(max_toks) = request.max_tokens {
        max_toks
    } else {
        data.pipeline_config.max_model_len - token_ids.len()
    };

    if token_ids.len() + max_tokens > data.pipeline_config.max_model_len {
        Err(APIError::new(format!(
            "This model's maximum context length is {} tokens. \
            However, you requested {} tokens ({} in the messages, \
            {} in the completion). Please reduce the length of the \
            messages or completion.",
            data.pipeline_config.max_model_len,
            max_tokens + token_ids.len(),
            token_ids.len(),
            max_tokens
        )))
    } else {
        Ok(token_ids)
    }
}

#[get("/v1/chat/completions")]
async fn chat_completions(
    data: web::Data<OpenAIServerData<'_>>,
    request: web::Json<ChatCompletionRequest>,
) -> Result<String, APIError> {
    let model_name = &request.model;
    verify_model(model_name)?;

    if request.logit_bias.as_ref().is_some()
        && request.logit_bias.as_ref().is_some_and(|x| !x.is_empty())
    {
        return Err(APIError::new_str(
            "`logit_bias` is not currently supported.",
        ));
    }

    let prompt = get_gen_prompt(&request).await?;
    let token_ids = check_length(&request, prompt, &data)?;

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let sampling_params = SamplingParams::new(
        request.n.unwrap_or(1),
        request.best_of,
        request.presence_penalty.unwrap_or(0.0),
        request.frequency_penalty.unwrap_or(0.0),
        1.0,
        request.temperature.unwrap_or(0.7),
        request.top_p.unwrap_or(1.0),
        request.top_k.unwrap_or(-1),
        request.use_beam_search.unwrap_or(false),
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        request.stop.clone().unwrap_or(vec![]),
        request.stop_token_ids.clone().unwrap_or(vec![]),
        request.ignore_eos.unwrap_or(false),
        request.max_tokens.unwrap_or(16),
        None,
        None,
        request.skip_special_tokens.unwrap_or(true),
    )?;

    Ok("Done".to_string())
}
