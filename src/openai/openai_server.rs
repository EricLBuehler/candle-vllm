use super::conversation::SeperatorStyle;
use super::responses::APIError;
use super::{conversation::Conversation, requests::ChatCompletionRequest};
use actix_web::{get, web};

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
    prompt: Option<&String>,
) -> Result<Vec<isize>, APIError> {
    todo!()
}

#[get("/v1/chat/completions")]
async fn chat_completions(request: web::Json<ChatCompletionRequest>) -> Result<String, APIError> {
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
    let token_ids = check_length(&request, Some(&prompt));

    Ok(prompt)
}
