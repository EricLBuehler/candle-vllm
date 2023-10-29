use super::requests::ChatCompletionRequest;
use super::responses::APIError;
use actix_web::{get, web};

fn verify_model(model_name: &String) -> Result<(), APIError> {
    match model_name.as_str() {
        "llama" => Ok(()),
        _ => Err(APIError::new(format!(
            "Model name `{model_name}` is invalid."
        ))),
    }
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

    Ok("Success".to_string())
}
