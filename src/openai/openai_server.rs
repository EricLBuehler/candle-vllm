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

    Ok("Success".to_string())
}
