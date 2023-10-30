use actix_web::{http::header::ContentType, test, App};
use candle_vllm::openai::{self, openai_server::chat_completions};

#[actix_web::test]
async fn chat_completion() {
    let app = test::init_service(App::new().service(chat_completions)).await;
    let req = test::TestRequest::with_uri("/v1/chat/completions")
        .insert_header(ContentType::json())
        .set_json(openai::requests::ChatCompletionRequest {
            model: "llama".to_string(),
            messages: vec![],
            temperature: None,
            top_p: None,
            n: None,
            max_tokens: None,
            stop: None,
            stream: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            top_k: None,
            best_of: None,
            use_beam_search: None,
            skip_special_tokens: None,
            ignore_eos: None,
            stop_token_ids: None,
        })
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}
