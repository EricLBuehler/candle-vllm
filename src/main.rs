use std::io::Result;
use std::sync::Mutex;

use actix_web::web::Data;
use actix_web::{http::header::ContentType, test, App};
use candle_core::{DType, Device};
use candle_vllm::openai::models::llama::{LlamaPipeline, LlamaSpecifcConfig};
use candle_vllm::openai::openai_server::chat_completions;
use candle_vllm::openai::{self, OpenAIServerData};

#[actix_web::main]
async fn main() -> Result<()> {
    /*HttpServer::new(|| App::new().service(candle_vllm::openai::openai_server::chat_completions))
    .bind(("127.0.0.1", 8000))?
    .run()
    .await*/

    let paths =
        LlamaPipeline::download_model("meta-llama/Llama-2-7b-hf".to_string(), None, ".hf_token")
            .unwrap();
    let model = LlamaPipeline::new_default(
        paths,
        LlamaSpecifcConfig::default(),
        DType::F16,
        Device::Cpu,
    )
    .unwrap();

    let server_data = OpenAIServerData {
        pipeline_config: model.1,
        model: Mutex::new(Box::new(model.0)),
        device: Device::Cpu,
    };

    let app = test::init_service(
        App::new()
            .service(chat_completions)
            .app_data(Data::new(server_data)),
    )
    .await;
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
    println!("{:?}", resp.status());
    println!("{:?}", resp.into_body());
    Ok(())
}
