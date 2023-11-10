use std::collections::HashMap;
use std::io::Result;
use std::path::Path;
use std::sync::Mutex;

use actix_web::web::Data;
use actix_web::{http::header::ContentType, test, App};
use candle_core::{DType, Device};
use candle_vllm::openai::openai_server::chat_completions;
use candle_vllm::openai::pipelines::llama::LlamaLoader;
use candle_vllm::openai::pipelines::ModelLoader;
use candle_vllm::openai::requests::Messages;
use candle_vllm::openai::{self, OpenAIServerData};
use clap::Parser;

const SERVED_MODELS: [&str; 1] = ["llama"];

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Select model to serve.
    #[arg(long)]
    selected_model: String,

    /// Huggingface token (optional)
    #[arg(long)]
    hf_token: Option<String>,
}

fn get_model_loader<'a, P: AsRef<Path>>(
    selected_model: &String,
) -> (Box<dyn ModelLoader<'a, P>>, String) {
    if !SERVED_MODELS.contains(&selected_model.as_str()) {
        panic!("Model {} is not supported", selected_model);
    }
    match selected_model.as_str() {
        "llama" => (
            Box::new(LlamaLoader),
            "meta-llama/Llama-2-7b-hf".to_string(),
        ),
        _ => {
            unreachable!();
        }
    }
}

#[actix_web::main]
async fn main() -> Result<()> {
    /*HttpServer::new(|| App::new().service(candle_vllm::openai::openai_server::chat_completions))
    .bind(("127.0.0.1", 8000))?
    .run()
    .await*/

    let args = Args::parse();

    let (loader, model_id) = get_model_loader(&args.selected_model);
    let paths = loader
        .download_model(model_id, None, args.hf_token)
        .unwrap();
    let model = loader.load_model(paths, DType::F16, Device::Cpu).unwrap();

    let server_data = OpenAIServerData {
        pipeline_config: model.1,
        model: Mutex::new(model.0),
        device: Device::Cpu,
    };

    let app = test::init_service(
        App::new()
            .service(chat_completions)
            .app_data(Data::new(server_data)),
    )
    .await;

    let mut system = HashMap::new();
    system.insert("role".to_string(), "system".to_string());
    system.insert(
        "content".to_string(),
        "You are a talented author who specializes in writing poems.".to_string(),
    );

    let mut user = HashMap::new();
    user.insert("role".to_string(), "user".to_string());
    user.insert(
        "content".to_string(),
        "Please write me a poem about why Rust is a great programming language:".to_string(),
    );

    let req = test::TestRequest::with_uri("/v1/chat/completions")
        .insert_header(ContentType::json())
        .set_json(openai::requests::ChatCompletionRequest {
            model: "llama".to_string(),
            messages: Messages::Map(vec![system, user]),
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
