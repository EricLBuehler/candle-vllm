use std::sync::{Arc, Mutex};

use actix_web::web::Data;
use actix_web::{App, HttpServer};
use candle_core::{DType, Device};
use candle_vllm::openai::openai_server::chat_completions;
use candle_vllm::openai::responses::APIError;
use candle_vllm::openai::OpenAIServerData;
use candle_vllm::{get_model_loader, ModelSelected};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Huggingface token environment variable (optional)
    #[arg(long)]
    hf_token: Option<String>,

    /// Port to serve on (localhost:port)
    #[arg(long)]
    port: u16,

    #[clap(subcommand)]
    command: ModelSelected,
}

#[actix_web::main]
async fn main() -> Result<(), APIError> {
    let args = Args::parse();

    let (loader, model_id) = get_model_loader(args.command);
    let paths = loader.download_model(model_id, None, args.hf_token)?;
    let model = loader.load_model(paths, DType::F16, Device::Cpu)?;

    let server_data = OpenAIServerData {
        pipeline_config: model.1,
        model: Arc::new(Mutex::new(model.0)),
        device: Device::Cpu,
    };

    println!("Starting server...");

    HttpServer::new(move || {
        App::new()
            .service(chat_completions)
            .app_data(Data::new(server_data.clone()))
    })
    .bind(("127.0.0.1", args.port))
    .map_err(|e| APIError::new(e.to_string()))?
    .run()
    .await
    .map_err(|e| APIError::new(e.to_string()))?;

    Ok(())
}
