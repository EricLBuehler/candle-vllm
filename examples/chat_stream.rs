use std::error::Error;
use std::io::{stdout, Write};

use async_openai::types::ChatCompletionRequestUserMessageArgs;
use async_openai::{types::CreateChatCompletionRequestArgs, Client, config::OpenAIConfig};
use futures::StreamExt;

// macOS: cargo run --example chat_stream --features metal

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let api_base = "http://localhost:2000/v1";
    let api_key = "EMPTY";

    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(api_base),
    );

    let request = CreateChatCompletionRequestArgs::default()
        .model("")
        .max_tokens(512u32)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("Who are you?")
            .build()?
            .into()])
        .build()?;

    let mut stream = client.chat().create_stream(request).await?;

    let mut lock = stdout().lock();
    while let Some(result) = stream.next().await {
        match result {
            Ok(response) => {
                response.choices.iter().for_each(|chat_choice| {
                    if let Some(ref content) = chat_choice.delta.content {
                        write!(lock, "{}", content).unwrap();
                    }
                });
            }
            Err(err) => {
                writeln!(lock, "error: {err}").unwrap();
            }
        }
        stdout().flush()?;
    }

    Ok(())
}
