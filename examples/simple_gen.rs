use candle_vllm::api::{EngineBuilder, ModelRepo};
use candle_vllm::openai::requests::ChatCompletionRequest;
use candle_vllm::openai::requests::Messages;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // let builder = EngineBuilder::new(ModelRepo::ModelPath("/home/GLM-4-9B-0414-Q4_K_M.gguf"))
    let builder =
        EngineBuilder::new(ModelRepo::ModelID(("Qwen/Qwen3-0.6B", None))).with_kvcache_mem_gpu(512); // Small mem for testing

    let engine = builder.build()?;

    let mut messages = HashMap::new();
    messages.insert("role".to_string(), "user".to_string());
    messages.insert("content".to_string(), "Talk about China.".to_string());

    let request = ChatCompletionRequest {
        messages: Messages::Map(vec![messages]),
        temperature: Some(0.7),
        top_p: Some(0.95),
        n: Some(1),
        max_tokens: Some(100),
        stream: Some(false),
        ..Default::default()
    };

    let response = engine.generate(vec![request])?;
    println!("Response: {:?}", response);

    engine.shutdown();

    Ok(())
}
