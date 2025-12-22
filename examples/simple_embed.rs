use candle_vllm::api::{EngineBuilder, ModelRepo};
use candle_vllm::openai::requests::{EmbeddingInput, EmbeddingRequest};
use std::env;

fn main() -> anyhow::Result<()> {
    // Basic setup similar to simple_gen
    // For embedding, we can use a model that supports it, or force a model (like bert or just llama) to output hidden states.
    // Assuming Llama model path specific to user environment, or defaults.
    // For this example, we reuse the path from simple_gen or use CLI arg.

    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "/home/Meta-Llama-3.1-8B-Instruct" // Default as seen in simple_gen.rs
    };

    println!("Loading model from {}", model_path);
    let builder = EngineBuilder::new(ModelRepo::ModelPath(Box::leak(
        model_path.to_string().into_boxed_str(),
    )))
    .with_kvcache_mem_gpu(512);

    let engine = builder.build()?;

    let input = "Hello, world!";
    println!("Embedding input: {}", input);

    let request = EmbeddingRequest {
        model: Some("default".to_string()),
        input: EmbeddingInput::String(input.to_string()),
        encoding_format: Default::default(),
        embedding_type: Default::default(),
    };

    let response = engine.embed(request)?;
    println!("Response object: {}", response.object);
    println!("Model: {}", response.model);
    if let Some(data) = response.data.first() {
        match &data.embedding {
            candle_vllm::openai::responses::EmbeddingOutput::Vector(vec) => {
                println!("Embedding vector length: {}", vec.len());
                println!("First 5 elements: {:?}", &vec[..vec.len().min(5)]);
            }
            _ => println!("Base64 embedding returned"),
        }
    }

    engine.shutdown();

    Ok(())
}
