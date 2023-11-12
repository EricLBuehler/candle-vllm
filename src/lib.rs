use openai::pipelines::{llama::LlamaLoader, ModelLoader};

const SERVED_MODELS: [&str; 1] = ["llama"];

pub fn get_model_loader<'a>(selected_model: &String) -> (Box<dyn ModelLoader<'a>>, String) {
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

pub mod openai;
