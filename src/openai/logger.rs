//! Chat completion request/response logger for the OpenAI-compatible server.
//! Enable by setting `CANDLE_VLLM_CHAT_LOGGER=true` or `1`.

use super::{requests::ChatCompletionRequest, responses::ChatCompletionResponse};
use crate::openai::responses::ChatCompletionChunk;
use crate::tools::ToolCall;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

pub fn is_logging_enabled() -> bool {
    std::env::var("CANDLE_VLLM_CHAT_LOGGER")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false)
}

pub struct ChatCompletionLogger {
    file_path: String,
}

impl ChatCompletionLogger {
    pub fn new() -> Option<Arc<Self>> {
        if !is_logging_enabled() {
            return None;
        }

        let log_dir = Path::new("log");
        if !log_dir.exists() {
            let _ = fs::create_dir_all(log_dir);
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let secs = now.as_secs();
        let millis = now.subsec_millis();
        let file_path = format!("log/openai_{}_{:03}.log", secs, millis);

        tracing::info!("OpenAI chat logging enabled, writing to: {}", file_path);

        Some(Arc::new(Self { file_path }))
    }

    fn write(&self, content: &str) {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
        {
            let _ = file.write_all(content.as_bytes());
        }
    }

    pub fn log_request(&self, request: &ChatCompletionRequest) {
        if let Ok(json) = serde_json::to_string_pretty(request) {
            self.write(&format!("=== OPENAI REQUEST ===\n{}\n\n", json));
        }
    }

    pub fn log_prompt(&self, prompt: &str) {
        self.write(&format!(
            "=== PROMPT (after chat template) ===\n{}\n\n",
            prompt
        ));
    }

    pub fn log_start_response(&self) {
        self.write("=== OPENAI STREAM RESPONSE ===\n");
    }

    pub fn log_stream_token(&self, token: &str) {
        self.write(token);
    }

    pub fn log_tool_calls(&self, label: &str, tool_calls: &[ToolCall]) {
        if tool_calls.is_empty() {
            return;
        }
        if let Ok(json) = serde_json::to_string_pretty(tool_calls) {
            self.write(&format!(
                "\n\n=== {} TOOL CALLS ({}) ===\n{}\n",
                label.to_uppercase(),
                tool_calls.len(),
                json
            ));
        }
    }

    pub fn log_stream_end(&self, final_chunk: &ChatCompletionChunk) {
        if let Ok(json) = serde_json::to_string_pretty(final_chunk) {
            self.write(&format!("\n\n=== FINAL CHUNK ===\n{}\n", json));
        }
    }

    pub fn log_response(&self, response: &ChatCompletionResponse) {
        if let Ok(json) = serde_json::to_string_pretty(response) {
            self.write(&format!("=== OPENAI RESPONSE ===\n{}\n", json));
        }
    }

    pub fn log_error(&self, error: &str) {
        self.write(&format!("=== ERROR ===\n{}\n", error));
    }
}
