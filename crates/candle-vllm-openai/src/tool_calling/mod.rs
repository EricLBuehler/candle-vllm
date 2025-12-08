//! Tool call parsing for different model formats.
//!
//! This module provides parsers that extract tool calls from model outputs.
//! Different models have different formats for tool calls, so we provide
//! model-specific parsers that implement a common trait.

pub mod parser;

// Re-export commonly used types
pub use parser::{
    get_tool_parser, get_tool_parser_by_name, AutoToolParser, JsonToolParser, LlamaToolParser,
    MistralToolParser, ParsedOutput, ParsedToolCall, QwenToolParser, ToolCallParser,
};
