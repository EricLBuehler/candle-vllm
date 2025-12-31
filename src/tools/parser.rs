// src/tools/parser.rs
//! Tool call parsing from model output
//!
//! Supports multiple formats used by different models.

use super::ToolCall;
use regex::Regex;
use serde_json::Value;

/// Parser for extracting tool calls from model output text
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ToolParser {
    /// Regex patterns for different formats
    patterns: Vec<(String, Regex)>,
}

impl Default for ToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolParser {
    /// Create a new parser with default patterns
    pub fn new() -> Self {
        let patterns = vec![
            // Qwen format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
            (
                "qwen".to_string(),
                Regex::new(r#"<tool_call>\s*(\{[^}]+\})\s*</tool_call>"#).unwrap()
            ),
            // Generic JSON object with name and arguments
            (
                "json".to_string(),
                Regex::new(r#"\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\}|\[[^\]]*\]|"[^"]*"|\d+|true|false|null)\s*\}"#).unwrap()
            ),
            // Function call format in code blocks
            (
                "func".to_string(),
                Regex::new(r#"```(?:json)?\s*\{[^}]*"name"[^}]*\}\s*```"#).unwrap()
            ),
        ];
        Self { patterns }
    }

    /// Parse tool calls from model output
    /// Only parses tool calls from the final answer (after reasoning end markers)
    pub fn parse(&self, text: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();
        let mut call_id = 0;

        // Extract only the final answer portion (after reasoning ends)
        let final_answer = Self::extract_final_answer(text);

        // Try Qwen format first
        if let Some(qwen_calls) = self.parse_qwen_format(&final_answer, &mut call_id) {
            calls.extend(qwen_calls);
        }

        // Try generic JSON format
        if calls.is_empty() {
            if let Some(json_calls) = self.parse_json_format(&final_answer, &mut call_id) {
                calls.extend(json_calls);
            }
        }

        // Try code block format
        if calls.is_empty() {
            if let Some(block_calls) = self.parse_code_block_format(&final_answer, &mut call_id) {
                calls.extend(block_calls);
            }
        }

        calls
    }

    /// Extract the final answer portion from model output, skipping reasoning blocks.
    /// Returns the text after reasoning end markers, or the full text if no reasoning found.
    pub fn extract_final_answer(text: &str) -> String {
        // Reasoning end markers used by different models
        let reasoning_end_markers = [
            "</think>",     // Common thinking format
            "</thought>",   // Alternative thinking format
            "<|/think|>",   // Qwen-style special tokens
            "[/THINK]",     // Bracket format
            "</reasoning>", // Reasoning tag
        ];

        // Find the last occurrence of any reasoning end marker
        let mut last_end_pos = None;
        for marker in &reasoning_end_markers {
            if let Some(pos) = text.rfind(marker) {
                let end_pos = pos + marker.len();
                if last_end_pos.is_none() || end_pos > last_end_pos.unwrap() {
                    last_end_pos = Some(end_pos);
                }
            }
        }

        // Return content after the last reasoning end marker, or full text if none found
        if let Some(pos) = last_end_pos {
            text[pos..].to_string()
        } else {
            text.to_string()
        }
    }

    /// Parse XML-wrapped tool call formats (<tool_call>)
    fn parse_qwen_format(&self, text: &str, call_id: &mut usize) -> Option<Vec<ToolCall>> {
        let mut calls = Vec::new();

        // Try both <tool_call> formats
        // Use a more flexible regex that allows for missing closing > if at end of string
        // This handles cases where generation stops exactly on </tool_call
        // Pattern: <tool_call> ... </tool_call>?
        // Note: We use a single regex with optional > to avoid duplicate matches
        let pattern = r"(?s)<tool_call>\s*(.*?)\s*</tool_call>?";

        if let Ok(re) = Regex::new(pattern) {
            for cap in re.captures_iter(text) {
                if let Some(json_str) = cap.get(1) {
                    // Validate that whatever we captured looks like JSON before parsing
                    // This prevents matching random text if </tool_call> is missing entirely and we match to end of string
                    let trimmed = json_str.as_str().trim();
                    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
                        continue;
                    }

                    if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                        if let Some(call) = self.value_to_tool_call(&parsed, call_id) {
                            calls.push(call);
                        }
                    }
                }
            }
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    /// Parse generic JSON format with name and arguments
    fn parse_json_format(&self, text: &str, call_id: &mut usize) -> Option<Vec<ToolCall>> {
        // Try to find JSON objects that look like tool calls
        let mut calls = Vec::new();

        // Simple approach: try to parse the entire text as JSON first
        if let Ok(parsed) = serde_json::from_str::<Value>(text.trim()) {
            if let Some(call) = self.value_to_tool_call(&parsed, call_id) {
                return Some(vec![call]);
            }
        }

        // Look for JSON blocks in the text
        let mut depth = 0;
        let mut start = None;

        for (i, c) in text.char_indices() {
            match c {
                '{' => {
                    if depth == 0 {
                        start = Some(i);
                    }
                    depth += 1;
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s) = start {
                            let json_str = &text[s..=i];
                            if let Ok(parsed) = serde_json::from_str::<Value>(json_str) {
                                if let Some(call) = self.value_to_tool_call(&parsed, call_id) {
                                    calls.push(call);
                                }
                            }
                        }
                        start = None;
                    }
                }
                _ => {}
            }
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    /// Parse tool calls from markdown code blocks
    fn parse_code_block_format(&self, text: &str, call_id: &mut usize) -> Option<Vec<ToolCall>> {
        let re = Regex::new(r"```(?:json)?\s*([\s\S]*?)\s*```").ok()?;
        let mut calls = Vec::new();

        for cap in re.captures_iter(text) {
            if let Some(content) = cap.get(1) {
                if let Ok(parsed) = serde_json::from_str::<Value>(content.as_str().trim()) {
                    if let Some(call) = self.value_to_tool_call(&parsed, call_id) {
                        calls.push(call);
                    }
                }
            }
        }

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    /// Convert a JSON Value to a ToolCall if it has the right structure
    fn value_to_tool_call(&self, value: &Value, call_id: &mut usize) -> Option<ToolCall> {
        let name = value.get("name")?.as_str()?;
        let arguments = value.get("arguments")?;

        let args_str = if arguments.is_string() {
            arguments.as_str().unwrap().to_string()
        } else {
            serde_json::to_string(arguments).ok()?
        };

        *call_id += 1;
        Some(ToolCall::new(
            format!("call_{}", call_id),
            name.to_string(),
            args_str,
        ))
    }

    /// Check if text contains any tool calls (only explicit XML tags in final answer)
    /// Note: Raw JSON patterns are NOT checked to avoid false positives in reasoning
    pub fn has_tool_calls(&self, text: &str) -> bool {
        let final_answer = Self::extract_final_answer(text);
        // Only check for explicit XML-wrapped tool calls
        final_answer.contains("<tool_call>")
    }

    /// Check if text contains a complete, parseable tool call
    /// Returns true only if the tool call has valid structure with both tags and valid JSON
    pub fn has_complete_tool_call(&self, text: &str) -> bool {
        let final_answer = Self::extract_final_answer(text);

        // Must have both opening and closing tags
        if !final_answer.contains("<tool_call>") || !final_answer.contains("</tool_call>") {
            return false;
        }

        // Try to parse - if successful, it's complete
        !self.parse(&final_answer).is_empty()
    }

    /// Check if text could be a partial tool call tag (for lookback detection)
    /// Used to detect when we might be in the middle of receiving "<tool_call>"
    pub fn could_be_partial_tag(text: &str) -> bool {
        const TAG: &str = "<tool_call>";
        // Check if end of text matches any prefix of tag (length 1 to len-1)
        for i in 1..TAG.len() {
            if text.ends_with(&TAG[..i]) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_qwen_format() {
        let parser = ToolParser::new();
        let text = r#"I'll help you with the weather.
<tool_call>
{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}
</tool_call>"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Tokyo"));
    }

    #[test]
    fn test_parse_json_format() {
        let parser = ToolParser::new();
        let text = r#"{"name": "calculate", "arguments": {"expression": "2+2"}}"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calculate");
    }

    #[test]
    fn test_parse_code_block() {
        let parser = ToolParser::new();
        let text = r#"Let me search for that:

```json
{"name": "search", "arguments": {"query": "rust programming"}}
```"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_multiple_tool_calls() {
        let parser = ToolParser::new();
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "London"}}
</tool_call>"#;

        let calls = parser.parse(text);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_has_tool_calls() {
        let parser = ToolParser::new();

        assert!(parser.has_tool_calls("<tool_call>{}</tool_call>"));
        // Note: has_tool_calls only checks for XML tags now
        assert!(!parser.has_tool_calls(r#"{"name": "foo", "arguments": {}}"#));
        assert!(!parser.has_tool_calls("Just a normal response"));
    }

    #[test]
    fn test_partial_and_complete_tags() {
        let parser = ToolParser::new();

        // Test complete tool call check
        assert!(parser
            .has_complete_tool_call("<tool_call>{\"name\":\"test\",\"arguments\":{}}</tool_call>"));
        assert!(!parser.has_complete_tool_call("<tool_call>{\"name\":\"test\"}")); // Missing closing
        assert!(!parser.has_complete_tool_call("{\"name\":\"test\"}</tool_call>")); // Missing opening

        // Test partial tag detection
        assert!(ToolParser::could_be_partial_tag("output <"));
        assert!(ToolParser::could_be_partial_tag("output <tool"));
        assert!(ToolParser::could_be_partial_tag("output <tool_call"));
        assert!(!ToolParser::could_be_partial_tag("output <tool_call>")); // Complete tag is not partial
        assert!(!ToolParser::could_be_partial_tag("output other"));
    }

    #[test]
    fn test_optional_closing_tag() {
        let parser = ToolParser::new();
        // Test parsing with missing closing tag (common in streaming)
        let text = r#"<tool_call>{"name": "test", "arguments": "args"}"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "test");
    }
}
