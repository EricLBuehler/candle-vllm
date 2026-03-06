// src/tools/helpers.rs
//! Helper functions for tool call processing.
//!
//! Handles schema mapping, tool call validation, argument repair/normalization,
//! and type coercion. Ported from vllm.rs.

use super::{FunctionCall, Tool, ToolCall};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Build a map of tool names to their parameter schemas
pub fn build_tool_schema_map(tools: &[Tool]) -> HashMap<String, Value> {
    tools
        .iter()
        .map(|tool| (tool.function.name.clone(), tool.function.parameters.clone()))
        .collect()
}

/// Filter tool calls into valid and invalid based on schema validation.
///
/// Valid calls have their arguments parsed, repaired, normalized, and coerced.
/// Invalid calls have unparseable arguments, unknown tool names, or missing required keys.
pub fn filter_tool_calls(
    tool_calls: &[ToolCall],
    schemas: &HashMap<String, Value>,
) -> (Vec<ToolCall>, Vec<ToolCall>) {
    let mut valid = Vec::new();
    let mut invalid = Vec::new();

    for call in tool_calls {
        let args_str = &call.function.arguments;
        let mut parsed_args = match serde_json::from_str::<Value>(args_str) {
            Ok(value) => value,
            Err(e) => {
                match repair_json_arguments(args_str)
                    .and_then(|repaired| serde_json::from_str::<Value>(&repaired).ok())
                {
                    Some(value) => {
                        tracing::warn!(
                            "Recovered malformed arguments for tool '{}' via structural JSON repair",
                            call.function.name
                        );
                        value
                    }
                    None => {
                        tracing::error!(
                            "Failed to parse arguments for tool '{}': {}. Args: {}",
                            call.function.name,
                            e,
                            args_str
                        );
                        invalid.push(call.clone());
                        continue;
                    }
                }
            }
        };

        // Unwrap double-encoded JSON strings
        if let Value::String(inner) = &parsed_args {
            if let Ok(decoded) = serde_json::from_str::<Value>(inner) {
                parsed_args = decoded;
            }
        }

        if !parsed_args.is_object() {
            tracing::error!(
                "Arguments for tool '{}' must be a JSON object. Got: {:?}",
                call.function.name,
                parsed_args
            );
            invalid.push(call.clone());
            continue;
        }

        let args_obj = match parsed_args.as_object() {
            Some(obj) => obj,
            None => {
                invalid.push(call.clone());
                continue;
            }
        };

        let schema = match schemas.get(&call.function.name) {
            Some(schema) => schema,
            None => {
                tracing::error!(
                    "Tool '{}' not found in schema map. Available tools: {:?}",
                    call.function.name,
                    schemas.keys().collect::<Vec<_>>()
                );
                invalid.push(call.clone());
                continue;
            }
        };

        let repaired_args_obj = repair_embedded_parameter_blocks(args_obj, schema);
        let normalized_args_obj = normalize_argument_keys(&repaired_args_obj, schema);
        let coerced_args_obj =
            coerce_argument_types(&normalized_args_obj, schema, &call.function.name);

        if let Some(missing) = missing_required_keys(&coerced_args_obj, schema) {
            tracing::error!(
                "Missing required argument(s) for tool '{}': {:?}. Args: {}",
                call.function.name,
                missing,
                args_str
            );
            invalid.push(call.clone());
            continue;
        }

        let filtered_args = Value::Object(coerced_args_obj);
        let normalized_args =
            serde_json::to_string(&filtered_args).unwrap_or_else(|_| args_str.to_string());

        valid.push(ToolCall {
            index: call.index,
            id: call.id.clone(),
            call_type: call.call_type.clone(),
            function: FunctionCall {
                name: call.function.name.clone(),
                arguments: normalized_args,
            },
        });
    }

    (valid, invalid)
}

/// Attempt to repair malformed/truncated JSON argument strings.
///
/// Closes unclosed strings and balances brackets/braces.
pub fn repair_json_arguments(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }

    if serde_json::from_str::<Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }

    let mut repaired = trimmed.to_string();
    let mut in_string = false;
    let mut escaped = false;
    let mut stack: Vec<char> = Vec::new();

    for ch in repaired.chars() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' | '[' => stack.push(ch),
            '}' => {
                if stack.last() == Some(&'{') {
                    stack.pop();
                }
            }
            ']' => {
                if stack.last() == Some(&'[') {
                    stack.pop();
                }
            }
            _ => {}
        }
    }

    // Close unclosed string
    if in_string {
        repaired.push('"');
    }

    // Strip trailing whitespace and commas
    while repaired
        .chars()
        .last()
        .is_some_and(|c| c.is_whitespace() || c == ',')
    {
        repaired.pop();
    }

    // Close unclosed brackets/braces
    while let Some(open) = stack.pop() {
        repaired.push(match open {
            '{' => '}',
            '[' => ']',
            _ => continue,
        });
    }

    Some(repaired)
}

/// Format tool calls for logging — returns a summary string.
pub fn format_tool_calls_summary(tool_calls: &[ToolCall]) -> String {
    if tool_calls.is_empty() {
        return String::new();
    }
    tool_calls
        .iter()
        .map(|call| {
            let args = call.function.arguments.replace('\n', " ");
            let truncated = if args.len() > 160 {
                let snippet: String = args.chars().take(160).collect();
                format!("{}...", snippet)
            } else {
                args
            };
            format!("{}(args={})", call.function.name, truncated)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Log tool calls with a label
pub fn log_tool_calls(label: &str, tool_calls: &[ToolCall]) {
    if tool_calls.is_empty() {
        return;
    }
    let summary = format_tool_calls_summary(tool_calls);
    tracing::info!("{} tool call(s): {}", label, summary);
}

// ── Internal helpers ──────────────────────────────────────────────────────

fn normalize_argument_keys(
    args_obj: &serde_json::Map<String, Value>,
    schema: &Value,
) -> serde_json::Map<String, Value> {
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return args_obj.clone();
    };

    let mut canonical_props: HashMap<String, Vec<String>> = HashMap::new();
    for prop in props.keys() {
        canonical_props
            .entry(canonicalize_key(prop))
            .or_default()
            .push(prop.clone());
    }

    let mut normalized = serde_json::Map::new();
    for (key, value) in args_obj {
        for candidate_key in normalized_key_candidates(key) {
            if props.contains_key(&candidate_key) {
                normalized
                    .entry(candidate_key.clone())
                    .or_insert_with(|| value.clone());
                continue;
            }

            let canonical = canonicalize_key(&candidate_key);
            if let Some(candidates) = canonical_props.get(&canonical) {
                if candidates.len() == 1 {
                    let target = &candidates[0];
                    normalized
                        .entry(target.clone())
                        .or_insert_with(|| value.clone());
                    continue;
                }
            }

            // Common fallback for editor/file tools where models often emit "file".
            if candidate_key == "file"
                && props.contains_key("filePath")
                && !props.contains_key("file")
            {
                normalized
                    .entry("filePath".to_string())
                    .or_insert_with(|| value.clone());
            }
        }
    }

    normalized
}

fn normalized_key_candidates(key: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    let trimmed = key.trim();
    if !trimmed.is_empty() {
        candidates.push(trimmed.to_string());
    }

    let stripped = strip_parameter_artifacts(trimmed);
    if !stripped.is_empty() && stripped != trimmed {
        candidates.push(stripped.to_string());
    }

    if let Some(name) = extract_parameter_name(trimmed) {
        if !name.is_empty() && !candidates.iter().any(|c| c == &name) {
            candidates.push(name);
        }
    }

    candidates
}

fn coerce_argument_types(
    args_obj: &serde_json::Map<String, Value>,
    schema: &Value,
    tool_name: &str,
) -> serde_json::Map<String, Value> {
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return args_obj.clone();
    };

    let mut coerced = args_obj.clone();
    for (key, prop_schema) in props {
        let Some(expected_ty) = prop_schema.get("type").and_then(Value::as_str) else {
            continue;
        };
        if expected_ty != "string" {
            continue;
        }

        let Some(value) = coerced.get_mut(key) else {
            continue;
        };
        match value {
            Value::Number(n) => {
                let converted = n.to_string();
                tracing::warn!(
                    "Coerced argument type for tool '{}': '{}' number -> string",
                    tool_name,
                    key
                );
                *value = Value::String(converted);
            }
            Value::Bool(b) => {
                let converted = b.to_string();
                tracing::warn!(
                    "Coerced argument type for tool '{}': '{}' bool -> string",
                    tool_name,
                    key
                );
                *value = Value::String(converted);
            }
            _ => {}
        }
    }

    coerced
}

fn strip_parameter_artifacts(key: &str) -> &str {
    let end = key
        .char_indices()
        .find_map(|(idx, ch)| matches!(ch, '\n' | '\r' | '<').then_some(idx))
        .unwrap_or(key.len());
    key[..end].trim()
}

fn canonicalize_key(key: &str) -> String {
    key.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn parameter_start_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?is)<\s*parameter\s*=\s*([^>\n\r]+)")
            .expect("parameter start regex must compile")
    })
}

fn missing_required_keys(
    args_obj: &serde_json::Map<String, Value>,
    schema: &Value,
) -> Option<Vec<String>> {
    let required = schema.get("required").and_then(|r| r.as_array())?;
    let mut missing = Vec::new();
    for key in required {
        let Some(name) = key.as_str() else {
            continue;
        };
        if !args_obj.get(name).is_some_and(|value| !value.is_null()) {
            missing.push(name.to_string());
        }
    }
    if missing.is_empty() {
        None
    } else {
        Some(missing)
    }
}

fn qwen_parameter_block_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?is)<\s*parameter\s*=\s*([^>\n\r]+)\s*>(.*?)</\s*parameter\s*>")
            .expect("qwen parameter regex must compile")
    })
}

fn parse_loose_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        v
    } else {
        Value::String(trimmed.to_string())
    }
}

fn parse_unclosed_parameter_block(text: &str) -> Option<(usize, String, Value)> {
    let caps = parameter_start_regex().captures(text)?;
    let (Some(full), Some(name_m)) = (caps.get(0), caps.get(1)) else {
        return None;
    };
    let start = full.start();
    let mut value_start = full.end();
    let name = name_m.as_str().trim();
    if name.is_empty() {
        return None;
    }

    if text.as_bytes().get(value_start) == Some(&b'>') {
        value_start += 1;
    }

    let value = parse_loose_value(&text[value_start..]);
    Some((start, name.to_string(), value))
}

fn repair_embedded_parameter_blocks(
    args_obj: &serde_json::Map<String, Value>,
    schema: &Value,
) -> serde_json::Map<String, Value> {
    let re = qwen_parameter_block_regex();
    let mut repaired = args_obj.clone();
    let mut extracted_raw = serde_json::Map::new();

    for (key, value) in args_obj {
        let Some(text) = value.as_str() else {
            continue;
        };
        if !parameter_start_regex().is_match(text) {
            if let Some(name) = extract_parameter_name(key) {
                extracted_raw.insert(name, parse_loose_value(text));
                repaired.remove(key);
            }
            continue;
        }

        let mut first_marker_start = None;
        let mut matched_any = false;
        for caps in re.captures_iter(text) {
            let (Some(full), Some(name_m), Some(value_m)) = (caps.get(0), caps.get(1), caps.get(2))
            else {
                continue;
            };
            matched_any = true;
            first_marker_start =
                Some(first_marker_start.map_or(full.start(), |s: usize| s.min(full.start())));
            let name = name_m.as_str().trim();
            if name.is_empty() {
                continue;
            }
            extracted_raw.insert(name.to_string(), parse_loose_value(value_m.as_str()));
        }

        if !matched_any {
            if let Some((marker_start, name, value)) = parse_unclosed_parameter_block(text) {
                matched_any = true;
                first_marker_start = Some(marker_start);
                extracted_raw.insert(name, value);
            }
        }

        if !matched_any {
            continue;
        }

        if let Some(marker_start) = first_marker_start {
            let prefix = text[..marker_start].trim();
            if prefix.is_empty() {
                repaired.remove(key);
            } else {
                repaired.insert(key.clone(), Value::String(prefix.to_string()));
            }
        }
    }

    if extracted_raw.is_empty() {
        return repaired;
    }

    // Canonicalize recovered keys against tool schema and merge.
    let extracted_normalized = normalize_argument_keys(&extracted_raw, schema);
    for (k, v) in extracted_normalized {
        repaired.insert(k, v);
    }
    repaired
}

fn extract_parameter_name(text: &str) -> Option<String> {
    let caps = parameter_start_regex().captures(text)?;
    let name = caps.get(1)?.as_str().trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_schema_map_works() {
        let tools = vec![crate::tools::function_tool("test", "desc")
            .param("arg1", "string", "desc", true)
            .build()];
        let map = build_tool_schema_map(&tools);
        assert!(map.contains_key("test"));
    }

    #[test]
    fn repair_json_closes_unclosed_brace() {
        let raw = r#"{"file_path":"/tmp/a.rs","content":"hello"#;
        let repaired = repair_json_arguments(raw).unwrap();
        let parsed: Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["file_path"], "/tmp/a.rs");
    }

    #[test]
    fn repair_json_closes_unclosed_string_and_brace() {
        let raw = r#"{"file_path":"/tmp/a.rs","content":"hello world"#;
        let repaired = repair_json_arguments(raw).unwrap();
        let parsed: Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["content"], "hello world");
    }

    #[test]
    fn repair_json_returns_empty_object_for_empty_input() {
        let repaired = repair_json_arguments("").unwrap();
        assert_eq!(repaired, "{}");
    }

    #[test]
    fn filter_valid_tool_call() {
        let tools = vec![crate::tools::function_tool("read", "desc")
            .param("path", "string", "path desc", true)
            .build()];
        let schemas = build_tool_schema_map(&tools);
        let calls = vec![ToolCall::new("call_1", "read", r#"{"path": "/tmp/a.txt"}"#)];

        let (valid, invalid) = filter_tool_calls(&calls, &schemas);
        assert_eq!(valid.len(), 1);
        assert!(invalid.is_empty());
        assert_eq!(valid[0].function.name, "read");
    }

    #[test]
    fn filter_rejects_unknown_tool() {
        let schemas = build_tool_schema_map(&[]);
        let calls = vec![ToolCall::new("call_1", "unknown", r#"{"arg": "val"}"#)];

        let (valid, invalid) = filter_tool_calls(&calls, &schemas);
        assert!(valid.is_empty());
        assert_eq!(invalid.len(), 1);
    }

    #[test]
    fn coerce_number_to_string() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "count": {"type": "string"}
            }
        });
        let mut args = serde_json::Map::new();
        args.insert("count".to_string(), Value::Number(42.into()));

        let coerced = coerce_argument_types(&args, &schema, "test");
        assert_eq!(coerced["count"], Value::String("42".to_string()));
    }

    #[test]
    fn normalize_keys_snake_to_camel() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "filePath": {"type": "string"}
            }
        });
        let mut args = serde_json::Map::new();
        args.insert(
            "file_path".to_string(),
            Value::String("/tmp/a.rs".to_string()),
        );

        let normalized = normalize_argument_keys(&args, &schema);
        assert!(normalized.contains_key("filePath"));
        assert_eq!(normalized["filePath"], "/tmp/a.rs");
    }

    #[test]
    fn format_summary_truncates_long_args() {
        let long_args = "a".repeat(200);
        let calls = vec![ToolCall::new("call_1", "test", &long_args)];
        let summary = format_tool_calls_summary(&calls);
        assert!(summary.contains("..."));
        assert!(summary.len() < 200);
    }
}
