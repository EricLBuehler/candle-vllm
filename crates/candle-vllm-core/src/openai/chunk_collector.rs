//! Chunk Collector for Non-Streaming Extensions
//!
//! This module provides a collector for gathering reasoning chunks and tool call
//! events during generation, which are then added to the `extensions` field in
//! non-streaming completion responses.

use serde::{Deserialize, Serialize};

/// Collector for chunks in non-streaming mode
///
/// Accumulates reasoning tokens and tool call events during generation,
/// which are then serialized into the `extensions` field of the final response.
#[derive(Debug, Clone, Default)]
pub struct ChunkCollector {
    /// Reasoning chunks (thinking/chain-of-thought tokens)
    reasoning_chunks: Vec<String>,
    /// Tool call event chunks
    tool_call_chunks: Vec<ToolCallChunkEvent>,
    /// Content chunks (for debugging/logging only, not included in extensions)
    content_chunks: Vec<String>,
}

/// A tool call event chunk for the extensions field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallChunkEvent {
    /// Type of event: "start", "args", or "end"
    #[serde(rename = "type")]
    pub event_type: ToolCallEventType,
    /// ID of the tool call
    pub tool_call_id: String,
    /// Tool name (only present in "start" events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Arguments delta (only present in "args" events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
}

/// Type of tool call event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallEventType {
    /// Tool call started (includes name)
    Start,
    /// Tool call arguments fragment
    Args,
    /// Tool call completed
    End,
}

impl ChunkCollector {
    /// Create a new chunk collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a reasoning chunk
    ///
    /// Reasoning chunks are accumulated and added to the `reasoning_chunks`
    /// array in the extensions field. Empty strings are ignored.
    pub fn add_reasoning(&mut self, text: String) {
        if !text.is_empty() {
            self.reasoning_chunks.push(text);
        }
    }

    /// Add a tool call start event
    ///
    /// This records that a tool call has started with the given ID and name.
    pub fn add_tool_call_start(&mut self, id: String, name: String) {
        self.tool_call_chunks.push(ToolCallChunkEvent {
            event_type: ToolCallEventType::Start,
            tool_call_id: id,
            tool_name: Some(name),
            delta: None,
        });
    }

    /// Add tool call arguments
    ///
    /// This records a fragment of the tool call arguments (typically JSON).
    pub fn add_tool_call_args(&mut self, id: String, args: String) {
        if !args.is_empty() {
            self.tool_call_chunks.push(ToolCallChunkEvent {
                event_type: ToolCallEventType::Args,
                tool_call_id: id,
                tool_name: None,
                delta: Some(args),
            });
        }
    }

    /// Add tool call end event
    ///
    /// This records that a tool call has completed.
    pub fn add_tool_call_end(&mut self, id: String) {
        self.tool_call_chunks.push(ToolCallChunkEvent {
            event_type: ToolCallEventType::End,
            tool_call_id: id,
            tool_name: None,
            delta: None,
        });
    }

    /// Add content chunk (for debugging/logging only)
    ///
    /// Content chunks are not included in the extensions field, but can be
    /// useful for debugging or logging purposes.
    pub fn add_content(&mut self, text: String) {
        if !text.is_empty() {
            self.content_chunks.push(text);
        }
    }

    /// Convert to extensions JSON
    ///
    /// Creates a JSON object with `reasoning_chunks` and/or `tool_call_chunks`
    /// arrays, depending on what was collected. Returns an empty object if
    /// nothing was collected.
    pub fn to_extensions(&self) -> serde_json::Value {
        let mut extensions = serde_json::Map::new();

        if !self.reasoning_chunks.is_empty() {
            extensions.insert(
                "reasoning_chunks".to_string(),
                serde_json::json!(self.reasoning_chunks),
            );
        }

        if !self.tool_call_chunks.is_empty() {
            extensions.insert(
                "tool_call_chunks".to_string(),
                serde_json::json!(self.tool_call_chunks),
            );
        }

        serde_json::Value::Object(extensions)
    }

    /// Check if collector has any data
    ///
    /// Returns `true` if either reasoning chunks or tool call chunks have been collected.
    pub fn is_empty(&self) -> bool {
        self.reasoning_chunks.is_empty() && self.tool_call_chunks.is_empty()
    }

    /// Get the number of reasoning chunks
    pub fn reasoning_count(&self) -> usize {
        self.reasoning_chunks.len()
    }

    /// Get the number of tool call events
    pub fn tool_call_event_count(&self) -> usize {
        self.tool_call_chunks.len()
    }

    /// Get the number of content chunks (for debugging)
    pub fn content_count(&self) -> usize {
        self.content_chunks.len()
    }

    /// Clear all collected data
    pub fn clear(&mut self) {
        self.reasoning_chunks.clear();
        self.tool_call_chunks.clear();
        self.content_chunks.clear();
    }

    /// Get a reference to the reasoning chunks
    pub fn reasoning_chunks(&self) -> &[String] {
        &self.reasoning_chunks
    }

    /// Get a reference to the tool call chunks
    pub fn tool_call_chunks(&self) -> &[ToolCallChunkEvent] {
        &self.tool_call_chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_collector_basic() {
        let mut collector = ChunkCollector::new();

        collector.add_reasoning("Thinking step 1".to_string());
        collector.add_reasoning("Thinking step 2".to_string());

        assert_eq!(collector.reasoning_count(), 2);
        assert!(!collector.is_empty());
    }

    #[test]
    fn test_tool_call_chunks() {
        let mut collector = ChunkCollector::new();

        collector.add_tool_call_start("call_123".to_string(), "calculate".to_string());
        collector.add_tool_call_args("call_123".to_string(), "{\"expr\":".to_string());
        collector.add_tool_call_args("call_123".to_string(), "\"2+2\"}".to_string());
        collector.add_tool_call_end("call_123".to_string());

        assert_eq!(collector.tool_call_event_count(), 4);
    }

    #[test]
    fn test_to_extensions() {
        let mut collector = ChunkCollector::new();

        collector.add_reasoning("Step 1".to_string());
        collector.add_reasoning("Step 2".to_string());

        collector.add_tool_call_start("call_123".to_string(), "get_weather".to_string());
        collector.add_tool_call_args("call_123".to_string(), "{\"location\":\"NYC\"}".to_string());
        collector.add_tool_call_end("call_123".to_string());

        let extensions = collector.to_extensions();
        assert!(!collector.is_empty());

        let reasoning = extensions
            .get("reasoning_chunks")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(reasoning.len(), 2);
        assert_eq!(reasoning[0], "Step 1");

        let tool_chunks = extensions
            .get("tool_call_chunks")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(tool_chunks.len(), 3);

        // Verify first chunk is a start event
        let first = &tool_chunks[0];
        assert_eq!(first["type"], "start");
        assert_eq!(first["tool_call_id"], "call_123");
        assert_eq!(first["tool_name"], "get_weather");
    }

    #[test]
    fn test_empty_collector() {
        let collector = ChunkCollector::new();
        assert!(collector.is_empty());

        let extensions = collector.to_extensions();
        assert_eq!(extensions.as_object().unwrap().len(), 0);
    }

    #[test]
    fn test_only_reasoning() {
        let mut collector = ChunkCollector::new();
        collector.add_reasoning("Test".to_string());

        let extensions = collector.to_extensions();
        assert!(extensions.get("reasoning_chunks").is_some());
        assert!(extensions.get("tool_call_chunks").is_none());
    }

    #[test]
    fn test_only_tool_calls() {
        let mut collector = ChunkCollector::new();
        collector.add_tool_call_start("call_1".to_string(), "test".to_string());
        collector.add_tool_call_end("call_1".to_string());

        let extensions = collector.to_extensions();
        assert!(extensions.get("reasoning_chunks").is_none());
        assert!(extensions.get("tool_call_chunks").is_some());
    }

    #[test]
    fn test_ignore_empty_strings() {
        let mut collector = ChunkCollector::new();

        collector.add_reasoning("".to_string());
        collector.add_tool_call_args("call_1".to_string(), "".to_string());
        collector.add_content("".to_string());

        assert_eq!(collector.reasoning_count(), 0);
        assert_eq!(collector.tool_call_event_count(), 0);
        assert_eq!(collector.content_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut collector = ChunkCollector::new();

        collector.add_reasoning("Test".to_string());
        collector.add_tool_call_start("call_1".to_string(), "test".to_string());
        collector.add_content("Content".to_string());

        assert!(!collector.is_empty());
        assert_eq!(collector.reasoning_count(), 1);
        assert_eq!(collector.tool_call_event_count(), 1);
        assert_eq!(collector.content_count(), 1);

        collector.clear();

        assert!(collector.is_empty());
        assert_eq!(collector.reasoning_count(), 0);
        assert_eq!(collector.tool_call_event_count(), 0);
        assert_eq!(collector.content_count(), 0);
    }

    #[test]
    fn test_serialization() {
        let mut collector = ChunkCollector::new();
        collector.add_tool_call_start("call_abc".to_string(), "get_weather".to_string());

        let extensions = collector.to_extensions();
        let json_str = serde_json::to_string(&extensions).unwrap();

        assert!(json_str.contains("tool_call_chunks"));
        assert!(json_str.contains("call_abc"));
        assert!(json_str.contains("get_weather"));
    }
}
