//! Tool Call Streaming State Machine
//!
//! This module provides a state machine for tracking and emitting incremental
//! tool call deltas during streaming responses. It follows the AG-UI protocol
//! pattern of Start → Args → End for tool call events.

use std::collections::HashMap;
use uuid::Uuid;

use super::requests::{FunctionCallDelta, ToolCall, ToolCallDelta};

/// State machine for tracking tool call streaming
#[derive(Debug, Clone)]
pub struct ToolCallStreamState {
    /// Current active tool calls (index -> state)
    active_calls: HashMap<usize, ActiveToolCall>,
    /// Next available tool call index
    next_index: usize,
}

/// Internal state for an active tool call
#[derive(Debug, Clone)]
struct ActiveToolCall {
    /// Unique ID for this tool call
    id: String,
    /// Tool/function name
    name: String,
    /// Accumulated arguments (JSON fragments)
    arguments_buffer: String,
    /// State of this tool call
    state: ToolCallState,
}

/// State of a tool call
#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolCallState {
    /// Just started, need to emit start event
    Started,
    /// Emitting arguments
    Arguments,
    /// Completed, need to emit end event
    Completed,
}

impl ToolCallStreamState {
    /// Create a new tool call state machine
    pub fn new() -> Self {
        Self {
            active_calls: HashMap::new(),
            next_index: 0,
        }
    }

    /// Start a new tool call
    ///
    /// Returns the index and a start delta that should be emitted immediately.
    /// The start delta contains the tool call ID, type, and function name.
    pub fn start_tool_call(&mut self, name: String) -> (usize, ToolCallDelta) {
        let id = format!(
            "call_{}",
            Uuid::new_v4().simple().to_string()[..12].to_string()
        );
        let index = self.next_index;
        self.next_index += 1;

        self.active_calls.insert(
            index,
            ActiveToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments_buffer: String::new(),
                state: ToolCallState::Started,
            },
        );

        // Create start delta
        let delta = ToolCallDelta {
            index,
            id: Some(id),
            call_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some(name),
                arguments: Some(String::new()),
            }),
        };

        (index, delta)
    }

    /// Add arguments to an active tool call
    ///
    /// Returns an arguments delta that should be emitted.
    /// The delta only contains the new arguments fragment.
    pub fn add_arguments(&mut self, index: usize, args: &str) -> Option<ToolCallDelta> {
        if let Some(call) = self.active_calls.get_mut(&index) {
            call.arguments_buffer.push_str(args);
            call.state = ToolCallState::Arguments;

            Some(ToolCallDelta {
                index,
                id: None,
                call_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(args.to_string()),
                }),
            })
        } else {
            None
        }
    }

    /// Complete a tool call
    ///
    /// Returns a completion delta (empty function) that signals the end.
    /// After this, the tool call is marked as completed.
    pub fn complete_tool_call(&mut self, index: usize) -> Option<ToolCallDelta> {
        if let Some(call) = self.active_calls.get_mut(&index) {
            call.state = ToolCallState::Completed;

            // Final delta (empty to signal completion)
            Some(ToolCallDelta {
                index,
                id: None,
                call_type: None,
                function: None,
            })
        } else {
            None
        }
    }

    /// Get all completed tool calls for final response
    ///
    /// This is used in non-streaming mode or when finalizing a stream
    /// to construct the complete list of tool calls.
    pub fn finalize(&self) -> Vec<ToolCall> {
        self.active_calls
            .values()
            .filter(|c| c.state == ToolCallState::Completed)
            .map(|c| ToolCall {
                id: c.id.clone(),
                call_type: "function".to_string(),
                function: super::requests::FunctionCall {
                    name: c.name.clone(),
                    arguments: c.arguments_buffer.clone(),
                },
            })
            .collect()
    }

    /// Check if there are any active tool calls
    pub fn has_active_calls(&self) -> bool {
        !self.active_calls.is_empty()
    }

    /// Get the number of active tool calls
    pub fn active_count(&self) -> usize {
        self.active_calls.len()
    }

    /// Clear all tool calls (for cleanup)
    pub fn clear(&mut self) {
        self.active_calls.clear();
        self.next_index = 0;
    }
}

impl Default for ToolCallStreamState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_state_machine_basic() {
        let mut state = ToolCallStreamState::new();

        // Start tool call
        let (index, start_delta) = state.start_tool_call("get_weather".to_string());
        assert_eq!(index, 0);
        assert!(start_delta.id.is_some());
        assert_eq!(start_delta.call_type, Some("function".to_string()));
        assert_eq!(
            start_delta.function.as_ref().unwrap().name,
            Some("get_weather".to_string())
        );
        assert_eq!(start_delta.index, 0);

        // Add arguments incrementally
        let args_delta1 = state.add_arguments(index, "{\"location\":").unwrap();
        assert_eq!(args_delta1.index, 0);
        assert_eq!(args_delta1.id, None);
        assert_eq!(
            args_delta1.function.as_ref().unwrap().arguments,
            Some("{\"location\":".to_string())
        );

        let args_delta2 = state.add_arguments(index, " \"NYC\"}").unwrap();
        assert_eq!(
            args_delta2.function.as_ref().unwrap().arguments,
            Some(" \"NYC\"}".to_string())
        );

        // Complete
        let end_delta = state.complete_tool_call(index).unwrap();
        assert_eq!(end_delta.index, 0);
        assert!(end_delta.function.is_none());
    }

    #[test]
    fn test_finalize() {
        let mut state = ToolCallStreamState::new();

        // Create and complete a tool call
        let (index, _) = state.start_tool_call("get_weather".to_string());
        state
            .add_arguments(index, "{\"location\": \"NYC\"}")
            .unwrap();
        state.complete_tool_call(index).unwrap();

        // Finalize
        let calls = state.finalize();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\": \"NYC\"}");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_multiple_tool_calls() {
        let mut state = ToolCallStreamState::new();

        // First tool call
        let (idx1, _) = state.start_tool_call("get_weather".to_string());
        state
            .add_arguments(idx1, "{\"location\": \"NYC\"}")
            .unwrap();
        state.complete_tool_call(idx1).unwrap();

        // Second tool call
        let (idx2, _) = state.start_tool_call("calculate".to_string());
        state.add_arguments(idx2, "{\"expr\": \"2+2\"}").unwrap();
        state.complete_tool_call(idx2).unwrap();

        assert_eq!(state.active_count(), 2);

        let calls = state.finalize();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_invalid_index() {
        let mut state = ToolCallStreamState::new();

        // Try to add arguments to non-existent tool call
        let result = state.add_arguments(99, "test");
        assert!(result.is_none());

        // Try to complete non-existent tool call
        let result = state.complete_tool_call(99);
        assert!(result.is_none());
    }

    #[test]
    fn test_clear() {
        let mut state = ToolCallStreamState::new();

        state.start_tool_call("test".to_string());
        assert_eq!(state.active_count(), 1);

        state.clear();
        assert_eq!(state.active_count(), 0);
        assert!(!state.has_active_calls());
    }
}
