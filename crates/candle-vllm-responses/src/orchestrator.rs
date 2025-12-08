//! Orchestrator for multi-turn agent conversations.
//!
//! This module handles tool call routing, execution, and result injection
//! for automated multi-turn conversations.

use crate::mcp_client::McpClient;
use candle_vllm_core::openai::requests::{ChatMessage, ToolCall};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Orchestrator for managing tool calls and routing them to MCP servers.
pub struct Orchestrator {
    mcp_clients: HashMap<String, McpClient>,
}

impl Orchestrator {
    /// Create a new orchestrator with MCP clients.
    pub fn new(mcp_clients: HashMap<String, McpClient>) -> Self {
        Self { mcp_clients }
    }

    /// Route and execute tool calls, returning tool response messages.
    ///
    /// This method:
    /// 1. Parses tool call names to extract server and tool name
    /// 2. Routes each tool call to the appropriate MCP server
    /// 3. Executes the tool calls
    /// 4. Converts results back to ChatMessage format
    pub async fn execute_tool_calls(
        &self,
        tool_calls: &[ToolCall],
        allowed_tools: Option<&[String]>,
    ) -> anyhow::Result<Vec<ChatMessage>> {
        info!("Executing {} tool call(s)", tool_calls.len());
        let mut responses = Vec::new();

        for tool_call in tool_calls {
            let full_tool_name = tool_call.name();

            // Check if tool is allowed
            if let Some(allowed) = allowed_tools {
                if !allowed.iter().any(|a| full_tool_name.contains(a)) {
                    warn!("Tool '{}' is not in allowed list, skipping", full_tool_name);
                    continue;
                }
            }

            // Parse server::tool_name format
            let (server, tool_name) = self.parse_tool_name(full_tool_name)?;

            info!("Executing tool call: {}::{}", server, tool_name);

            // Get the MCP client for this server
            let client = self
                .mcp_clients
                .get(&server)
                .ok_or_else(|| anyhow::anyhow!("MCP server '{server}' not found for tool call"))?;

            // Convert tool call to MCP format
            let mcp_payload = tool_call.to_mcp_call();

            // Execute the tool call
            debug!(
                "Calling MCP tool {}::{} with payload: {:?}",
                server, tool_name, mcp_payload
            );
            let result = client.call_tool(&tool_name, mcp_payload).await?;
            info!("Tool call {}::{} completed successfully", server, tool_name);

            // Convert result to ChatMessage
            let response = ChatMessage::from_mcp_result(
                tool_call.id.clone(),
                &result,
                Some(tool_name.clone()),
            );

            responses.push(response);
        }

        Ok(responses)
    }

    /// Parse tool name in format "server::tool_name" into (server, tool_name).
    fn parse_tool_name(&self, name: &str) -> anyhow::Result<(String, String)> {
        if let Some((server, tool)) = name.split_once("::") {
            Ok((server.to_string(), tool.to_string()))
        } else {
            // If no server prefix, try to find a matching server
            // For now, return error - tools should be prefixed
            anyhow::bail!("Tool name '{name}' must be in format 'server::tool_name'");
        }
    }

    /// Filter tools by allowed names.
    pub fn filter_allowed_tools(
        tools: &[candle_vllm_core::openai::requests::Tool],
        allowed: Option<&[String]>,
    ) -> Vec<candle_vllm_core::openai::requests::Tool> {
        if let Some(allowed) = allowed {
            let allowed_refs: Vec<&str> = allowed.iter().map(|s| s.as_str()).collect();
            candle_vllm_core::openai::requests::Tool::filter_by_names(tools, &allowed_refs)
        } else {
            tools.to_vec()
        }
    }
}
