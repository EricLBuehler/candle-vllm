// src/mcp/client.rs
//! MCP Client implementation
//!
//! Connect to external MCP servers to discover and call tools.

use super::transport::{framing, Transport, TransportError};
use super::types::*;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};

/// MCP Client for connecting to MCP servers
pub struct McpClient<T: Transport> {
    /// Transport layer
    transport: T,
    /// Client info
    client_info: Implementation,
    /// Server info (after initialization)
    server_info: Option<Implementation>,
    /// Server capabilities (after initialization)
    server_capabilities: Option<ServerCapabilities>,
    /// Cached tools list
    tools_cache: Vec<McpTool>,
    /// Request ID counter
    request_counter: AtomicI64,
    /// Whether initialized
    initialized: bool,
}

impl<T: Transport> McpClient<T> {
    /// Create a new MCP client
    pub fn new(transport: T, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            transport,
            client_info: Implementation {
                name: name.into(),
                version: version.into(),
            },
            server_info: None,
            server_capabilities: None,
            tools_cache: Vec::new(),
            request_counter: AtomicI64::new(1),
            initialized: false,
        }
    }

    /// Get the next request ID
    fn next_id(&self) -> RequestId {
        RequestId::Number(self.request_counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Send a request and wait for response
    fn send_request(
        &mut self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, McpClientError> {
        let id = self.next_id();
        let request = JsonRpcRequest::new(id.clone(), method, params);

        let request_str = framing::encode_line(&request).map_err(McpClientError::Transport)?;

        self.transport
            .send(&request_str)
            .map_err(McpClientError::Transport)?;

        // Wait for response
        loop {
            let line = self
                .transport
                .receive()
                .map_err(McpClientError::Transport)?;

            if line.is_empty() {
                continue;
            }

            let message = framing::parse_message(&line).map_err(McpClientError::Transport)?;

            match message {
                super::transport::McpMessage::Response(response) => {
                    if response.id == id {
                        if let Some(error) = response.error {
                            return Err(McpClientError::ServerError(error));
                        }
                        return response.result.ok_or(McpClientError::EmptyResponse);
                    }
                }
                super::transport::McpMessage::Notification(_) => {
                    // Handle notifications - continue waiting for response
                    continue;
                }
                super::transport::McpMessage::Request(_) => {
                    // Server sending request to client - handle if needed
                    continue;
                }
            }
        }
    }

    /// Send a notification (no response expected)
    fn send_notification(
        &mut self,
        method: &str,
        params: Option<Value>,
    ) -> Result<(), McpClientError> {
        let notification = JsonRpcNotification {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
        };

        let notification_str =
            framing::encode_line(&notification).map_err(McpClientError::Transport)?;

        self.transport
            .send(&notification_str)
            .map_err(McpClientError::Transport)?;

        Ok(())
    }

    /// Initialize the connection with the server
    pub fn initialize(&mut self) -> Result<InitializeResult, McpClientError> {
        let params = InitializeParams {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: self.client_info.clone(),
        };

        let result = self.send_request("initialize", Some(serde_json::to_value(&params)?))?;
        let init_result: InitializeResult = serde_json::from_value(result)?;

        self.server_info = Some(init_result.server_info.clone());
        self.server_capabilities = Some(init_result.capabilities.clone());

        // Send initialized notification
        self.send_notification("notifications/initialized", None)?;
        self.initialized = true;

        Ok(init_result)
    }

    /// List available tools from the server
    pub fn list_tools(&mut self) -> Result<Vec<McpTool>, McpClientError> {
        if !self.initialized {
            return Err(McpClientError::NotInitialized);
        }

        let result = self.send_request("tools/list", None)?;
        let list_result: ListToolsResult = serde_json::from_value(result)?;

        self.tools_cache = list_result.tools.clone();
        Ok(list_result.tools)
    }

    /// Call a tool on the server
    pub fn call_tool(
        &mut self,
        name: impl Into<String>,
        arguments: HashMap<String, Value>,
    ) -> Result<CallToolResult, McpClientError> {
        if !self.initialized {
            return Err(McpClientError::NotInitialized);
        }

        let params = CallToolParams {
            name: name.into(),
            arguments,
        };

        let result = self.send_request("tools/call", Some(serde_json::to_value(&params)?))?;
        let call_result: CallToolResult = serde_json::from_value(result)?;

        Ok(call_result)
    }

    /// Get cached tools (from last list_tools call)
    pub fn cached_tools(&self) -> &[McpTool] {
        &self.tools_cache
    }

    /// Check if a tool exists (from cache)
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools_cache.iter().any(|t| t.name == name)
    }

    /// Get server info (after initialization)
    pub fn server_info(&self) -> Option<&Implementation> {
        self.server_info.as_ref()
    }

    /// Get server capabilities (after initialization)
    pub fn capabilities(&self) -> Option<&ServerCapabilities> {
        self.server_capabilities.as_ref()
    }

    /// Close the connection
    pub fn close(mut self) -> Result<(), McpClientError> {
        self.transport.close().map_err(McpClientError::Transport)
    }
}

/// Errors that can occur during MCP client operations
#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    #[error("Transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("Server error: {0}")]
    ServerError(JsonRpcError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Client not initialized")]
    NotInitialized,

    #[error("Empty response from server")]
    EmptyResponse,

    #[error("Tool not found: {0}")]
    ToolNotFound(String),
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code: {})", self.message, self.code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_client_error_display() {
        let err = McpClientError::NotInitialized;
        assert!(format!("{}", err).contains("not initialized"));

        let err = McpClientError::ToolNotFound("test".to_string());
        assert!(format!("{}", err).contains("test"));
    }

    #[test]
    fn test_json_rpc_error_display() {
        let err = JsonRpcError {
            code: -32600,
            message: "Invalid Request".to_string(),
            data: None,
        };
        assert!(format!("{}", err).contains("Invalid Request"));
        assert!(format!("{}", err).contains("-32600"));
    }
}
