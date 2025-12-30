// src/mcp/server.rs
//! MCP Server implementation
//!
//! Exposes tools, resources, and prompts to MCP clients.

use super::transport::{framing, Transport, TransportError};
use super::types::*;
use crate::tools::Tool;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Handler function for tool execution
pub type ToolHandler =
    Box<dyn Fn(HashMap<String, Value>) -> Result<CallToolResult, String> + Send + Sync>;

/// MCP Server that exposes tools to clients
#[allow(dead_code)]
pub struct McpServer {
    /// Server info
    server_info: Implementation,
    /// Server capabilities
    capabilities: ServerCapabilities,
    /// Registered tools
    tools: HashMap<String, (McpTool, Option<ToolHandler>)>,
    /// Registered resources
    resources: Vec<Resource>,
    /// Registered prompts
    prompts: Vec<Prompt>,
    /// Whether initialized
    initialized: bool,
    /// Request counter for IDs
    request_counter: i64,
}

impl McpServer {
    /// Create a new MCP server
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            server_info: Implementation {
                name: name.into(),
                version: version.into(),
            },
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: false,
                }),
                resources: None,
                prompts: None,
                logging: None,
            },
            tools: HashMap::new(),
            resources: Vec::new(),
            prompts: Vec::new(),
            initialized: false,
            request_counter: 0,
        }
    }

    /// Register a tool with an optional handler
    pub fn register_tool(&mut self, tool: McpTool, handler: Option<ToolHandler>) {
        self.tools.insert(tool.name.clone(), (tool, handler));
    }

    /// Register a tool from our internal Tool type
    pub fn register_internal_tool(&mut self, tool: &Tool, handler: Option<ToolHandler>) {
        let mcp_tool = McpTool {
            name: tool.function.name.clone(),
            description: Some(tool.function.description.clone()),
            input_schema: tool.function.parameters.clone(),
            output_schema: None,
        };
        self.register_tool(mcp_tool, handler);
    }

    /// Register a resource
    pub fn register_resource(&mut self, resource: Resource) {
        self.resources.push(resource);
        self.capabilities.resources = Some(ResourcesCapability {
            subscribe: false,
            list_changed: false,
        });
    }

    /// Register a prompt
    pub fn register_prompt(&mut self, prompt: Prompt) {
        self.prompts.push(prompt);
        self.capabilities.prompts = Some(PromptsCapability {
            list_changed: false,
        });
    }

    /// Handle an incoming JSON-RPC request
    pub fn handle_request(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(&request.params),
            "notifications/initialized" => {
                self.initialized = true;
                return JsonRpcResponse::success(request.id.clone(), json!({}));
            }
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(&request.params),
            "resources/list" => self.handle_resources_list(),
            "prompts/list" => self.handle_prompts_list(),
            "ping" => Ok(json!({})),
            _ => Err(JsonRpcError::method_not_found()),
        };

        match result {
            Ok(value) => JsonRpcResponse::success(request.id.clone(), value),
            Err(error) => JsonRpcResponse::error(request.id.clone(), error),
        }
    }

    fn handle_initialize(&mut self, params: &Option<Value>) -> Result<Value, JsonRpcError> {
        let _params: InitializeParams = params
            .as_ref()
            .ok_or_else(|| JsonRpcError::invalid_params("Missing params"))
            .and_then(|p| {
                serde_json::from_value(p.clone())
                    .map_err(|e| JsonRpcError::invalid_params(e.to_string()))
            })?;

        let result = InitializeResult {
            protocol_version: MCP_VERSION.to_string(),
            capabilities: self.capabilities.clone(),
            server_info: self.server_info.clone(),
            instructions: Some(format!(
                "vLLM.rs MCP Server - {} available tools",
                self.tools.len()
            )),
        };

        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    fn handle_tools_list(&self) -> Result<Value, JsonRpcError> {
        let tools: Vec<McpTool> = self.tools.values().map(|(tool, _)| tool.clone()).collect();

        let result = ListToolsResult {
            tools,
            next_cursor: None,
        };

        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    fn handle_tools_call(&self, params: &Option<Value>) -> Result<Value, JsonRpcError> {
        let call_params: CallToolParams = params
            .as_ref()
            .ok_or_else(|| JsonRpcError::invalid_params("Missing params"))
            .and_then(|p| {
                serde_json::from_value(p.clone())
                    .map_err(|e| JsonRpcError::invalid_params(e.to_string()))
            })?;

        let (_, handler) = self.tools.get(&call_params.name).ok_or_else(|| {
            JsonRpcError::invalid_params(format!("Unknown tool: {}", call_params.name))
        })?;

        let result = if let Some(handler) = handler {
            handler(call_params.arguments).map_err(|e| JsonRpcError::internal_error(e))?
        } else {
            // No handler - return a placeholder response
            CallToolResult {
                content: vec![ToolContent::text(format!(
                    "Tool '{}' has no handler registered",
                    call_params.name
                ))],
                is_error: true,
            }
        };

        serde_json::to_value(result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    fn handle_resources_list(&self) -> Result<Value, JsonRpcError> {
        serde_json::to_value(json!({
            "resources": self.resources,
            "nextCursor": null
        }))
        .map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    fn handle_prompts_list(&self) -> Result<Value, JsonRpcError> {
        serde_json::to_value(json!({
            "prompts": self.prompts,
            "nextCursor": null
        }))
        .map_err(|e| JsonRpcError::internal_error(e.to_string()))
    }

    /// Run the server on a transport (blocking)
    pub fn run<T: Transport>(&mut self, transport: &mut T) -> Result<(), TransportError> {
        loop {
            let line = transport.receive()?;

            if line.is_empty() {
                continue;
            }

            let message = framing::parse_message(&line)?;

            match message {
                super::transport::McpMessage::Request(request) => {
                    let response = self.handle_request(&request);
                    let response_str = framing::encode_line(&response)?;
                    transport.send(&response_str)?;
                }
                super::transport::McpMessage::Notification(notification) => {
                    // Handle notifications (no response needed)
                    if notification.method == "notifications/initialized" {
                        self.initialized = true;
                    }
                }
                super::transport::McpMessage::Response(_) => {
                    // Servers typically don't receive responses
                }
            }
        }
    }

    /// Get list of registered tool names
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

/// Builder for creating MCP servers with tools
pub struct McpServerBuilder {
    name: String,
    version: String,
    tools: Vec<(McpTool, Option<ToolHandler>)>,
}

impl McpServerBuilder {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            tools: Vec::new(),
        }
    }

    pub fn tool(mut self, tool: McpTool) -> Self {
        self.tools.push((tool, None));
        self
    }

    pub fn tool_with_handler(mut self, tool: McpTool, handler: ToolHandler) -> Self {
        self.tools.push((tool, Some(handler)));
        self
    }

    pub fn build(self) -> McpServer {
        let mut server = McpServer::new(self.name, self.version);
        for (tool, handler) in self.tools {
            server.register_tool(tool, handler);
        }
        server
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tool() -> McpTool {
        McpTool {
            name: "test_tool".to_string(),
            description: Some("A test tool".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            }),
            output_schema: None,
        }
    }

    #[test]
    fn test_server_creation() {
        let server = McpServer::new("test-server", "1.0.0");
        assert_eq!(server.server_info.name, "test-server");
        assert!(server.tools.is_empty());
    }

    #[test]
    fn test_register_tool() {
        let mut server = McpServer::new("test", "1.0");
        let tool = create_test_tool();
        server.register_tool(tool, None);

        assert_eq!(server.tools.len(), 1);
        assert!(server.tools.contains_key("test_tool"));
    }

    #[test]
    fn test_handle_tools_list() {
        let mut server = McpServer::new("test", "1.0");
        server.register_tool(create_test_tool(), None);

        let result = server.handle_tools_list().unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn test_handle_initialize() {
        let mut server = McpServer::new("test", "1.0");

        let request = JsonRpcRequest::new(
            1i64,
            "initialize",
            Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            })),
        );

        let response = server.handle_request(&request);
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_tool_with_handler() {
        let mut server = McpServer::new("test", "1.0");

        let tool = create_test_tool();
        let handler: ToolHandler = Box::new(|args| {
            let input = args
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("default");
            Ok(CallToolResult {
                content: vec![ToolContent::text(format!("Received: {}", input))],
                is_error: false,
            })
        });

        server.register_tool(tool, Some(handler));

        let result = server
            .handle_tools_call(&Some(json!({
                "name": "test_tool",
                "arguments": {"input": "hello"}
            })))
            .unwrap();

        let content = &result["content"][0]["text"];
        assert!(content.as_str().unwrap().contains("hello"));
    }
}
