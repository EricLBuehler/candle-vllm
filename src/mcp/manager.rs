// src/mcp/manager.rs
//! MCP client manager for vLLM.rs
//!
//! Manages a background MCP client thread and cached tool list.

use super::client::{McpClient, McpClientError};
use super::transport::StdioTransport;
use crate::tools::Tool;
use parking_lot::{Mutex, RwLock};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct McpToolConfig {
    pub command: String,
    pub args: Vec<String>,
    pub tool_refresh_interval: Duration,
}

impl McpToolConfig {
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            tool_refresh_interval: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpConfigFile {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerConfigFile>,
}

/// MCP server configuration from JSON config file.
/// Supports both local (stdio) and remote (HTTP) servers.
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerConfigFile {
    // Local (stdio) config - command is required for local servers
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    // Remote (HTTP) config - url is required for remote servers
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

/// Transport type for MCP server
#[derive(Debug, Clone)]
pub enum McpTransportType {
    /// Local stdio transport (spawns subprocess)
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    /// Remote HTTP transport (connects via HTTP/SSE)
    Http {
        url: String,
        headers: HashMap<String, String>,
    },
}

#[derive(Debug, Clone)]
pub struct McpServerDefinition {
    pub id: String,
    pub transport: McpTransportType,
}

#[derive(Debug, Clone)]
pub struct McpManagerConfig {
    pub servers: Vec<McpServerDefinition>,
    pub tool_refresh_interval: Duration,
}

impl McpManagerConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, McpClientError> {
        let contents = std::fs::read_to_string(path.as_ref())
            .map_err(|err| McpClientError::Config(format!("Failed to read MCP config: {err}")))?;
        let config: McpConfigFile =
            serde_json::from_str(&contents).map_err(McpClientError::Serialization)?;
        let mut servers = Vec::new();
        for (id, server) in config.mcp_servers {
            let transport = if let Some(url) = server.url {
                // Remote HTTP server
                McpTransportType::Http {
                    url,
                    headers: server.headers,
                }
            } else if let Some(command) = server.command {
                // Local stdio server
                McpTransportType::Stdio {
                    command,
                    args: server.args,
                    env: server.env,
                }
            } else {
                return Err(McpClientError::Config(format!(
                    "MCP server '{}' must have either 'command' (local) or 'url' (remote)",
                    id
                )));
            };
            servers.push(McpServerDefinition { id, transport });
        }
        Ok(Self {
            servers,
            tool_refresh_interval: Duration::from_secs(30),
        })
    }

    pub fn from_single(config: McpToolConfig) -> Self {
        Self {
            servers: vec![McpServerDefinition {
                id: "default".to_string(),
                transport: McpTransportType::Stdio {
                    command: config.command,
                    args: config.args,
                    env: HashMap::new(),
                },
            }],
            tool_refresh_interval: config.tool_refresh_interval,
        }
    }

    pub fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.tool_refresh_interval = interval;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ToolCache {
    tools: Vec<Tool>,
}

impl ToolCache {
    fn new() -> Self {
        Self { tools: Vec::new() }
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }

    pub fn tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Dynamic MCP client wrapper that supports both transport types
pub enum DynMcpClient {
    Stdio(McpClient<StdioTransport>),
    Http(McpClient<super::transport::HttpTransport>),
}

impl DynMcpClient {
    /// List available tools from the server
    pub fn list_tools(&mut self) -> Result<Vec<super::types::McpTool>, McpClientError> {
        match self {
            DynMcpClient::Stdio(client) => client.list_tools(),
            DynMcpClient::Http(client) => client.list_tools(),
        }
    }

    /// Call a tool on the server
    pub fn call_tool(
        &mut self,
        name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> Result<super::types::CallToolResult, McpClientError> {
        match self {
            DynMcpClient::Stdio(client) => client.call_tool(name, arguments),
            DynMcpClient::Http(client) => client.call_tool(name, arguments),
        }
    }
}

pub struct McpClientManager {
    tool_cache: Arc<RwLock<ToolCache>>,
    routing_table: Arc<RwLock<HashMap<String, ToolRouting>>>,
    clients: Arc<RwLock<HashMap<String, Arc<Mutex<DynMcpClient>>>>>,
    available: Arc<AtomicBool>,
}

impl McpClientManager {
    pub fn new(config: McpManagerConfig) -> Result<Self, McpClientError> {
        if config.servers.is_empty() {
            return Err(McpClientError::Config(
                "MCP manager requires at least one server".to_string(),
            ));
        }

        let tool_cache = Arc::new(RwLock::new(ToolCache::new()));
        let routing_table = Arc::new(RwLock::new(HashMap::new()));
        let clients: Arc<RwLock<HashMap<String, Arc<Mutex<DynMcpClient>>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let available = Arc::new(AtomicBool::new(false));
        // stop_flag removed

        let mut any_client = false;
        for server in &config.servers {
            match &server.transport {
                McpTransportType::Stdio { command, args, env } => {
                    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
                    match StdioTransport::spawn_with_env(command, &args_refs, env) {
                        Ok(transport) => {
                            let mut client = McpClient::new(transport, "vllm-rs", "0.6.0");
                            if let Err(err) = client.initialize() {
                                tracing::error!(
                                    "Failed to initialize MCP server {}: {:?}",
                                    server.id,
                                    err
                                );
                                continue;
                            }
                            clients.write().insert(
                                server.id.clone(),
                                Arc::new(Mutex::new(DynMcpClient::Stdio(client))),
                            );
                            any_client = true;
                        }
                        Err(err) => {
                            tracing::error!("Failed to spawn MCP server {}: {:?}", server.id, err);
                        }
                    }
                }
                McpTransportType::Http { url, headers } => {
                    match super::transport::HttpTransport::new(url.clone(), headers.clone()) {
                        Ok(transport) => {
                            let mut client = McpClient::new(transport, "vllm-rs", "0.6.0");
                            if let Err(err) = client.initialize() {
                                tracing::error!(
                                    "Failed to initialize remote MCP server {}: {:?}",
                                    server.id,
                                    err
                                );
                                continue;
                            }
                            tracing::info!(
                                "Connected to remote MCP server '{}' at {}",
                                server.id,
                                url
                            );
                            clients.write().insert(
                                server.id.clone(),
                                Arc::new(Mutex::new(DynMcpClient::Http(client))),
                            );
                            any_client = true;
                        }
                        Err(err) => {
                            tracing::error!(
                                "Failed to connect to remote MCP server {}: {:?}",
                                server.id,
                                err
                            );
                        }
                    }
                }
            }
        }

        if !any_client {
            return Err(McpClientError::Config(
                "Failed to initialize any MCP servers".to_string(),
            ));
        }

        // Perform initial synchronous tool fetch to ensure tools are available immediately
        refresh_tools(&clients.read(), &tool_cache, &routing_table, &available);

        Ok(Self {
            tool_cache,
            routing_table,
            clients,
            available,
        })
    }

    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Relaxed)
    }

    pub fn wait_for_available(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        while !self.is_available() {
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        true
    }

    pub fn cached_tools(&self) -> Vec<Tool> {
        self.tool_cache.read().tools()
    }

    pub fn call_tool(
        &self,
        name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> Result<super::types::CallToolResult, McpClientError> {
        let routing = self
            .routing_table
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| McpClientError::ToolNotFound(name.to_string()))?;

        let client = self
            .clients
            .read()
            .get(&routing.server_id)
            .cloned()
            .ok_or_else(|| McpClientError::ToolNotFound(name.to_string()))?;

        let mut client = client.lock();
        client.call_tool(&routing.original_name, arguments)
    }

    pub fn stop(&self) {
        // No background thread to stop
    }
}

#[derive(Debug, Clone)]
struct ToolRouting {
    server_id: String,
    original_name: String,
}

fn refresh_tools(
    clients: &HashMap<String, Arc<Mutex<DynMcpClient>>>,
    tool_cache: &RwLock<ToolCache>,
    routing_table: &RwLock<HashMap<String, ToolRouting>>,
    available: &AtomicBool,
) {
    let mut mapped_tools = Vec::new();
    let mut routing = HashMap::new();
    let mut any_success = false;

    for (server_id, client) in clients.iter() {
        let mut client = client.lock();
        match client.list_tools() {
            Ok(tools) => {
                any_success = true;
                mapped_tools.extend(map_mcp_tools(server_id, tools, &mut routing));
            }
            Err(err) => {
                tracing::error!("Failed to refresh MCP tools for {}: {:?}", server_id, err);
            }
        }
    }

    if any_success {
        tool_cache.write().set_tools(mapped_tools);
        *routing_table.write() = routing;
        available.store(true, Ordering::Relaxed);
    } else {
        available.store(false, Ordering::Relaxed);
    }
}

fn map_mcp_tools(
    server_id: &str,
    tools: Vec<super::types::McpTool>,
    routing: &mut HashMap<String, ToolRouting>,
) -> Vec<Tool> {
    tools
        .into_iter()
        .map(|tool| {
            let prefixed_name = format!("{server_id}_{}", tool.name);
            routing.insert(
                prefixed_name.clone(),
                ToolRouting {
                    server_id: server_id.to_string(),
                    original_name: tool.name,
                },
            );
            Tool {
                tool_type: "function".to_string(),
                function: crate::tools::FunctionDefinition {
                    name: prefixed_name,
                    description: tool.description.unwrap_or_default(),
                    parameters: tool.input_schema,
                    strict: None,
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::transport::{MemoryTransport, Transport};
    use crate::mcp::types::*;
    use serde_json::json;

    #[test]
    fn tool_cache_roundtrip() {
        let cache = ToolCache::new();
        assert!(cache.is_empty());
    }

    #[test]
    fn map_mcp_tool_to_openai_tool() {
        let mcp_tool = McpTool {
            name: "search".to_string(),
            description: Some("Search docs".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
            output_schema: None,
        };

        let tools = vec![mcp_tool];
        let mut routing = HashMap::new();
        let mapped = map_mcp_tools("filesystem", tools, &mut routing);

        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].function.name, "filesystem_search");
        let routing = routing.get("filesystem_search").unwrap();
        assert_eq!(routing.server_id, "filesystem");
        assert_eq!(routing.original_name, "search");
    }

    #[test]
    fn memory_transport_client_roundtrip() {
        let (mut client_transport, mut server_transport) = MemoryTransport::pair();
        let server = thread::spawn(move || {
            let mut server = crate::mcp::server::McpServer::new("test", "0.1");
            server.register_tool(
                McpTool {
                    name: "echo".to_string(),
                    description: None,
                    input_schema: json!({"type": "object"}),
                    output_schema: None,
                },
                Some(Box::new(|args| {
                    Ok(CallToolResult {
                        content: vec![ToolContent::text(format!(
                            "echo: {}",
                            args.get("message").and_then(|v| v.as_str()).unwrap_or("")
                        ))],
                        is_error: false,
                    })
                })),
            );
            let mut handled = 0;
            while handled < 3 {
                let line = server_transport.receive().unwrap();
                let msg = crate::mcp::transport::framing::parse_message(&line).unwrap();
                match msg {
                    crate::mcp::transport::McpMessage::Request(req) => {
                        let response = server.handle_request(&req);
                        let response_str =
                            crate::mcp::transport::framing::encode_line(&response).unwrap();
                        server_transport.send(&response_str).unwrap();
                        handled += 1;
                    }
                    crate::mcp::transport::McpMessage::Notification(_) => {}
                    crate::mcp::transport::McpMessage::Response(_) => {}
                }
            }
        });

        let mut client = McpClient::new(client_transport, "test-client", "0.1");
        let _ = client.initialize().unwrap();
        let tools = client.list_tools().unwrap();
        assert_eq!(tools.len(), 1);

        let result = client
            .call_tool(
                "echo",
                [("message".to_string(), json!("hello"))]
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        match &result.content[0] {
            ToolContent::Text { text } => assert_eq!(text, "echo: hello"),
            _ => panic!("unexpected tool content"),
        }
        server.join().unwrap();
    }

    #[test]
    fn parse_mcp_config_file() {
        let json = r#"{
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                },
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_example"
                    }
                }
            }
        }"#;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("mcp_config_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();

        let config = McpManagerConfig::from_file(&path).unwrap();
        assert_eq!(config.servers.len(), 2);
        let filesystem = config
            .servers
            .iter()
            .find(|server| server.id == "filesystem")
            .unwrap();
        match &filesystem.transport {
            McpTransportType::Stdio { command, args, .. } => {
                assert_eq!(command, "npx");
                assert_eq!(
                    args,
                    &vec!["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                );
            }
            _ => panic!("Expected Stdio transport"),
        }
        let github = config
            .servers
            .iter()
            .find(|server| server.id == "github")
            .unwrap();
        match &github.transport {
            McpTransportType::Stdio { env, .. } => {
                assert_eq!(
                    env.get("GITHUB_PERSONAL_ACCESS_TOKEN").unwrap(),
                    "ghp_example"
                );
            }
            _ => panic!("Expected Stdio transport"),
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_mcp_config_with_remote_servers() {
        let json = r#"{
            "mcpServers": {
                "remote-mcp": {
                    "url": "https://mcp.example.com/api/",
                    "headers": {
                        "Authorization": "Bearer token123",
                        "X-Custom-Header": "value"
                    }
                },
                "local-filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
                }
            }
        }"#;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("mcp_config_remote_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();

        let config = McpManagerConfig::from_file(&path).unwrap();
        assert_eq!(config.servers.len(), 2);

        let remote = config
            .servers
            .iter()
            .find(|server| server.id == "remote-mcp")
            .unwrap();
        match &remote.transport {
            McpTransportType::Http { url, headers } => {
                assert_eq!(url, "https://mcp.example.com/api/");
                assert_eq!(headers.get("Authorization").unwrap(), "Bearer token123");
                assert_eq!(headers.get("X-Custom-Header").unwrap(), "value");
            }
            _ => panic!("Expected Http transport"),
        }

        let local = config
            .servers
            .iter()
            .find(|server| server.id == "local-filesystem")
            .unwrap();
        match &local.transport {
            McpTransportType::Stdio { command, .. } => {
                assert_eq!(command, "npx");
            }
            _ => panic!("Expected Stdio transport"),
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_mcp_config_invalid_missing_command_and_url() {
        let json = r#"{
            "mcpServers": {
                "invalid-server": {
                    "args": ["--some-arg"]
                }
            }
        }"#;

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("mcp_config_invalid_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();

        let result = McpManagerConfig::from_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{:?}", err).contains("must have either 'command' (local) or 'url' (remote)")
        );

        let _ = std::fs::remove_file(&path);
    }
}
