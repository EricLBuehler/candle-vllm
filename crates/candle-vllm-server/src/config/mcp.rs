use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::fs;
use std::collections::HashMap;

/// Single MCP server entry sourced from `mcp.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerDefinition {
    pub name: String,
    pub url: String,
    pub auth: Option<String>,
    pub timeout_secs: Option<u64>,
    pub instructions: Option<String>,
}

/// MCP server entry from mcpServers format (with type, command, args, env)
#[derive(Debug, Clone, Deserialize)]
struct McpServersEntry {
    #[serde(rename = "type")]
    _type: Option<String>,
    command: Option<String>,
    args: Option<Vec<String>>,
    env: Option<HashMap<String, String>>,
    // HTTP server format
    url: Option<String>,
    auth: Option<String>,
    timeout_secs: Option<u64>,
    instructions: Option<String>,
}

/// Collection of MCP servers.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: Vec<McpServerDefinition>,
}

impl McpConfig {
    /// Load MCP config from file, supporting both formats:
    /// - `servers` array format: `{"servers": [{"name": "...", "url": "..."}]}`
    /// - `mcpServers` object format: `{"mcpServers": {"server-name": {"type": "node", "command": "..."}}}`
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let json: Value = serde_json::from_str(&content)?;
        
        // Try servers array format first
        if let Some(servers_array) = json.get("servers").and_then(|s| s.as_array()) {
            let servers: Vec<McpServerDefinition> = serde_json::from_value(
                serde_json::Value::Array(servers_array.clone())
            )?;
            return Ok(McpConfig { servers });
        }
        
        // Try mcpServers object format
        if let Some(mcp_servers) = json.get("mcpServers").and_then(|s| s.as_object()) {
            let mut servers = Vec::new();
            for (name, entry_value) in mcp_servers {
                let entry: McpServersEntry = serde_json::from_value(entry_value.clone())?;
                
                // For HTTP servers, use url directly
                if let Some(url) = entry.url {
                    servers.push(McpServerDefinition {
                        name: name.clone(),
                        url,
                        auth: entry.auth,
                        timeout_secs: entry.timeout_secs,
                        instructions: entry.instructions,
                    });
                } else if let Some(command) = entry.command {
                    // For command-based servers, we need to convert to HTTP URL
                    // This assumes the server is running locally and exposes HTTP
                    // Default to localhost:3000 + server name
                    let url = format!("http://localhost:3000/{}", name);
                    tracing::warn!(
                        "MCP server '{}' uses command format. Converting to HTTP URL: {}",
                        name, url
                    );
                    servers.push(McpServerDefinition {
                        name: name.clone(),
                        url,
                        auth: None,
                        timeout_secs: Some(30),
                        instructions: None,
                    });
                }
            }
            return Ok(McpConfig { servers });
        }
        
        // Fallback: try direct array
        if let Ok(servers) = serde_json::from_value::<Vec<McpServerDefinition>>(json.clone()) {
            return Ok(McpConfig { servers });
        }
        
        anyhow::bail!("Invalid MCP config format. Expected 'servers' array or 'mcpServers' object")
    }
}
