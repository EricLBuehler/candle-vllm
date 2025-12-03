use anyhow::Result;
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;
use tracing::{debug, info};

#[derive(Clone, Debug)]
pub struct McpServerConfig {
    pub url: String,
    pub auth: Option<String>,
    pub timeout_secs: u64,
}

#[derive(Clone)]
pub struct McpClient {
    config: McpServerConfig,
    client: Client,
}

impl McpClient {
    /// Connect to an MCP server with proper error handling for connection failures.
    pub async fn connect(config: McpServerConfig) -> Result<Self> {
        let mut builder = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .connect_timeout(Duration::from_secs(config.timeout_secs.min(10)));

        if let Some(token) = &config.auth {
            builder = builder.default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    reqwest::header::HeaderValue::from_str(token)
                        .map_err(|e| anyhow::anyhow!("Invalid auth token format: {e}"))?,
                );
                headers
            });
        }

        let client = builder.build().map_err(|e| {
            anyhow::anyhow!(
                "Failed to create HTTP client for MCP server {}: {e}",
                config.url
            )
        })?;

        Ok(Self { config, client })
    }

    /// List available tools from the MCP server with timeout and error handling.
    pub async fn list_tools(&self) -> Result<Value> {
        let url = format!("{}/tools", self.config.url.trim_end_matches('/'));
        debug!("Listing tools from MCP server: {}", url);
        let resp = self.client.get(&url).send().await.map_err(|e| {
            if e.is_timeout() {
                anyhow::anyhow!(
                    "MCP server timeout after {}s: {}",
                    self.config.timeout_secs,
                    url
                )
            } else if e.is_connect() {
                anyhow::anyhow!("Failed to connect to MCP server: {}", url)
            } else {
                anyhow::anyhow!("MCP server request failed: {e}")
            }
        })?;

        let status = resp.status();
        let body = resp
            .json::<Value>()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse MCP server response: {e}"))?;

        if !status.is_success() {
            anyhow::bail!("MCP server returned error status {status}: {body}");
        }
        Ok(body)
    }

    /// Call a tool on the MCP server with timeout and error handling.
    pub async fn call_tool(&self, name: &str, payload: Value) -> Result<Value> {
        let url = format!("{}/tools/{}", self.config.url.trim_end_matches('/'), name);
        info!("Calling MCP tool '{}' at {}", name, url);
        debug!("Tool call payload: {:?}", payload);
        let resp = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    anyhow::anyhow!(
                        "MCP tool call timeout after {}s: {name}",
                        self.config.timeout_secs
                    )
                } else if e.is_connect() {
                    anyhow::anyhow!("Failed to connect to MCP server for tool call: {name}")
                } else {
                    anyhow::anyhow!("MCP tool call request failed: {e}")
                }
            })?;

        let status = resp.status();
        let body = resp
            .json::<Value>()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse MCP tool call response: {e}"))?;

        if !status.is_success() {
            anyhow::bail!("MCP tool call {name} failed with status {status}: {body}");
        }
        Ok(body)
    }
}
