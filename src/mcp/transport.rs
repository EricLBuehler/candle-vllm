// src/mcp/transport.rs
//! MCP transport layer implementations
//!
//! Supports stdio (for local processes) and HTTP/SSE (for remote servers)

use super::types::*;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Transport trait for sending and receiving MCP messages
pub trait Transport: Send + Sync {
    /// Send a message
    fn send(&mut self, message: &str) -> Result<(), TransportError>;

    /// Receive a message (blocking)
    fn receive(&mut self) -> Result<String, TransportError>;

    /// Close the transport
    fn close(&mut self) -> Result<(), TransportError>;
}

/// Transport errors
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Process error: {0}")]
    Process(String),

    #[error("Connection closed")]
    Closed,

    #[error("Timeout")]
    Timeout,

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("HTTP error: {0}")]
    Http(String),
}

/// Stdio transport for communicating with local MCP server processes
pub struct StdioTransport {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout_reader: Option<BufReader<ChildStdout>>,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning a process
    pub fn spawn(command: &str, args: &[&str]) -> Result<Self, TransportError> {
        Self::spawn_with_env(command, args, &std::collections::HashMap::new())
    }

    /// Create a new stdio transport with additional environment variables
    pub fn spawn_with_env(
        command: &str,
        args: &[&str],
        env: &std::collections::HashMap<String, String>,
    ) -> Result<Self, TransportError> {
        let mut child = Command::new(command)
            .args(args)
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stdout_reader = stdout.map(BufReader::new);

        Ok(Self {
            child,
            stdin,
            stdout_reader,
        })
    }
}

impl Transport for StdioTransport {
    fn send(&mut self, message: &str) -> Result<(), TransportError> {
        if let Some(ref mut stdin) = self.stdin {
            writeln!(stdin, "{}", message)?;
            stdin.flush()?;
            Ok(())
        } else {
            Err(TransportError::Closed)
        }
    }

    fn receive(&mut self) -> Result<String, TransportError> {
        if let Some(ref mut reader) = self.stdout_reader {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(TransportError::Closed);
            }
            Ok(line.trim().to_string())
        } else {
            Err(TransportError::Closed)
        }
    }

    fn close(&mut self) -> Result<(), TransportError> {
        self.stdin = None;
        self.stdout_reader = None;
        let _ = self.child.kill();
        Ok(())
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// In-memory transport for testing (thread-safe via crossbeam channels)
pub struct MemoryTransport {
    tx: crossbeam::channel::Sender<String>,
    rx: crossbeam::channel::Receiver<String>,
}

impl MemoryTransport {
    /// Create a pair of connected transports for testing
    pub fn pair() -> (Self, Self) {
        let (tx1, rx1) = crossbeam::channel::unbounded();
        let (tx2, rx2) = crossbeam::channel::unbounded();

        (Self { tx: tx1, rx: rx2 }, Self { tx: tx2, rx: rx1 })
    }
}

impl Transport for MemoryTransport {
    fn send(&mut self, message: &str) -> Result<(), TransportError> {
        self.tx
            .send(message.to_string())
            .map_err(|_| TransportError::Closed)
    }

    fn receive(&mut self) -> Result<String, TransportError> {
        self.rx.recv().map_err(|_| TransportError::Closed)
    }

    fn close(&mut self) -> Result<(), TransportError> {
        Ok(())
    }
}

/// HTTP transport for communicating with remote MCP servers via HTTP/SSE
/// Supports both "Streamable HTTP" (newer) and "HTTP+SSE" (older) transport modes.
pub struct HttpTransport {
    client: reqwest::blocking::Client,
    base_url: String,
    post_url: String, // URL for POSTing messages (may differ from base_url for HTTP+SSE)
    headers: std::collections::HashMap<String, String>,
    response_buffer: std::collections::VecDeque<String>,
    initialized: bool,
    session_id: Option<String>, // Mcp-Session-Id from server for subsequent requests
}

impl HttpTransport {
    /// Create a new HTTP transport for a remote MCP server
    pub fn new(
        url: impl Into<String>,
        headers: std::collections::HashMap<String, String>,
    ) -> Result<Self, TransportError> {
        let base_url = url.into();
        tracing::info!("Connecting to remote MCP server at: {}", base_url);

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| TransportError::Http(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            post_url: base_url.clone(), // Initially same as base_url, may change after SSE init
            base_url,
            headers,
            response_buffer: std::collections::VecDeque::new(),
            initialized: false,
            session_id: None,
        })
    }

    /// Try to initialize using older HTTP+SSE transport (GET the endpoint URL)
    /// Per MCP spec: Issue a GET request to the server URL, expecting SSE stream
    /// with an endpoint event as the first event.
    fn try_sse_init(&mut self) -> Result<(), TransportError> {
        // GET the same URL (not /sse) - per MCP spec backwards compatibility
        let mut request = self
            .client
            .get(&self.base_url)
            .header("Accept", "text/event-stream");

        // Add custom headers
        for (key, value) in &self.headers {
            request = request.header(key.as_str(), value.as_str());
        }

        let response = request
            .send()
            .map_err(|e| TransportError::Http(format!("SSE connection failed: {e}")))?;

        if !response.status().is_success() {
            return Err(TransportError::Http(format!(
                "SSE init failed: {} {}",
                response.status().as_u16(),
                response.status().canonical_reason().unwrap_or("Unknown")
            )));
        }

        // Parse SSE response to find endpoint event
        let response_text = response
            .text()
            .map_err(|e| TransportError::Http(format!("Failed to read SSE response: {e}")))?;

        // Parse SSE events to find the endpoint event
        // Format: "event: endpoint\ndata: /messages?session_id=xxx\n\n"
        let mut found_endpoint_event = false;
        for line in response_text.lines() {
            if line.starts_with("event: endpoint") {
                found_endpoint_event = true;
                continue;
            }
            if line.starts_with("data: ") && found_endpoint_event {
                let endpoint = line[6..].trim();
                // The endpoint is a relative or absolute URL for POST
                if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
                    self.post_url = endpoint.to_string();
                } else if endpoint.starts_with('/') {
                    // Relative URL - combine with base (extract scheme://host)
                    if let Some(idx) = self.base_url.find("://") {
                        let after_scheme = &self.base_url[idx + 3..];
                        let host_end = after_scheme.find('/').unwrap_or(after_scheme.len());
                        let base = &self.base_url[..idx + 3 + host_end];
                        self.post_url = format!("{}{}", base, endpoint);
                    } else {
                        self.post_url =
                            format!("{}{}", self.base_url.trim_end_matches('/'), endpoint);
                    }
                } else {
                    self.post_url = format!("{}/{}", self.base_url.trim_end_matches('/'), endpoint);
                }
                self.initialized = true;
                return Ok(());
            }
        }

        Err(TransportError::Http(
            "SSE init: no endpoint event received".to_string(),
        ))
    }

    /// Parse SSE event data from response text
    /// Only extracts JSON-RPC messages (starting with '{'), filtering out ping and other non-JSON events
    fn parse_sse_events(text: &str) -> Vec<String> {
        let mut events = Vec::new();
        let mut current_data = String::new();

        for line in text.lines() {
            if line.starts_with("data: ") {
                current_data.push_str(&line[6..]);
            } else if line.is_empty() && !current_data.is_empty() {
                // End of event - only add if it looks like JSON (starts with '{')
                let trimmed = current_data.trim();
                if trimmed.starts_with('{') {
                    events.push(std::mem::take(&mut current_data));
                } else {
                    // Skip non-JSON events like 'ping'
                    current_data.clear();
                }
            }
            // Ignore "event:" lines and other SSE metadata
        }
        // Handle case where there's no trailing newline
        if !current_data.is_empty() {
            let trimmed = current_data.trim();
            if trimmed.starts_with('{') {
                events.push(current_data);
            }
        }

        events
    }
}

impl Transport for HttpTransport {
    fn send(&mut self, message: &str) -> Result<(), TransportError> {
        // First request: try Streamable HTTP, fall back to HTTP+SSE if needed
        if !self.initialized {
            // Try Streamable HTTP (direct POST)
            let mut request = self
                .client
                .post(&self.post_url)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream");

            for (key, value) in &self.headers {
                request = request.header(key.as_str(), value.as_str());
            }

            let response = request.body(message.to_string()).send();

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() || status == reqwest::StatusCode::ACCEPTED {
                        self.initialized = true;
                        return self.handle_response(resp);
                    } else if status.is_client_error() {
                        // 4xx error - try HTTP+SSE fallback
                        if self.try_sse_init().is_ok() {
                            // SSE init succeeded, now POST to the endpoint
                            return self.send_post(message);
                        } else {
                            return Err(TransportError::Http(format!(
                                "HTTP error: {} {}",
                                status.as_u16(),
                                status.canonical_reason().unwrap_or("Unknown")
                            )));
                        }
                    } else {
                        return Err(TransportError::Http(format!(
                            "HTTP error: {} {}",
                            status.as_u16(),
                            status.canonical_reason().unwrap_or("Unknown")
                        )));
                    }
                }
                Err(e) => {
                    // Network error - try SSE fallback
                    if self.try_sse_init().is_ok() {
                        return self.send_post(message);
                    }
                    return Err(TransportError::Http(format!("HTTP request failed: {e}")));
                }
            }
        }

        self.send_post(message)
    }

    fn receive(&mut self) -> Result<String, TransportError> {
        self.response_buffer
            .pop_front()
            .ok_or(TransportError::Closed)
    }

    fn close(&mut self) -> Result<(), TransportError> {
        self.response_buffer.clear();
        Ok(())
    }
}

impl HttpTransport {
    fn send_post(&mut self, message: &str) -> Result<(), TransportError> {
        let mut request = self
            .client
            .post(&self.post_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        // Include session ID if we have one (required by some servers)
        if let Some(ref session_id) = self.session_id {
            request = request.header("Mcp-Session-Id", session_id.as_str());
        }

        for (key, value) in &self.headers {
            request = request.header(key.as_str(), value.as_str());
        }

        let response = request
            .body(message.to_string())
            .send()
            .map_err(|e| TransportError::Http(format!("HTTP request failed: {e}")))?;

        self.handle_response(response)
    }

    fn handle_response(
        &mut self,
        response: reqwest::blocking::Response,
    ) -> Result<(), TransportError> {
        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::ACCEPTED {
            return Err(TransportError::Http(format!(
                "HTTP error: {} {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown")
            )));
        }

        // Capture Mcp-Session-Id header if present (required for subsequent requests)
        if let Some(session_id) = response.headers().get("mcp-session-id") {
            if let Ok(session_str) = session_id.to_str() {
                self.session_id = Some(session_str.to_string());
            }
        }

        // For 202 Accepted (notifications/responses), there's no response body
        if status == reqwest::StatusCode::ACCEPTED {
            return Ok(());
        }

        // Check content type to determine how to parse response
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_default();

        let response_text = response
            .text()
            .map_err(|e| TransportError::Http(format!("Failed to read response: {e}")))?;

        if content_type.contains("text/event-stream") {
            // Parse SSE events
            let events = Self::parse_sse_events(&response_text);
            for event in events {
                if !event.is_empty() {
                    self.response_buffer.push_back(event);
                }
            }
        } else {
            // JSON response
            if !response_text.is_empty() {
                self.response_buffer.push_back(response_text);
            }
        }

        Ok(())
    }
}

/// Message framing utilities for MCP over different transports
pub mod framing {
    use super::*;

    /// Encode a JSON-RPC message for line-based transport
    pub fn encode_line(message: &impl serde::Serialize) -> Result<String, TransportError> {
        serde_json::to_string(message).map_err(|e| TransportError::Parse(e.to_string()))
    }

    /// Decode a JSON-RPC message from line-based transport
    pub fn decode_line<T: serde::de::DeserializeOwned>(line: &str) -> Result<T, TransportError> {
        serde_json::from_str(line).map_err(|e| TransportError::Parse(e.to_string()))
    }

    /// Parse any JSON-RPC message (request, response, or notification)
    pub fn parse_message(line: &str) -> Result<McpMessage, TransportError> {
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|e| TransportError::Parse(e.to_string()))?;

        // Check if it's a response (has result or error)
        if value.get("result").is_some() || value.get("error").is_some() {
            let response: JsonRpcResponse =
                serde_json::from_value(value).map_err(|e| TransportError::Parse(e.to_string()))?;
            return Ok(McpMessage::Response(response));
        }

        // Check if it's a notification (no id)
        if value.get("id").is_none() {
            let notification: JsonRpcNotification =
                serde_json::from_value(value).map_err(|e| TransportError::Parse(e.to_string()))?;
            return Ok(McpMessage::Notification(notification));
        }

        // It's a request
        let request: JsonRpcRequest =
            serde_json::from_value(value).map_err(|e| TransportError::Parse(e.to_string()))?;
        Ok(McpMessage::Request(request))
    }
}

/// Parsed MCP message types
#[derive(Debug)]
pub enum McpMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_transport() {
        let (mut t1, mut t2) = MemoryTransport::pair();

        t1.send("hello").unwrap();
        assert_eq!(t2.receive().unwrap(), "hello");

        t2.send("world").unwrap();
        assert_eq!(t1.receive().unwrap(), "world");
    }

    #[test]
    fn test_framing() {
        let req = JsonRpcRequest::new(1i64, "test", None);
        let encoded = framing::encode_line(&req).unwrap();
        let decoded: JsonRpcRequest = framing::decode_line(&encoded).unwrap();

        assert_eq!(decoded.method, "test");
    }

    #[test]
    fn test_parse_message() {
        let request = r#"{"jsonrpc":"2.0","id":1,"method":"test"}"#;
        let response = r#"{"jsonrpc":"2.0","id":1,"result":{}}"#;
        let notification = r#"{"jsonrpc":"2.0","method":"notify"}"#;

        assert!(matches!(
            framing::parse_message(request).unwrap(),
            McpMessage::Request(_)
        ));
        assert!(matches!(
            framing::parse_message(response).unwrap(),
            McpMessage::Response(_)
        ));
        assert!(matches!(
            framing::parse_message(notification).unwrap(),
            McpMessage::Notification(_)
        ));
    }

    #[test]
    fn test_http_transport_sse_parsing() {
        // Test SSE event parsing with single event
        let sse_single = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n";
        let events = HttpTransport::parse_sse_events(sse_single);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], r#"{"jsonrpc":"2.0","id":1,"result":{}}"#);

        // Test SSE event parsing with multiple events
        let sse_multi = "data: {\"jsonrpc\":\"2.0\",\"method\":\"notify1\"}\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n";
        let events = HttpTransport::parse_sse_events(sse_multi);
        assert_eq!(events.len(), 2);

        // Test SSE event without trailing newline
        let sse_no_newline = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}";
        let events = HttpTransport::parse_sse_events(sse_no_newline);
        assert_eq!(events.len(), 1);

        // Test empty input
        let events = HttpTransport::parse_sse_events("");
        assert!(events.is_empty());
    }

    #[test]
    fn test_http_transport_new() {
        let headers = std::collections::HashMap::from([(
            "Authorization".to_string(),
            "Bearer token".to_string(),
        )]);
        let transport = HttpTransport::new("https://example.com/mcp", headers);
        assert!(transport.is_ok());
    }
}
