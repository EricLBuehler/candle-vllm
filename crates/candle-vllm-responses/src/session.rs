use crate::mcp_client::{McpClient, McpServerConfig};
use crate::orchestrator::Orchestrator;
use candle_vllm_core::api::InferenceEngine;
use candle_vllm_core::openai::requests::{ChatCompletionRequest, ChatMessage, Messages, Tool};
use candle_vllm_openai::adapter::OpenAIAdapter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use tracing::warn;

#[derive(Clone, Debug, Default)]
pub struct ConversationOptions {
    pub max_turns: usize,
    pub allowed_tools: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default)]
pub struct SessionConfig {
    pub model_path: Option<String>,
    pub device: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpConfigFile {
    #[serde(default)]
    servers: Vec<McpConfigEntry>,
    // Support mcpServers format (object with server names as keys)
    #[serde(default, rename = "mcpServers")]
    mcp_servers: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize)]
struct McpConfigEntry {
    name: String,
    url: String,
    auth: Option<String>,
    timeout_secs: Option<u64>,
    #[allow(dead_code)]
    instructions: Option<String>,
    // Support command-based format
    #[serde(rename = "type")]
    _type: Option<String>,
    command: Option<String>,
    #[allow(dead_code)]
    args: Option<Vec<String>>,
    #[allow(dead_code)]
    env: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Result of a multi-turn conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationResult {
    /// The final message from the assistant
    pub final_message: String,
    /// All tool calls made during the conversation
    pub tool_calls: Vec<candle_vllm_core::openai::requests::ToolCall>,
    /// Number of turns taken
    pub turns_taken: usize,
    /// Whether the conversation completed successfully
    pub completed: bool,
}

pub struct ResponsesSession {
    mcp_clients: HashMap<String, McpClient>,
    adapter: Option<OpenAIAdapter>,
}

impl ResponsesSession {
    /// Create a new session without an inference engine (MCP-only).
    pub async fn new(_config: SessionConfig) -> anyhow::Result<Self> {
        Ok(Self {
            mcp_clients: HashMap::new(),
            adapter: None,
        })
    }

    /// Create a new session with an inference engine.
    pub async fn with_engine(
        engine: InferenceEngine,
        _config: SessionConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            mcp_clients: HashMap::new(),
            adapter: Some(OpenAIAdapter::new(engine)),
        })
    }

    /// Create a session by loading MCP servers from a JSON file (mcp.json).
    pub async fn from_config_path(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: McpConfigFile = serde_json::from_str(&content)?;
        Self::from_config(config).await
    }

    /// Create a session from an already-parsed config value.
    pub async fn from_config_value(value: &Value) -> anyhow::Result<Self> {
        let config: McpConfigFile = serde_json::from_value(value.clone())?;
        Self::from_config(config).await
    }

    pub async fn from_config(config: McpConfigFile) -> anyhow::Result<Self> {
        let mut clients = HashMap::new();
        
        // Process servers array format
        for entry in config.servers {
            let timeout = entry.timeout_secs.unwrap_or(30).max(1);
            
            // If URL is provided, use it directly
            // Otherwise, if command is provided, convert to HTTP URL (assumes local server)
            let url = if !entry.url.is_empty() {
                entry.url
            } else if let Some(_command) = entry.command {
                // For command-based servers, assume they expose HTTP on localhost
                let default_url = format!("http://localhost:3000/{}", entry.name);
                warn!(
                    "MCP server '{}' uses command format. Converting to HTTP URL: {}",
                    entry.name, default_url
                );
                default_url
            } else {
                anyhow::bail!("MCP server '{}' must have either 'url' or 'command'", entry.name);
            };
            
            let client = McpClient::connect(McpServerConfig {
                url,
                auth: entry.auth.clone(),
                timeout_secs: timeout,
            })
            .await?;
            clients.insert(entry.name.clone(), client);
        }
        
        // Process mcpServers object format
        if let Some(mcp_servers) = config.mcp_servers {
            for (name, entry_value) in mcp_servers {
                // Try to deserialize as McpConfigEntry
                if let Ok(mut entry) = serde_json::from_value::<McpConfigEntry>(entry_value.clone()) {
                    // Set name if not already set
                    if entry.name.is_empty() {
                        entry.name = name.clone();
                    }
                    
                    let timeout = entry.timeout_secs.unwrap_or(30).max(1);
                    let url = if !entry.url.is_empty() {
                        entry.url
                    } else if let Some(_command) = entry.command {
                        let default_url = format!("http://localhost:3000/{}", name);
                        warn!(
                            "MCP server '{}' uses command format. Converting to HTTP URL: {}",
                            name, default_url
                        );
                        default_url
                    } else {
                        anyhow::bail!("MCP server '{}' must have either 'url' or 'command'", name);
                    };
                    
                    let client = McpClient::connect(McpServerConfig {
                        url,
                        auth: entry.auth.clone(),
                        timeout_secs: timeout,
                    })
                    .await?;
                    clients.insert(name, client);
                }
            }
        }
        
        Ok(Self {
            mcp_clients: clients,
            adapter: None,
        })
    }

    /// List available MCP tools across servers and convert to OpenAI tool format.
    pub async fn list_openai_tools(
        &self,
        allowed_tools: Option<Vec<String>>,
    ) -> anyhow::Result<Vec<Tool>> {
        let mut tools = Vec::new();
        for (name, client) in &self.mcp_clients {
            let mcp_tools = client.list_tools().await?;
            let list = extract_tools_array(&mcp_tools);
            let mut converted = Tool::from_mcp_list(&list);
            if let Some(allowed) = &allowed_tools {
                let allowed_refs: Vec<&str> = allowed.iter().map(|s| s.as_str()).collect();
                converted = Tool::filter_by_names(&converted, &allowed_refs);
            }
            // prefix tool names with server for disambiguation
            for tool in converted.iter_mut() {
                tool.function.name = format!("{}::{}", name, tool.function.name);
            }
            tools.extend(converted);
        }
        Ok(tools)
    }

    /// Add an MCP server to this session.
    pub async fn add_mcp_server(
        &mut self,
        name: String,
        config: McpServerConfig,
    ) -> anyhow::Result<()> {
        let client = McpClient::connect(config).await?;
        self.mcp_clients.insert(name, client);
        Ok(())
    }

    /// Execute a specific tool by server and tool name (un-prefixed).
    pub async fn call_tool(
        &self,
        server: &str,
        tool_name: &str,
        payload: Value,
    ) -> anyhow::Result<Value> {
        let client = self
            .mcp_clients
            .get(server)
            .ok_or_else(|| anyhow::anyhow!("MCP server '{server}' not found"))?;
        client.call_tool(tool_name, payload).await
    }

    /// Execute MCP tool calls using the orchestrator.
    pub async fn execute_mcp_tool_calls(
        &self,
        tool_calls: &[candle_vllm_core::openai::requests::ToolCall],
    ) -> anyhow::Result<Vec<ChatMessage>> {
        let orchestrator = Orchestrator::new(self.mcp_clients.clone());
        orchestrator.execute_tool_calls(tool_calls, None).await
    }

    /// Run a multi-turn conversation with automatic tool execution.
    ///
    /// This method:
    /// 1. Starts with the initial messages
    /// 2. Generates a response using the inference engine
    /// 3. If tool calls are present, executes them and continues
    /// 4. Repeats until max_turns or no tool calls
    /// 5. Returns the final result
    pub async fn run_conversation(
        &mut self,
        initial_messages: Vec<ChatMessage>,
        options: ConversationOptions,
    ) -> anyhow::Result<ConversationResult> {
        // Clone MCP clients for orchestrator (needed because we need mutable access to adapter)
        let mcp_clients = self.mcp_clients.clone();
        let orchestrator = Orchestrator::new(mcp_clients);

        // Get available tools before borrowing adapter
        let tools = self
            .list_openai_tools(options.allowed_tools.clone())
            .await?;
        let allowed_refs: Vec<String> = options
            .allowed_tools
            .clone()
            .unwrap_or_else(|| tools.iter().map(|t| t.function.name.clone()).collect());

        let adapter = self
            .adapter
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Inference engine not configured. Use ResponsesSessionBuilder to set up the engine."))?;

        let mut messages = initial_messages;
        let mut all_tool_calls = Vec::new();
        let mut turns = 0;
        let max_turns = options.max_turns.max(1);

        loop {
            if turns >= max_turns {
                return Ok(ConversationResult {
                    final_message: "Conversation reached max_turns limit".to_string(),
                    tool_calls: all_tool_calls,
                    turns_taken: turns,
                    completed: false,
                });
            }

            turns += 1;

            // Create chat completion request
            let request = ChatCompletionRequest {
                model: "local".to_string(),
                messages: Messages::Chat(messages.clone()),
                tools: Some(tools.clone()),
                tool_choice: None,
                temperature: None,
                top_p: None,
                top_k: None,
                min_p: None,
                max_tokens: None,
                stop: None,
                stream: None,
                n: None,
                presence_penalty: None,
                frequency_penalty: None,
                repeat_last_n: None,
                logit_bias: None,
                user: None,
                best_of: None,
                use_beam_search: None,
                ignore_eos: None,
                skip_special_tokens: None,
                stop_token_ids: None,
                logprobs: None,
                thinking: None,
                parallel_tool_calls: None,
                conversation_id: None,
                resource_id: None,
            };

            // Generate response
            let response = adapter
                .chat_completion(request)
                .await
                .map_err(|e| anyhow::anyhow!("Chat completion failed: {e}"))?;

            // Extract assistant message
            let choice = response
                .choices
                .first()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

            // Convert ChatChoiceData to ChatMessage
            let assistant_msg = ChatMessage {
                role: choice.message.role.clone(),
                content: choice.message.content.clone(),
                tool_calls: choice.message.tool_calls.clone(),
                tool_call_id: None,
                name: None,
            };
            messages.push(assistant_msg.clone());

            // Check for tool calls
            if let Some(ref tool_calls) = assistant_msg.tool_calls {
                if tool_calls.is_empty() {
                    // No tool calls, conversation complete
                    return Ok(ConversationResult {
                        final_message: assistant_msg.content.unwrap_or_default(),
                        tool_calls: all_tool_calls,
                        turns_taken: turns,
                        completed: true,
                    });
                }

                // Execute tool calls
                all_tool_calls.extend(tool_calls.clone());
                let tool_responses = orchestrator
                    .execute_tool_calls(tool_calls, Some(&allowed_refs))
                    .await?;

                // Add tool responses to conversation
                messages.extend(tool_responses);
            } else {
                // No tool calls, conversation complete
                return Ok(ConversationResult {
                    final_message: assistant_msg.content.unwrap_or_default(),
                    tool_calls: all_tool_calls,
                    turns_taken: turns,
                    completed: true,
                });
            }
        }
    }
}

/// Builder for ResponsesSession.
pub struct ResponsesSessionBuilder {
    config: SessionConfig,
    mcp_servers: Vec<(String, McpServerConfig)>,
    engine: Option<InferenceEngine>,
}

impl ResponsesSessionBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: SessionConfig::default(),
            mcp_servers: Vec::new(),
            engine: None,
        }
    }

    /// Set the model path.
    pub fn model_path(mut self, path: String) -> Self {
        self.config.model_path = Some(path);
        self
    }

    /// Set the device ordinal.
    pub fn device(mut self, device: usize) -> Self {
        self.config.device = Some(device);
        self
    }

    /// Add an MCP server.
    pub fn add_mcp_server(mut self, name: String, config: McpServerConfig) -> Self {
        self.mcp_servers.push((name, config));
        self
    }

    /// Set the inference engine.
    pub fn engine(mut self, engine: InferenceEngine) -> Self {
        self.engine = Some(engine);
        self
    }

    /// Build the session.
    pub async fn build(self) -> anyhow::Result<ResponsesSession> {
        let mut session = if let Some(engine) = self.engine {
            ResponsesSession::with_engine(engine, self.config).await?
        } else {
            ResponsesSession::new(self.config).await?
        };

        // Add MCP servers
        for (name, config) in self.mcp_servers {
            session.add_mcp_server(name, config).await?;
        }

        Ok(session)
    }
}

impl Default for ResponsesSessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_tools_array(value: &Value) -> Vec<Value> {
    match value {
        Value::Array(arr) => arr.clone(),
        Value::Object(map) => map
            .get("tools")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}
