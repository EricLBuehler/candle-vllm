//! Agent Framework Example
//!
//! This example demonstrates how to build an AI agent with tool calling capabilities
//! using candle-vllm. Key features demonstrated:
//!
//! - Tool/function definitions
//! - Tool call parsing from model output
//! - Multi-turn conversation with tool results
//! - Agent execution loop
//! - MCP (Model Context Protocol) integration concepts
//!
//! # Building
//!
//! ```bash
//! cargo build --release --features metal  # For Apple Silicon
//! cargo build --release --features cuda   # For NVIDIA GPUs
//! ```
//!
//! # Architecture
//!
//! The agent loop:
//! 1. User sends a message
//! 2. Model generates response (possibly with tool calls)
//! 3. If tool calls present, execute tools and continue
//! 4. Repeat until model produces final response

use anyhow::Result;
use candle_vllm_openai::{
    get_tool_parser, FunctionDefinition, ParsedOutput, Tool, ToolCall, ToolCallParser,
    ToolConversationBuilder,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use tracing::{info, warn};

// ============================================================================
// Tool Definitions
// ============================================================================

/// Weather tool response
#[derive(Debug, Serialize, Deserialize)]
struct WeatherResult {
    location: String,
    temperature: f32,
    unit: String,
    conditions: String,
}

/// Calculator tool response
#[derive(Debug, Serialize, Deserialize)]
struct CalculatorResult {
    expression: String,
    result: f64,
}

/// Search tool response
#[derive(Debug, Serialize, Deserialize)]
struct SearchResult {
    query: String,
    results: Vec<SearchItem>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchItem {
    title: String,
    snippet: String,
    url: String,
}

// ============================================================================
// Tool Registry
// ============================================================================

/// A tool that can be executed by the agent
pub trait ExecutableTool: Send + Sync {
    /// Get the tool definition for the model
    fn definition(&self) -> Tool;

    /// Execute the tool with given arguments
    fn execute(&self, arguments: &str) -> Result<String>;
}

/// Weather tool implementation
struct WeatherTool;

impl ExecutableTool for WeatherTool {
    fn definition(&self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some(
                    "Get the current weather for a location. Returns temperature and conditions."
                        .to_string(),
                ),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country, e.g., 'Paris, France'"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                })),
                strict: None,
            },
        }
    }

    fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let location = args["location"].as_str().unwrap_or("Unknown");
        let unit = args["unit"].as_str().unwrap_or("celsius");

        // Mock weather data
        let (temp, temp_unit) = match unit {
            "fahrenheit" => (72.0, "°F"),
            _ => (22.0, "°C"),
        };

        let result = WeatherResult {
            location: location.to_string(),
            temperature: temp,
            unit: temp_unit.to_string(),
            conditions: "Partly cloudy".to_string(),
        };

        Ok(serde_json::to_string(&result)?)
    }
}

/// Calculator tool implementation
struct CalculatorTool;

impl ExecutableTool for CalculatorTool {
    fn definition(&self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "calculator".to_string(),
                description: Some(
                    "Perform mathematical calculations. Supports basic arithmetic operations."
                        .to_string(),
                ),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate, e.g., '2 + 3 * 4'"
                        }
                    },
                    "required": ["expression"]
                })),
                strict: None,
            },
        }
    }

    fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let expression = args["expression"].as_str().unwrap_or("0");

        // Simple evaluation (in production, use a proper expression parser)
        // For demo, we just handle basic cases
        let result = evaluate_simple_expression(expression);

        let calc_result = CalculatorResult {
            expression: expression.to_string(),
            result,
        };

        Ok(serde_json::to_string(&calc_result)?)
    }
}

fn evaluate_simple_expression(expr: &str) -> f64 {
    // Very simple expression evaluator for demo purposes
    // In production, use a proper math expression parser
    let expr = expr.replace(' ', "");

    // Handle simple addition
    if let Some(idx) = expr.rfind('+') {
        if idx > 0 {
            let left = evaluate_simple_expression(&expr[..idx]);
            let right = evaluate_simple_expression(&expr[idx + 1..]);
            return left + right;
        }
    }

    // Handle simple subtraction
    if let Some(idx) = expr.rfind('-') {
        if idx > 0 {
            let left = evaluate_simple_expression(&expr[..idx]);
            let right = evaluate_simple_expression(&expr[idx + 1..]);
            return left - right;
        }
    }

    // Handle simple multiplication
    if let Some(idx) = expr.find('*') {
        let left = evaluate_simple_expression(&expr[..idx]);
        let right = evaluate_simple_expression(&expr[idx + 1..]);
        return left * right;
    }

    // Handle simple division
    if let Some(idx) = expr.find('/') {
        let left = evaluate_simple_expression(&expr[..idx]);
        let right = evaluate_simple_expression(&expr[idx + 1..]);
        if right != 0.0 {
            return left / right;
        }
    }

    // Parse as number
    expr.parse().unwrap_or(0.0)
}

/// Search tool implementation
struct SearchTool;

impl ExecutableTool for SearchTool {
    fn definition(&self) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "search".to_string(),
                description: Some(
                    "Search the web for information on a given topic.".to_string(),
                ),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                })),
                strict: None,
            },
        }
    }

    fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let query = args["query"].as_str().unwrap_or("unknown");

        // Mock search results
        let result = SearchResult {
            query: query.to_string(),
            results: vec![
                SearchItem {
                    title: format!("Wikipedia: {}", query),
                    snippet: format!("{} is a topic with many interesting aspects...", query),
                    url: format!("https://en.wikipedia.org/wiki/{}", query.replace(' ', "_")),
                },
                SearchItem {
                    title: format!("{} - Latest News", query),
                    snippet: format!("Recent developments in {}...", query),
                    url: "https://news.example.com".to_string(),
                },
            ],
        };

        Ok(serde_json::to_string(&result)?)
    }
}

// ============================================================================
// Tool Registry
// ============================================================================

/// Registry of available tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn ExecutableTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };

        // Register default tools
        registry.register(Box::new(WeatherTool));
        registry.register(Box::new(CalculatorTool));
        registry.register(Box::new(SearchTool));

        registry
    }

    pub fn register(&mut self, tool: Box<dyn ExecutableTool>) {
        let name = tool.definition().function.name.clone();
        self.tools.insert(name, tool);
    }

    pub fn get_definitions(&self) -> Vec<Tool> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    pub fn execute(&self, name: &str, arguments: &str) -> Result<String> {
        match self.tools.get(name) {
            Some(tool) => tool.execute(arguments),
            None => anyhow::bail!("Unknown tool: {}", name),
        }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Agent
// ============================================================================

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Model name for tool parsing
    pub model_name: String,
    /// Maximum number of tool call iterations
    pub max_iterations: usize,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model_name: "mistral".to_string(),
            max_iterations: 5,
            verbose: true,
        }
    }
}

/// AI Agent with tool calling capabilities
pub struct Agent {
    config: AgentConfig,
    tool_registry: ToolRegistry,
    tool_parser: Box<dyn ToolCallParser>,
    conversation: ToolConversationBuilder,
}

impl Agent {
    pub fn new(config: AgentConfig) -> Self {
        let tool_parser = get_tool_parser(&config.model_name);
        let conversation = ToolConversationBuilder::new(&config.model_name);

        Self {
            config,
            tool_registry: ToolRegistry::new(),
            tool_parser,
            conversation,
        }
    }

    /// Get available tool definitions
    pub fn get_tools(&self) -> Vec<Tool> {
        self.tool_registry.get_definitions()
    }

    /// Process user input and execute the agent loop
    pub fn process(&mut self, user_input: &str) -> Result<AgentResponse> {
        info!("User: {}", user_input);
        self.conversation.add_user_message(user_input);

        let mut iterations = 0;
        let mut all_tool_calls = Vec::new();
        let mut final_response = String::new();

        while iterations < self.config.max_iterations {
            iterations += 1;
            info!("Agent iteration {}", iterations);

            // Simulate model response (in production, call the actual model)
            let model_output = self.simulate_model_response(user_input, iterations);
            info!("Model output: {}", model_output);

            // Parse for tool calls
            let parsed = self.tool_parser.parse(&model_output);

            match parsed {
                ParsedOutput::ToolCalls(tool_calls) => {
                    info!("Found {} tool call(s)", tool_calls.len());
                    let api_calls = tool_calls
                        .into_iter()
                        .map(|tc| tc.to_tool_call())
                        .collect::<Vec<_>>();

                    // Execute tools and collect results
                    for call in &api_calls {
                        info!("Executing tool: {} with args: {}", call.function.name, call.function.arguments);
                        
                        match self.tool_registry.execute(&call.function.name, &call.function.arguments) {
                            Ok(result) => {
                                info!("Tool result: {}", result);
                                self.conversation.add_tool_result(
                                    &call.id,
                                    &result,
                                    Some(call.function.name.clone()),
                                );
                            }
                            Err(e) => {
                                warn!("Tool error: {}", e);
                                self.conversation.add_tool_result(
                                    &call.id,
                                    &format!("Error: {}", e),
                                    Some(call.function.name.clone()),
                                );
                            }
                        }
                    }

                    all_tool_calls.extend(api_calls);
                }
                ParsedOutput::Mixed { text, tool_calls } => {
                    info!("Found text and {} tool call(s)", tool_calls.len());
                    final_response = text;

                    let api_calls = tool_calls
                        .into_iter()
                        .map(|tc| tc.to_tool_call())
                        .collect::<Vec<_>>();

                    for call in &api_calls {
                        match self.tool_registry.execute(&call.function.name, &call.function.arguments) {
                            Ok(result) => {
                                self.conversation.add_tool_result(
                                    &call.id,
                                    &result,
                                    Some(call.function.name.clone()),
                                );
                            }
                            Err(e) => {
                                self.conversation.add_tool_result(
                                    &call.id,
                                    &format!("Error: {}", e),
                                    Some(call.function.name.clone()),
                                );
                            }
                        }
                    }

                    all_tool_calls.extend(api_calls);
                }
                ParsedOutput::Text(text) => {
                    // No tool calls, this is the final response
                    info!("Final response (no tool calls)");
                    final_response = text;
                    break;
                }
            }
        }

        self.conversation.add_assistant_response(&final_response);

        Ok(AgentResponse {
            content: final_response,
            tool_calls: all_tool_calls,
            iterations,
        })
    }

    /// Simulate model response (for demo purposes)
    /// In production, this would call the actual model via OpenAIAdapter
    fn simulate_model_response(&self, user_input: &str, iteration: usize) -> String {
        let input_lower = user_input.to_lowercase();

        // On first iteration, generate tool calls if relevant keywords found
        if iteration == 1 {
            if input_lower.contains("weather") {
                // Extract location from input or use default
                let location = if input_lower.contains("paris") {
                    "Paris, France"
                } else if input_lower.contains("tokyo") {
                    "Tokyo, Japan"
                } else if input_lower.contains("new york") {
                    "New York, USA"
                } else {
                    "London, UK"
                };

                return format!(
                    r#"[TOOL_CALLS] [{{"name": "get_weather", "arguments": {{"location": "{}"}}}}]"#,
                    location
                );
            }

            if input_lower.contains("calculate") || input_lower.contains("math") || input_lower.contains("compute") {
                // Try to extract expression
                let expression = if let Some(expr) = extract_math_expression(&input_lower) {
                    expr
                } else {
                    "2 + 2".to_string()
                };

                return format!(
                    r#"[TOOL_CALLS] [{{"name": "calculator", "arguments": {{"expression": "{}"}}}}]"#,
                    expression
                );
            }

            if input_lower.contains("search") || input_lower.contains("find") || input_lower.contains("look up") {
                let query = input_lower
                    .replace("search", "")
                    .replace("find", "")
                    .replace("look up", "")
                    .replace("for", "")
                    .trim()
                    .to_string();

                return format!(
                    r#"[TOOL_CALLS] [{{"name": "search", "arguments": {{"query": "{}"}}}}]"#,
                    if query.is_empty() { "AI agents" } else { &query }
                );
            }
        }

        // Generate final response based on conversation context
        format!(
            "Based on the information gathered, here's my response:\n\n\
             I've processed your request about '{}'. \
             In a real implementation, this would be the model's generated response \
             after incorporating any tool call results.\n\n\
             The agent completed in {} iteration(s).",
            user_input, iteration
        )
    }
}

/// Extract a simple math expression from user input
fn extract_math_expression(input: &str) -> Option<String> {
    // Simple extraction - look for numbers and operators
    let chars: Vec<char> = input.chars().collect();
    let mut expr = String::new();
    let mut in_expr = false;

    for c in chars {
        if c.is_numeric() || c == '.' {
            in_expr = true;
            expr.push(c);
        } else if in_expr && (c == '+' || c == '-' || c == '*' || c == '/' || c == ' ') {
            expr.push(c);
        } else if in_expr && !c.is_whitespace() {
            // End of expression
            break;
        }
    }

    let trimmed = expr.trim().to_string();
    if trimmed.len() > 2 {
        Some(trimmed)
    } else {
        None
    }
}

/// Agent response
#[derive(Debug)]
pub struct AgentResponse {
    /// The final text response
    pub content: String,
    /// All tool calls made during processing
    pub tool_calls: Vec<ToolCall>,
    /// Number of iterations taken
    pub iterations: usize,
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("candle_vllm_agent_framework_example=info".parse().unwrap()),
        )
        .init();

    info!("candle-vllm Agent Framework Example");
    info!("====================================");
    info!("");

    // Create agent
    let config = AgentConfig::default();
    let mut agent = Agent::new(config);

    // Show available tools
    let tools = agent.get_tools();
    info!("Available tools:");
    for tool in &tools {
        info!("  - {} : {}", 
            tool.function.name,
            tool.function.description.as_deref().unwrap_or("No description")
        );
    }
    info!("");

    // Demo: Weather query
    info!("=== Demo 1: Weather Query ===");
    let response = agent.process("What's the weather like in Paris?")?;
    info!("Response: {}", response.content);
    info!("Tool calls made: {}", response.tool_calls.len());
    info!("");

    // Demo: Calculator
    info!("=== Demo 2: Calculator ===");
    let mut agent2 = Agent::new(AgentConfig::default());
    let response = agent2.process("Calculate 15 * 7 + 23")?;
    info!("Response: {}", response.content);
    info!("");

    // Demo: Search
    info!("=== Demo 3: Search ===");
    let mut agent3 = Agent::new(AgentConfig::default());
    let response = agent3.process("Search for Rust programming language")?;
    info!("Response: {}", response.content);
    info!("");

    // Demo: No tool call needed
    info!("=== Demo 4: Simple Question (No Tools) ===");
    let mut agent4 = Agent::new(AgentConfig::default());
    let response = agent4.process("Hello, how are you?")?;
    info!("Response: {}", response.content);
    info!("");

    info!("=== Integration Notes ===");
    info!("");
    info!("To integrate with a real model:");
    info!("1. Use OpenAIAdapter.chat_completion() with tools parameter");
    info!("2. Parse response for tool calls using get_tool_parser()");
    info!("3. Execute tools and add results to conversation");
    info!("4. Continue until model returns final response");
    info!("");
    info!("Example with real model:");
    info!("```rust");
    info!("let adapter = OpenAIAdapter::new(engine);");
    info!("let request = ChatCompletionRequest {{");
    info!("    model: \"mistral\".to_string(),");
    info!("    messages: Messages::Chat(messages),");
    info!("    tools: Some(tool_definitions),");
    info!("    ..Default::default()");
    info!("}};");
    info!("let response = adapter.chat_completion(request).await?;");
    info!("```");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weather_tool() {
        let tool = WeatherTool;
        let result = tool.execute(r#"{"location": "Paris, France"}"#).unwrap();
        let parsed: WeatherResult = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.location, "Paris, France");
    }

    #[test]
    fn test_calculator_tool() {
        let tool = CalculatorTool;
        let result = tool.execute(r#"{"expression": "2 + 3"}"#).unwrap();
        let parsed: CalculatorResult = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.result, 5.0);
    }

    #[test]
    fn test_calculator_multiplication() {
        let tool = CalculatorTool;
        let result = tool.execute(r#"{"expression": "4 * 5"}"#).unwrap();
        let parsed: CalculatorResult = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.result, 20.0);
    }

    #[test]
    fn test_search_tool() {
        let tool = SearchTool;
        let result = tool.execute(r#"{"query": "Rust programming"}"#).unwrap();
        let parsed: SearchResult = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.query, "Rust programming");
        assert!(!parsed.results.is_empty());
    }

    #[test]
    fn test_tool_registry() {
        let registry = ToolRegistry::new();
        let definitions = registry.get_definitions();
        assert_eq!(definitions.len(), 3);

        let names: Vec<_> = definitions.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"get_weather"));
        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"search"));
    }

    #[test]
    fn test_simple_expression_eval() {
        assert_eq!(evaluate_simple_expression("2 + 3"), 5.0);
        assert_eq!(evaluate_simple_expression("10 - 4"), 6.0);
        assert_eq!(evaluate_simple_expression("3 * 4"), 12.0);
        assert_eq!(evaluate_simple_expression("20 / 5"), 4.0);
    }
}

