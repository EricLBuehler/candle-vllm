# MCP Tool Calling in candle-vllm

`candle-vllm` now supports the Model Context Protocol (MCP) and OpenAI-compatible tool calling. This allows Large Language Models (LLMs) served by `candle-vllm` to interact with external tools and resources.

## Features

- **OpenAI-Compatible Tool Calling**: Parse and execute tool calls using standard OpenAI API format.
- **MCP Client**: Connect to MCP servers to dynamically load tools and resources.
- **Multiple MCP Servers**: Configure multiple MCP servers via a configuration file or CLI arguments.
- **Prompt Injection**: Automatically injects available tools into the system prompt.

## Usage

### 1. Enable Tool Calling in Requests

When making a request to `/v1/chat/completions`, you can include `tools` definitions manually, or rely on the server to inject MCP tools if configured.

**Proprietary Tools (OpenAI Style):**
```json
{
  "model": "qwen2.5-7b",
  "messages": [{"role": "user", "content": "What's the weather?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": { ... }
      }
    }
  ]
}
```

**MCP Tools:**
If you have configured MCP servers, `candle-vllm` will automatically fetch available tools and inject them into the system prompt. The model will then be able to generate tool calls for them.

### 2. Handling Tool Calls

The server will return tool calls in the response:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_123",
            "type": "function",
            "function": {
              "name": "filesystem_read_file",
              "arguments": "{\"path\": \"/home/user/workspace/README.md\"}"
            }
          }
        ]
      }
    }
  ]
}
```

**Note:** Currently, `candle-vllm` acts as an OpenAI-compatible *inference server*. It parses tool calls but does typically not execute them automatically (unless configured as an agent, which is experimental). The client application is responsible for executing the tool call and sending the result back to the model if using the standard OpenAI flow.

However, the internal `McpClientManager` facilitates checking tools and could be extended for server-side execution in future updates.

## Supported Models

Tool calling relies on the model's ability to output structured calls. We support:
- OpenAI-compatible (XML/JSON format, with `<tool_call>` and `</tool_call>` flags, and `tool_calls` response chunk)

## TODO

Internal MCP tool call execution (like vLLM.rs)

### Configuration

#### CLI Arguments

You can connect to a local MCP server (stdio transport) using CLI arguments:

```bash
cargo run --release -- --port 2000 --mcp-command "npx" --mcp-args "-y @modelcontextprotocol/server-filesystem /path/to/allow"
```

- `--mcp-command`: The command to run the MCP server (e.g., `npx`, `uvx`, `python`).
- `--mcp-args`: Arguments for the command.

#### Configuration File

For more complex setups, use a configuration file (JSON):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/workspace"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
      }
    }
  }
}
```

Run with:
```bash
cargo run --release -- --port 2000 --mcp-config mcp_config.json
```