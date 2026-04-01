# MCP Integration and Tool Calling

`candle-vllm` supports Model Context Protocol (MCP) integration and OpenAI-style tool calling.

Important: `candle-vllm` follows the standard OpenAI tool-calling flow. The server injects tools and parses model-emitted tool calls, but clients are still responsible for executing the tools and sending tool results back in the next request.

## Overview

Implemented capabilities:

- stdio MCP servers
- multiple MCP servers from config
- OpenAI-style `tools` requests
- streaming and non-streaming tool parsing
- tool result validation on follow-up requests

## Tool calling workflow

1. Configure MCP servers with CLI flags or an MCP config file.
2. `candle-vllm` loads tool definitions and injects them into the prompt.
3. The model emits a tool call.
4. The response finishes with `finish_reason="tool_calls"`.
5. The client executes the tool and sends the result back as a `role="tool"` message with the matching `tool_call_id`.

## Request example

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "List files in the current directory"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string"}
          },
          "required": ["path"]
        }
      }
    }
  ]
}
```

## Response shape

When the model calls a tool, the assistant response returns `tool_calls`:

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
              "name": "list_files",
              "arguments": "{\"path\":\".\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

The follow-up tool result must be sent back as:

```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "file1\nfile2\nfile3"
}
```

## Configuration

### CLI

- `--mcp-command`: executable for a single stdio MCP server
- `--mcp-args`: arguments for the MCP server command
- `--mcp-config`: JSON config file for one or more MCP servers
- `--enforce-parser`: override the model-selected tool parser

Single-server example:

```bash
cargo run --release -- --p 8000 \
  --mcp-command npx \
  --mcp-args "-y @modelcontextprotocol/server-filesystem /tmp"
```

### Config file

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/workspace"]
    }
  }
}
```

Run with:

```bash
cargo run --release -- --p 8000 --mcp-config mcp_config.json
```

## Reasoning-content routing

For tool-enabled requests, `CANDLE_VLLM_STREAM_AS_REASONING_CONTENT` controls whether streamed reasoning is emitted in OpenAI-style `reasoning_content` chunks.

```bash
export CANDLE_VLLM_STREAM_AS_REASONING_CONTENT=1
```

Set it to `0`, `false`, or `no` to keep reasoning in ordinary `content`
instead.

## Troubleshooting

- If tool calls are malformed for a specific model, try `--enforce-parser`.
- For Qwen coder models, `--enforce-parser qwen_coder` is usually the best choice.
- To inspect exact request/stream output while debugging clients:

```bash
export CANDLE_VLLM_CHAT_LOGGER=1
```
