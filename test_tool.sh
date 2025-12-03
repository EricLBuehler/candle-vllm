#!/bin/bash

# Test tool calling with candle-vllm
# Make sure the server is running first, e.g.:
# cargo run --release -- --model mistralai/Mistral-7B-Instruct-v0.3 --port 2000

curl http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "mistral",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather like in San Francisco today?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use"
              }
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 256
  }'
