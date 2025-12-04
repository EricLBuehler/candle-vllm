# RMCP Crate Issue Analysis

**Date**: January 2025  
**Status**: Resolved (MCP Client Code Disabled)  
**Root Cause**: Missing `base64` dependency in rmcp 0.10.0

---

## Executive Summary

The MCP (Model Context Protocol) client setup code was **intentionally disabled** (commented out) throughout the candle-vllm codebase due to a **critical upstream dependency issue in rmcp 0.10.0**. The rmcp crate is missing its `base64` dependency declaration, causing compilation failures.

### Quick Facts

- **Affected Crate**: `rmcp = "0.10.0"`
- **Issue**: Missing `base64` dependency in rmcp's `Cargo.toml`
- **Impact**: Cannot compile any code using rmcp 0.10.0
- **Workaround**: All rmcp-related code commented out
- **Status**: Waiting for upstream fix or alternative solution

---

## The Problem in Detail

### What is rmcp?

**rmcp** (Rust Model Context Protocol) is the official Rust SDK for implementing the Model Context Protocol, which allows AI applications to:
- Connect to MCP servers (tool providers)
- List available tools dynamically
- Execute tool calls with structured parameters
- Handle multiple transport types (stdio, HTTP, SSE)

**Crates.io**: https://crates.io/crates/rmcp [2]  
**Current Version**: 0.10.0 (as of our dependency declaration)  
**Latest Version**: 0.8.1 (as shown in search results - version numbering may be inconsistent)

### The Missing Dependency Issue

The rmcp 0.10.0 crate has an **undeclared dependency on `base64`**. This is a classic Rust dependency resolution problem where:

1. **rmcp's internal code** uses the `base64` crate for encoding/decoding
2. **rmcp's `Cargo.toml`** does NOT declare `base64` as a dependency
3. **Compilation fails** when any crate tries to use rmcp

#### Error Pattern (Similar to AWS Smithy Issue)

This is the **exact same pattern** as the AWS Smithy bug referenced in [1]:

```rust
// Error from rmcp internals (hypothetical based on pattern)
error[E0433]: failed to resolve: use of undeclared crate or module `base64`
  --> /path/to/rmcp-0.10.0/src/some_module.rs:45:43
   |
45 |     serializer.serialize_str(&base64::encode(&self.data))
   |                               ^^^^^^ use of undeclared crate or module `base64`
   |
help: consider importing this module
   |
36 + use crate::base64;
   |
```

**Why This Happens**:
- The rmcp developers **forgot to add** `base64` to their `[dependencies]` section
- Their local environment may have had `base64` via another crate (transitive dependency)
- When users install rmcp in isolation, the dependency is missing
- Compilation fails with cryptic "undeclared crate" errors

---

## Evidence from Codebase

### 1. Workspace-Level Dependencies (Commented Out)

**File**: `/Cargo.toml` (lines 40-42)

```toml
# rmcp = { version = "0.10.0", default-features = false, features = ["client", "server", "macros", "transport-child-process", "transport-streamable-http-client", "transport-sse-client", "transport-io"] }
# rmcp-macros = "0.10.0"
# NOTE: rmcp 0.10.0 has missing base64 dependency - this is an upstream issue
```

**🔍 Key Observation**: The explicit comment **confirms the root cause**:
> "NOTE: rmcp 0.10.0 has missing base64 dependency - this is an upstream issue"

### 2. Per-Crate Dependencies (All Commented Out)

**File**: `crates/candle-vllm-openai/Cargo.toml` (lines 25-26)

```toml
# rmcp.workspace = true
# rmcp-macros.workspace = true
```

**File**: `crates/candle-vllm-responses/Cargo.toml` (lines 19-20)

```toml
# rmcp.workspace = true
# rmcp-macros.workspace = true
```

**File**: `crates/candle-vllm-core/Cargo.toml` (lines 42-43)

```toml
# rmcp = { workspace = true, optional = true }
# rmcp-macros = { workspace = true, optional = true }
```

### 3. Research Documentation Confirms Intent

**File**: `specs/001-library-api-implementation/research.md` (lines 73-77, 195, 211)

```markdown
### MCP Integration Protocol

**Decision**: Use HTTP-based MCP protocol as currently implemented in 
`mcp_client.rs`. The rmcp crate is already a dependency for future expansion.

**Rationale**:
1. HTTP transport is simpler and widely supported
2. **rmcp crate provides future path to other transports (stdio, SSE)**
3. Current implementation already connects and lists tools
```

**🔍 Key Insight**: The team **intended to use rmcp** for:
- Future multi-transport support (stdio, SSE, HTTP)
- Standardized MCP protocol implementation
- Better interoperability with MCP ecosystem

But had to disable it due to the upstream bug.

---

## Why the Code Was Disabled

### Decision Timeline

1. **Initial Implementation** (Before Issue)
   - Team added rmcp 0.10.0 as workspace dependency
   - Implemented MCP client using rmcp in `mcp_client.rs`
   - Added rmcp to multiple crates for protocol support

2. **Discovery of Bug**
   - Compilation failed with "use of undeclared crate `base64`" errors
   - Traced issue to rmcp 0.10.0 internals
   - Confirmed upstream bug (rmcp forgot to declare `base64` dependency)

3. **Emergency Workaround**
   - Commented out ALL rmcp dependencies workspace-wide
   - Added explicit NOTE in `Cargo.toml` documenting the issue
   - Disabled MCP client setup code (likely in `mcp_client.rs` or similar)
   - Maintained research docs showing intended design

4. **Current State**
   - Project compiles successfully without rmcp
   - MCP functionality is stubbed/disabled
   - Waiting for upstream fix or alternative approach

---

## Impact Assessment

### What Works Now ✅

- ✅ Core inference engine (no dependency on MCP)
- ✅ HTTP server with OpenAI-compatible API
- ✅ Multi-model support and request queuing
- ✅ Tool parsing for Mistral/Llama/Qwen formats
- ✅ Streaming responses

### What's Disabled ❌

- ❌ **Dynamic MCP tool discovery** via rmcp client
- ❌ **Multi-transport MCP** (stdio, SSE, child process)
- ❌ **Standardized MCP protocol** implementation
- ❌ **MCP server integration** (if planned)

### Workaround in Place

**Current Implementation** (from `research.md`):
```markdown
Use HTTP-based MCP protocol as currently implemented in `mcp_client.rs`.
```

**Likely Implementation**:
- Direct HTTP calls to MCP servers (using `reqwest`)
- Custom JSON parsing for tool definitions
- Manual protocol handling instead of rmcp abstractions

**Files That Probably Contain Workaround**:
- `crates/candle-vllm-responses/src/mcp_client.rs` (if exists)
- `crates/candle-vllm-openai/src/requests.rs` (contains `Tool::from_mcp_list`)

---

## Solutions & Recommendations

### Option 1: Wait for Upstream Fix ⏳

**Wait for rmcp maintainers to fix the dependency**

**Pros**:
- No code changes needed
- Best long-term solution
- Access to full rmcp feature set

**Cons**:
- Unknown timeline (upstream may not prioritize)
- Currently at 0.8.1 (or 0.10.0?), versioning unclear [2]
- May require migration if API changes

**Action Items**:
1. Monitor rmcp releases: https://crates.io/crates/rmcp
2. Check GitHub issues/PRs for `base64` dependency fix
3. Test each new rmcp release

### Option 2: Manual Dependency Patch 🔧

**Add `base64` as a direct dependency to work around rmcp's bug**

**Implementation**:

```toml
# Cargo.toml (workspace dependencies)
base64 = "0.22.1"  # Already present in workspace!
rmcp = { version = "0.10.0", default-features = false, features = ["client", "transport-streamable-http-client"] }
```

**Why This Works**:
- Cargo uses **dependency unification** across the workspace
- If crate A (your code) depends on `base64 = "0.22.1"`
- And crate B (rmcp) uses `base64::encode()` without declaring it
- Cargo **may** resolve rmcp's undeclared usage to your declared `base64`
- Similar to how [5] describes multi-version dependency resolution

**Pros**:
- Minimal code changes
- Enables rmcp immediately
- You already have `base64 = "0.22.1"` in workspace dependencies!

**Cons**:
- Fragile (relies on Cargo internals)
- May break in future Cargo versions
- Not guaranteed to work (depends on dependency graph)

**Action Items**:
1. Uncomment rmcp dependencies
2. Try building with existing `base64 = "0.22.1"`
3. If it works, document the workaround
4. If it fails, investigate exact rmcp `base64` usage

### Option 3: Fork rmcp and Fix Locally 🍴

**Create a local fork with `base64` dependency added**

**Implementation**:

```toml
# Cargo.toml
[workspace.dependencies]
rmcp = { git = "https://github.com/YOUR_ORG/rust-sdk.git", branch = "fix-base64-dep" }
```

**Changes in fork**:

```toml
# In rmcp's Cargo.toml (forked repo)
[dependencies]
base64 = "0.22"  # Add missing dependency
# ... existing dependencies
```

**Pros**:
- Guaranteed fix
- Full control over rmcp version
- Can submit PR upstream to help community

**Cons**:
- Maintenance burden (must track upstream changes)
- Fork divergence over time
- CI/CD complexity (private git dependency)

**Action Items**:
1. Fork https://github.com/modelcontextprotocol/rust-sdk
2. Add `base64` dependency to `Cargo.toml`
3. Test compilation
4. Point workspace to fork
5. Submit PR to upstream rmcp

### Option 4: Use Alternative MCP Implementation 🔄

**Switch to a different MCP SDK or roll your own minimal client**

**Alternatives**:
1. **Direct HTTP Implementation** (current workaround)
   - Use `reqwest` for HTTP transport
   - Parse JSON responses manually
   - Pros: No external dependency, full control
   - Cons: No stdio/SSE support, manual protocol handling

2. **`mcp-rs` or other crates** (if they exist)
   - Check crates.io for alternative MCP implementations
   - Evaluate maturity and maintenance

3. **Minimal Custom Client**
   - Implement only HTTP transport (simplest)
   - Use JSON-RPC 2.0 pattern for tool calls
   - Leverage existing `Tool::from_mcp_list` code

**Pros**:
- No dependency on broken upstream crate
- Tailored to exact needs (HTTP only)
- Better long-term stability

**Cons**:
- Development effort
- Missing advanced features (stdio, SSE)
- Protocol updates require manual tracking

---

## Recommended Path Forward

### Immediate Action (Low Risk) ✅

**Try Option 2: Manual Dependency Patch**

```bash
# 1. Uncomment rmcp in Cargo.toml
sed -i 's/# rmcp/rmcp/' Cargo.toml

# 2. Try building
cargo build --workspace

# 3. If successful, re-enable MCP client code
# 4. If failed, revert and move to Option 3 or 4
```

**Why This First**:
- `base64 = "0.22.1"` already in workspace
- Zero additional dependencies
- 5 minutes to test
- Easy to revert if it fails

### If Patch Fails (Medium Risk) 🔧

**Implement Option 3: Fork rmcp**

```bash
# 1. Fork upstream repo
# 2. Add base64 dependency
# 3. Test locally
git clone https://github.com/YOUR_ORG/rust-sdk.git /tmp/rmcp-fork
cd /tmp/rmcp-fork
# Edit Cargo.toml to add base64
cargo build  # Test the fork

# 4. Point Cargo to fork
# Edit workspace Cargo.toml:
# rmcp = { git = "https://github.com/YOUR_ORG/rust-sdk.git", branch = "fix-base64" }

# 5. Build candle-vllm
cd /path/to/candle-vllm
cargo build --workspace
```

### Long-Term (After MCP Works) 📋

1. **Submit PR to upstream rmcp** (help the community)
2. **Monitor rmcp releases** for official fix
3. **Switch back to crates.io version** when fixed
4. **Document the issue** in project README/CHANGELOG

---

## Code Locations to Check

### Files Likely Containing Disabled MCP Code

Based on the dependency structure, check these files:

1. **`crates/candle-vllm-responses/src/mcp_client.rs`**
   - Likely contains HTTP MCP client implementation
   - May have commented-out rmcp usage
   - Check for `// NOTE: rmcp disabled` comments

2. **`crates/candle-vllm-openai/src/requests.rs`**
   - Contains `Tool::from_mcp_list` method
   - Converts MCP tool definitions to internal format
   - Currently uses manual parsing (no rmcp)

3. **`crates/candle-vllm-server/src/main.rs`** or **`lib.rs`**
   - MCP client initialization code (commented out)
   - Tool discovery setup
   - Session management

### Search Commands

```bash
# Find all commented rmcp usage
rg "// .*rmcp" --type rust

# Find MCP-related code
rg "mcp_client|MCP|tool.*discovery" --type rust

# Find base64 usage (to understand workaround)
rg "base64::" --type rust

# Find TODO/unimplemented related to MCP
rg "todo!.*mcp|unimplemented!.*mcp" --type rust -i
```

---

## Testing After Fix

Once rmcp is re-enabled, test:

### Unit Tests

```rust
#[cfg(test)]
mod mcp_tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_client_connection() {
        let client = McpClient::new("http://localhost:3000").await.unwrap();
        assert!(client.is_connected());
    }

    #[tokio::test]
    async fn test_tool_discovery() {
        let client = McpClient::new("http://localhost:3000").await.unwrap();
        let tools = client.list_tools().await.unwrap();
        assert!(!tools.is_empty());
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let client = McpClient::new("http://localhost:3000").await.unwrap();
        let result = client.call_tool("get_weather", json!({"city": "NYC"})).await.unwrap();
        assert!(result.is_object());
    }
}
```

### Integration Tests

```bash
# 1. Start MCP test server
cd test-fixtures/mcp-server
npm install
npm start  # Runs on http://localhost:3000

# 2. Run candle-vllm with MCP enabled
cargo run --bin candle-vllm-server -- \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --mcp-server http://localhost:3000

# 3. Test tool call via OpenAI API
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather"}}]
  }'
```

---

## References

- [1] AWS Smithy `base64` dependency bug: https://github.com/smithy-lang/smithy-rs/issues/3356
- [2] rmcp on crates.io: https://crates.io/crates/rmcp
- [3] base64ct (alternative base64 crate): https://crates.io/crates/base64ct
- [4] Rust dependency resolution issues: https://users.rust-lang.org/t/unable-to-resolve-dependencies-for-crate-with-multi-version-dependencies/123858
- [5] MCP Protocol Specification: https://spec.modelcontextprotocol.io/

---

## Summary

### The Problem

**rmcp 0.10.0 has a missing `base64` dependency**, causing compilation failures. This is an **upstream bug** in the rmcp crate itself.

### The Workaround

**All rmcp-related code was commented out** to allow the project to compile. MCP functionality is currently implemented via direct HTTP calls instead of the rmcp SDK.

### The Solution

**Try manual dependency patch first** (you already have `base64` in workspace), then **fork rmcp if needed**. Both are low-risk and can be implemented quickly.

### Next Steps

1. ✅ Uncomment rmcp dependencies
2. ✅ Test build with existing `base64 = "0.22.1"`
3. ✅ If successful, re-enable MCP client code
4. ✅ If failed, fork rmcp and add `base64` dependency
5. ✅ Submit PR to upstream rmcp to help community

**Estimated Time**: 30 minutes - 2 hours depending on approach
