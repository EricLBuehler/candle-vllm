use super::{McpConfig, McpServerDefinition, ModelRegistryConfig};
use std::collections::HashSet;

pub fn validate_models(registry: &ModelRegistryConfig) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut names = HashSet::new();
    for profile in &registry.models {
        if !names.insert(profile.name.clone()) {
            errors.push(format!("duplicate model name '{}'", profile.name));
        }
        if !profile.has_source() {
            errors.push(format!(
                "model '{}' missing source (hf_id or local_path)",
                profile.name
            ));
        }
        if let Some(kvcache) = profile.params.kvcache_mem_gpu {
            if kvcache == 0 {
                errors.push(format!(
                    "model '{}' kvcache_mem_gpu must be > 0",
                    profile.name
                ));
            }
        }
        if let Some(timeout) = profile.params.prefill_chunk_size {
            if timeout == 0 {
                errors.push(format!(
                    "model '{}' prefill_chunk_size must be > 0",
                    profile.name
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub fn validate_mcp(config: &McpConfig) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut names = HashSet::new();
    for server in &config.servers {
        validate_mcp_server(server, &mut errors, &mut names);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_mcp_server(
    server: &McpServerDefinition,
    errors: &mut Vec<String>,
    names: &mut HashSet<String>,
) {
    if !names.insert(server.name.clone()) {
        errors.push(format!("duplicate MCP server name '{}'", server.name));
    }
    if server.url.trim().is_empty() {
        errors.push(format!("MCP server '{}' missing url", server.name));
    }
    if let Some(timeout) = server.timeout_secs {
        if timeout == 0 {
            errors.push(format!(
                "MCP server '{}' timeout_secs must be greater than zero",
                server.name
            ));
        }
    }
}
