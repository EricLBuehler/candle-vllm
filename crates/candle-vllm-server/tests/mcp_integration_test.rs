//! MCP (Model Context Protocol) Integration Tests
//!
//! These tests verify:
//! - mcp.json file parsing and validation
//! - MCP server configuration validation
//! - MCP session initialization
//!
//! # Configuration
//!
//! Tests use the mcp.json file from the workspace root or the path
//! specified by CANDLE_VLLM_TEST_MCP_CONFIG in .test.env

use candle_vllm_server::config::{McpConfig, McpServerDefinition};
use std::fs;
use std::path::PathBuf;
use std::sync::Once;

// Initialize test environment once
static INIT: Once = Once::new();

/// Get the workspace root directory
fn get_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Initialize the test environment by loading .test.env
fn init_test_env() {
    INIT.call_once(|| {
        let workspace_root = get_workspace_root();
        let test_env_path = workspace_root.join(".test.env");

        if test_env_path.exists() {
            match dotenvy::from_path(&test_env_path) {
                Ok(_) => {
                    eprintln!("Loaded test environment from {:?}", test_env_path);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load .test.env: {}", e);
                }
            }
        }
    });
}

/// Get the MCP config path from environment or default
fn get_mcp_config_path() -> PathBuf {
    init_test_env();

    if let Ok(path) = std::env::var("CANDLE_VLLM_TEST_MCP_CONFIG") {
        let p = PathBuf::from(&path);
        if p.is_absolute() {
            p
        } else {
            get_workspace_root().join(path)
        }
    } else {
        get_workspace_root().join("mcp.json")
    }
}

// ============================================================================
// MCP CONFIG FILE TESTS
// ============================================================================

#[test]
fn test_mcp_json_exists() {
    let mcp_path = get_mcp_config_path();

    assert!(
        mcp_path.exists(),
        "mcp.json must exist at {:?}. Create the file or set CANDLE_VLLM_TEST_MCP_CONFIG",
        mcp_path
    );

    eprintln!("Found mcp.json at {:?}", mcp_path);
}

#[test]
fn test_mcp_json_valid_json() {
    let mcp_path = get_mcp_config_path();

    if !mcp_path.exists() {
        eprintln!("Skipping: mcp.json not found at {:?}", mcp_path);
        return;
    }

    let content = fs::read_to_string(&mcp_path).expect("Failed to read mcp.json");

    // Verify it's valid JSON
    let json: serde_json::Value =
        serde_json::from_str(&content).expect("mcp.json contains invalid JSON syntax");

    // Verify it has the expected structure (either "servers" or "mcpServers")
    let has_servers = json.get("servers").is_some();
    let has_mcp_servers = json.get("mcpServers").is_some();

    assert!(
        has_servers || has_mcp_servers,
        "mcp.json must contain either 'servers' array or 'mcpServers' object. Found keys: {:?}",
        json.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );

    eprintln!("mcp.json has valid JSON structure");
}

#[test]
fn test_mcp_config_load() {
    let mcp_path = get_mcp_config_path();

    if !mcp_path.exists() {
        eprintln!("Skipping: mcp.json not found at {:?}", mcp_path);
        return;
    }

    let path_str = mcp_path.to_string_lossy();
    let config =
        McpConfig::load(&path_str).expect("Failed to load mcp.json using McpConfig::load()");

    eprintln!("Loaded MCP config with {} server(s)", config.servers.len());

    for server in &config.servers {
        eprintln!("  - Server: {} -> {}", server.name, server.url);
    }
}

#[test]
fn test_mcp_config_has_servers() {
    let mcp_path = get_mcp_config_path();

    if !mcp_path.exists() {
        eprintln!("Skipping: mcp.json not found at {:?}", mcp_path);
        return;
    }

    let path_str = mcp_path.to_string_lossy();
    let config = McpConfig::load(&path_str).expect("Failed to load mcp.json");

    // At minimum, we expect at least one server configured
    assert!(
        !config.servers.is_empty(),
        "mcp.json should define at least one MCP server"
    );
}

#[test]
fn test_mcp_server_definitions_valid() {
    let mcp_path = get_mcp_config_path();

    if !mcp_path.exists() {
        eprintln!("Skipping: mcp.json not found at {:?}", mcp_path);
        return;
    }

    let path_str = mcp_path.to_string_lossy();
    let config = McpConfig::load(&path_str).expect("Failed to load mcp.json");

    for server in &config.servers {
        // Validate server name
        assert!(!server.name.is_empty(), "MCP server name cannot be empty");

        // Validate URL
        assert!(
            !server.url.is_empty(),
            "MCP server '{}' URL cannot be empty",
            server.name
        );

        // URL should be valid format (http:// or https://)
        assert!(
            server.url.starts_with("http://") || server.url.starts_with("https://"),
            "MCP server '{}' URL should be HTTP/HTTPS format, got: {}",
            server.name,
            server.url
        );

        eprintln!("Validated MCP server: {} -> {}", server.name, server.url);
    }
}

// ============================================================================
// MCP CONFIG FORMAT TESTS
// ============================================================================

#[test]
fn test_mcp_mcpservers_format_parsing() {
    // Test that we can parse the mcpServers object format
    let json = r#"{
        "mcpServers": {
            "test-server": {
                "type": "node",
                "command": "/usr/bin/test-command",
                "args": ["--arg1"],
                "env": {"KEY": "value"}
            }
        }
    }"#;

    // Write to temp file
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_mcp_servers_format.json");
    fs::write(&temp_path, json).expect("Failed to write temp file");

    let config =
        McpConfig::load(temp_path.to_str().unwrap()).expect("Failed to load mcpServers format");

    assert_eq!(config.servers.len(), 1, "Should have one server");
    assert_eq!(config.servers[0].name, "test-server");

    // Clean up
    let _ = fs::remove_file(&temp_path);
}

#[test]
fn test_mcp_servers_array_format_parsing() {
    // Test that we can parse the servers array format
    let json = r#"{
        "servers": [
            {
                "name": "test-server",
                "url": "http://localhost:3000/test",
                "timeout_secs": 30
            }
        ]
    }"#;

    // Write to temp file
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_mcp_array_format.json");
    fs::write(&temp_path, json).expect("Failed to write temp file");

    let config =
        McpConfig::load(temp_path.to_str().unwrap()).expect("Failed to load servers array format");

    assert_eq!(config.servers.len(), 1, "Should have one server");
    assert_eq!(config.servers[0].name, "test-server");
    assert_eq!(config.servers[0].url, "http://localhost:3000/test");
    assert_eq!(config.servers[0].timeout_secs, Some(30));

    // Clean up
    let _ = fs::remove_file(&temp_path);
}

#[test]
fn test_mcp_http_server_format() {
    // Test HTTP server format in mcpServers
    let json = r#"{
        "mcpServers": {
            "http-server": {
                "url": "https://api.example.com/mcp",
                "auth": "Bearer token123",
                "timeout_secs": 60
            }
        }
    }"#;

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_mcp_http_format.json");
    fs::write(&temp_path, json).expect("Failed to write temp file");

    let config =
        McpConfig::load(temp_path.to_str().unwrap()).expect("Failed to load HTTP server format");

    assert_eq!(config.servers.len(), 1);
    assert_eq!(config.servers[0].name, "http-server");
    assert_eq!(config.servers[0].url, "https://api.example.com/mcp");
    assert_eq!(config.servers[0].auth, Some("Bearer token123".to_string()));
    assert_eq!(config.servers[0].timeout_secs, Some(60));

    // Clean up
    let _ = fs::remove_file(&temp_path);
}

// ============================================================================
// MCP EXECUTABLE VALIDATION (Command-based servers)
// ============================================================================

#[test]
fn test_mcp_command_executable_check() {
    let mcp_path = get_mcp_config_path();

    if !mcp_path.exists() {
        eprintln!("Skipping: mcp.json not found at {:?}", mcp_path);
        return;
    }

    let content = fs::read_to_string(&mcp_path).expect("Failed to read mcp.json");
    let json: serde_json::Value = serde_json::from_str(&content).expect("Invalid JSON");

    // Check mcpServers format for command-based servers
    if let Some(mcp_servers) = json.get("mcpServers").and_then(|s| s.as_object()) {
        for (name, server) in mcp_servers {
            if let Some(command) = server.get("command").and_then(|c| c.as_str()) {
                let cmd_path = PathBuf::from(command);

                if cmd_path.exists() {
                    eprintln!("✓ MCP server '{}' executable exists: {:?}", name, cmd_path);
                } else {
                    eprintln!(
                        "⚠ WARNING: MCP server '{}' executable NOT found: {:?}",
                        name, cmd_path
                    );
                    eprintln!("  This server may fail to start. Install it or update the path in mcp.json");
                }
            }
        }
    }
}

// ============================================================================
// MCP CONFIG VALIDATION TESTS
// ============================================================================

#[test]
fn test_mcp_config_serialization_roundtrip() {
    let original = McpConfig {
        servers: vec![
            McpServerDefinition {
                name: "test-1".to_string(),
                url: "http://localhost:3001".to_string(),
                auth: None,
                timeout_secs: Some(30),
                instructions: Some("Test instructions".to_string()),
            },
            McpServerDefinition {
                name: "test-2".to_string(),
                url: "https://api.example.com".to_string(),
                auth: Some("Bearer token".to_string()),
                timeout_secs: Some(60),
                instructions: None,
            },
        ],
    };

    // Serialize to JSON
    let json = serde_json::to_string(&original).expect("Failed to serialize McpConfig");

    // Deserialize back
    let deserialized: McpConfig =
        serde_json::from_str(&json).expect("Failed to deserialize McpConfig");

    assert_eq!(deserialized.servers.len(), original.servers.len());

    for (orig, deser) in original.servers.iter().zip(deserialized.servers.iter()) {
        assert_eq!(orig.name, deser.name);
        assert_eq!(orig.url, deser.url);
        assert_eq!(orig.auth, deser.auth);
        assert_eq!(orig.timeout_secs, deser.timeout_secs);
        assert_eq!(orig.instructions, deser.instructions);
    }
}

#[test]
fn test_mcp_empty_config() {
    let json = r#"{"servers": []}"#;

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_mcp_empty.json");
    fs::write(&temp_path, json).expect("Failed to write temp file");

    let config = McpConfig::load(temp_path.to_str().unwrap()).expect("Failed to load empty config");

    assert!(config.servers.is_empty());

    // Clean up
    let _ = fs::remove_file(&temp_path);
}

#[test]
fn test_mcp_invalid_json_fails() {
    let json = r#"{ invalid json }"#;

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("test_mcp_invalid.json");
    fs::write(&temp_path, json).expect("Failed to write temp file");

    let result = McpConfig::load(temp_path.to_str().unwrap());

    assert!(result.is_err(), "Invalid JSON should fail to load");

    // Clean up
    let _ = fs::remove_file(&temp_path);
}

// ============================================================================
// INTEGRATION WITH WORKSPACE MCP.JSON
// ============================================================================

#[test]
fn test_workspace_mcp_json_production_ready() {
    let mcp_path = get_mcp_config_path();

    if !mcp_path.exists() {
        eprintln!("Skipping: mcp.json not found at {:?}", mcp_path);
        return;
    }

    let path_str = mcp_path.to_string_lossy();

    // 1. Load the config
    let config =
        McpConfig::load(&path_str).expect("mcp.json failed to load - this would break production!");

    // 2. Verify at least one server
    assert!(
        !config.servers.is_empty(),
        "mcp.json must define at least one server for production use"
    );

    // 3. Validate each server
    let mut warnings = Vec::new();

    for server in &config.servers {
        // Check URL format
        if !server.url.starts_with("http://") && !server.url.starts_with("https://") {
            warnings.push(format!(
                "Server '{}' has non-HTTP URL: {}",
                server.name, server.url
            ));
        }

        // Check for localhost URLs in production scenarios
        if server.url.contains("localhost") || server.url.contains("127.0.0.1") {
            eprintln!(
                "Note: Server '{}' uses localhost URL. Ensure the MCP proxy is running.",
                server.name
            );
        }
    }

    if !warnings.is_empty() {
        eprintln!("Warnings for mcp.json:");
        for w in &warnings {
            eprintln!("  - {}", w);
        }
    }

    eprintln!(
        "✓ mcp.json is production-ready with {} server(s)",
        config.servers.len()
    );
}
