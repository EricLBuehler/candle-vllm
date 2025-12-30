// src/mcp/mod.rs
//! Model Context Protocol (MCP) support
//!
//! Implements the MCP protocol for connecting to external tools and data sources.

pub mod client;
pub mod manager;
pub mod server;
pub mod transport;
pub mod types;

pub use client::McpClient;
pub use manager::{McpClientManager, McpManagerConfig};
pub use server::McpServer;
pub use transport::{HttpTransport, StdioTransport, Transport};
pub use types::*;
