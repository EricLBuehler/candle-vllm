//! Utility functions for OpenAI compatibility layer.
//!
//! This module provides helper functions used across the OpenAI implementation.

use std::time::{SystemTime, UNIX_EPOCH};

/// Get the current time in seconds since UNIX epoch.
///
/// Used for timestamping API responses.
pub fn get_created_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time is before UNIX epoch")
        .as_secs()
}
