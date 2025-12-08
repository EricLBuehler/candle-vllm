//! Async backend traits for mailbox and queue services.
//!
//! These traits define the interface that all storage backends must implement,
//! allowing for pluggable backends (memory, SQLite, PostgreSQL, SurrealDB).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Error type for backend operations.
#[derive(Debug, Clone)]
pub struct BackendError {
    pub message: String,
    pub kind: BackendErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendErrorKind {
    /// Record not found
    NotFound,
    /// Connection or I/O error
    Connection,
    /// Serialization/deserialization error
    Serialization,
    /// Transaction or consistency error
    Transaction,
    /// Configuration error
    Configuration,
    /// Other errors
    Other,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::fmt::Display for BackendErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "not found"),
            Self::Connection => write!(f, "connection error"),
            Self::Serialization => write!(f, "serialization error"),
            Self::Transaction => write!(f, "transaction error"),
            Self::Configuration => write!(f, "configuration error"),
            Self::Other => write!(f, "error"),
        }
    }
}

impl std::error::Error for BackendError {}

impl BackendError {
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: BackendErrorKind::NotFound,
        }
    }

    pub fn connection(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: BackendErrorKind::Connection,
        }
    }

    pub fn serialization(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: BackendErrorKind::Serialization,
        }
    }

    pub fn transaction(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: BackendErrorKind::Transaction,
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: BackendErrorKind::Configuration,
        }
    }

    pub fn other(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: BackendErrorKind::Other,
        }
    }

    pub fn is_not_found(&self) -> bool {
        self.kind == BackendErrorKind::NotFound
    }

    pub fn is_connection(&self) -> bool {
        self.kind == BackendErrorKind::Connection
    }
}

impl From<BackendError> for candle_core::Error {
    fn from(e: BackendError) -> Self {
        candle_core::Error::Msg(e.message)
    }
}

/// A mailbox record storing the result of a completed inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailboxRecord {
    pub request_id: String,
    pub model: String,
    pub created: u64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
}

/// A queue record representing a pending inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueRecord {
    pub id: String,
    pub model: String,
    pub queued_at: u64,
    pub request: serde_json::Value,
}

/// Async trait for mailbox storage backends.
///
/// Implementations must be thread-safe (`Send + Sync`) to support
/// concurrent access from multiple request handlers.
#[async_trait]
pub trait MailboxBackendOps: Send + Sync {
    /// Store a mailbox record, upserting if the request_id already exists.
    async fn store(&self, record: MailboxRecord) -> Result<(), BackendError>;

    /// Retrieve a mailbox record by request_id.
    async fn get(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError>;

    /// List all mailbox records.
    async fn list(&self) -> Result<Vec<MailboxRecord>, BackendError>;

    /// Delete a mailbox record by request_id. Returns true if deleted.
    async fn delete(&self, request_id: &str) -> Result<bool, BackendError>;

    /// Atomically get and delete a mailbox record. Returns the record if found.
    async fn get_and_delete(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError>;

    /// Clean up expired records based on retention policy.
    /// Returns the number of records deleted.
    async fn cleanup_expired(&self, max_age_secs: u64) -> Result<usize, BackendError>;
}

/// Async trait for queue storage backends.
///
/// Implementations must be thread-safe (`Send + Sync`) to support
/// concurrent access from multiple request handlers.
#[async_trait]
pub trait QueueBackendOps: Send + Sync {
    /// Enqueue a record to the queue for the specified model.
    async fn enqueue(&self, record: QueueRecord) -> Result<(), BackendError>;

    /// Atomically dequeue up to `max` records from the queue for the specified model.
    /// Records are removed from the queue and returned in FIFO order.
    async fn dequeue(&self, model: &str, max: usize) -> Result<Vec<QueueRecord>, BackendError>;

    /// Get the number of pending records in the queue for the specified model.
    async fn len(&self, model: &str) -> Result<usize, BackendError>;

    /// List records without removing them. If model is None, list all records.
    async fn list(&self, model: Option<&str>) -> Result<Vec<QueueRecord>, BackendError>;

    /// Check if the queue is empty for the specified model.
    async fn is_empty(&self, model: &str) -> Result<bool, BackendError> {
        Ok(self.len(model).await? == 0)
    }

    /// Clear all records from the queue for the specified model.
    /// Returns the number of records removed.
    async fn clear(&self, model: &str) -> Result<usize, BackendError>;
}

/// Helper to get current time in seconds since UNIX epoch.
pub fn now_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_error_display() {
        let err = BackendError::not_found("record xyz not found");
        assert_eq!(format!("{}", err), "not found: record xyz not found");
        assert!(err.is_not_found());
    }

    #[test]
    fn test_backend_error_kinds() {
        assert!(BackendError::connection("db down").is_connection());
        assert!(!BackendError::other("misc").is_not_found());
    }
}
