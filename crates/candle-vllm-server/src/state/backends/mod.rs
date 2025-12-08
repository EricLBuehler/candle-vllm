//! Storage backend implementations for mailbox and queue services.
//!
//! This module provides pluggable storage backends:
//! - `memory` - In-memory storage (default, non-persistent)
//! - `sqlite` - SQLite-based persistent storage (feature: `queue-sqlite`)
//! - `postgres` - PostgreSQL-based persistent storage (feature: `queue-postgres`)
//! - `surreal` - SurrealDB-based persistent storage (feature: `queue-surreal`)

pub mod memory;

#[cfg(feature = "queue-sqlite")]
pub mod sqlite;

#[cfg(feature = "queue-postgres")]
pub mod postgres;

#[cfg(feature = "queue-surreal")]
pub mod surreal;

// Re-exports
pub use memory::{MemoryMailboxBackend, MemoryQueueBackend};

#[cfg(feature = "queue-sqlite")]
pub use sqlite::{SqliteMailboxBackend, SqliteQueueBackend};

#[cfg(feature = "queue-postgres")]
pub use postgres::{PostgresMailboxBackend, PostgresQueueBackend};

#[cfg(feature = "queue-surreal")]
pub use surreal::{SurrealMailboxBackend, SurrealQueueBackend};

use crate::config::{MailboxBackendConfig, QueueBackendConfig};
use crate::state::backend_traits::{
    BackendError, MailboxBackendOps, QueueBackendOps,
};
use std::sync::Arc;

/// Dynamic mailbox backend type using trait objects.
pub type DynMailboxBackend = Arc<dyn MailboxBackendOps>;

/// Dynamic queue backend type using trait objects.
pub type DynQueueBackend = Arc<dyn QueueBackendOps>;

/// Build a mailbox backend from configuration.
pub async fn build_mailbox_backend(
    config: &MailboxBackendConfig,
) -> Result<DynMailboxBackend, BackendError> {
    match config.backend.to_lowercase().as_str() {
        "memory" => {
            let backend = MemoryMailboxBackend::new();
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "queue-sqlite")]
        "sqlite" => {
            let path = config
                .sqlite_path
                .as_deref()
                .unwrap_or("./data/mailbox.db");
            let backend = SqliteMailboxBackend::new(path).await?;
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "queue-postgres")]
        "postgres" => {
            let url = config.postgres_url.as_deref().ok_or_else(|| {
                BackendError::config("mailbox.backend=postgres requires postgres_url")
            })?;
            let backend = PostgresMailboxBackend::new(url).await?;
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "queue-surreal")]
        "surrealdb" => {
            let path = config
                .surreal_path
                .as_deref()
                .unwrap_or("./data/mailbox.surreal");
            let backend = SurrealMailboxBackend::new(path).await?;
            Ok(Arc::new(backend))
        }
        other => Err(BackendError::config(format!(
            "unsupported mailbox backend '{}'; available: memory{}{}{}",
            other,
            if cfg!(feature = "queue-sqlite") { ", sqlite" } else { "" },
            if cfg!(feature = "queue-postgres") { ", postgres" } else { "" },
            if cfg!(feature = "queue-surreal") { ", surrealdb" } else { "" },
        ))),
    }
}

/// Build a queue backend from configuration.
pub async fn build_queue_backend(
    config: &QueueBackendConfig,
) -> Result<DynQueueBackend, BackendError> {
    match config.backend.to_lowercase().as_str() {
        "memory" => {
            let backend = MemoryQueueBackend::new();
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "queue-sqlite")]
        "sqlite" => {
            let path = config
                .sqlite_path
                .as_deref()
                .unwrap_or("./data/queue.db");
            let backend = SqliteQueueBackend::new(path).await?;
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "queue-postgres")]
        "postgres" => {
            let url = config.postgres_url.as_deref().ok_or_else(|| {
                BackendError::config("queue.backend=postgres requires postgres_url")
            })?;
            let backend = PostgresQueueBackend::new(url).await?;
            Ok(Arc::new(backend))
        }
        #[cfg(feature = "queue-surreal")]
        "surrealdb" => {
            let path = config
                .surreal_path
                .as_deref()
                .unwrap_or("./data/queue.surreal");
            let backend = SurrealQueueBackend::new(path).await?;
            Ok(Arc::new(backend))
        }
        "yaque" => {
            // Yaque is a file-based queue, map it to memory for now
            // TODO: Implement proper yaque support if needed
            tracing::warn!("yaque backend not fully implemented, using memory fallback");
            let backend = MemoryQueueBackend::new();
            Ok(Arc::new(backend))
        }
        other => Err(BackendError::config(format!(
            "unsupported queue backend '{}'; available: memory{}{}{}",
            other,
            if cfg!(feature = "queue-sqlite") { ", sqlite" } else { "" },
            if cfg!(feature = "queue-postgres") { ", postgres" } else { "" },
            if cfg!(feature = "queue-surreal") { ", surrealdb" } else { "" },
        ))),
    }
}
