//! SQLite storage backend implementation.
//!
//! This backend uses SQLite with tokio-rusqlite for async operations.
//! It provides persistent storage with ACID transactions.

use async_trait::async_trait;
use rusqlite::params;
use tokio_rusqlite::Connection;
use tracing::{debug, info};

use crate::state::backend_traits::{
    BackendError, MailboxBackendOps, MailboxRecord, QueueBackendOps, QueueRecord, now_secs,
};

/// SQLite mailbox backend using tokio-rusqlite for async operations.
pub struct SqliteMailboxBackend {
    conn: Connection,
}

impl SqliteMailboxBackend {
    /// Create a new SQLite mailbox backend at the given path.
    pub async fn new(path: &str) -> Result<Self, BackendError> {
        info!(path = %path, "Initializing SQLite mailbox backend");

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                BackendError::connection(format!("failed to create database directory: {}", e))
            })?;
        }

        let conn = Connection::open(path)
            .await
            .map_err(|e| BackendError::connection(format!("failed to open SQLite: {}", e)))?;

        // Initialize schema
        conn.call(|conn| {
            conn.execute(
                "CREATE TABLE IF NOT EXISTS mailbox_records (
                    request_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    created INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    response TEXT
                )",
                [],
            )?;
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mailbox_created ON mailbox_records(created)",
                [],
            )?;
            Ok(())
        })
        .await
        .map_err(|e| BackendError::connection(format!("failed to create schema: {}", e)))?;

        info!("SQLite mailbox backend initialized");
        Ok(Self { conn })
    }
}

#[async_trait]
impl MailboxBackendOps for SqliteMailboxBackend {
    async fn store(&self, record: MailboxRecord) -> Result<(), BackendError> {
        debug!(request_id = %record.request_id, "Storing mailbox record");

        let response_json = record
            .response
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default());

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT OR REPLACE INTO mailbox_records 
                     (request_id, model, created, status, response) 
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        record.request_id,
                        record.model,
                        record.created as i64,
                        record.status,
                        response_json,
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to store record: {}", e)))?;

        Ok(())
    }

    async fn get(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError> {
        debug!(request_id = %request_id, "Getting mailbox record");

        let request_id = request_id.to_string();
        
        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT request_id, model, created, status, response 
                     FROM mailbox_records WHERE request_id = ?1",
                )?;
                
                let mut rows = stmt.query(params![request_id])?;
                
                if let Some(row) = rows.next()? {
                    let response_str: Option<String> = row.get(4)?;
                    let response = response_str
                        .as_ref()
                        .and_then(|s| serde_json::from_str(s).ok());
                    
                    Ok(Some(MailboxRecord {
                        request_id: row.get(0)?,
                        model: row.get(1)?,
                        created: row.get::<_, i64>(2)? as u64,
                        status: row.get(3)?,
                        response,
                    }))
                } else {
                    Ok(None)
                }
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to get record: {}", e)))
    }

    async fn list(&self) -> Result<Vec<MailboxRecord>, BackendError> {
        debug!("Listing all mailbox records");

        self.conn
            .call(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT request_id, model, created, status, response 
                     FROM mailbox_records ORDER BY created DESC",
                )?;
                
                let rows = stmt.query_map([], |row| {
                    let response_str: Option<String> = row.get(4)?;
                    let response = response_str
                        .as_ref()
                        .and_then(|s| serde_json::from_str(s).ok());
                    
                    Ok(MailboxRecord {
                        request_id: row.get(0)?,
                        model: row.get(1)?,
                        created: row.get::<_, i64>(2)? as u64,
                        status: row.get(3)?,
                        response,
                    })
                })?;
                
                let mut records = Vec::new();
                for row in rows {
                    records.push(row?);
                }
                Ok(records)
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to list records: {}", e)))
    }

    async fn delete(&self, request_id: &str) -> Result<bool, BackendError> {
        debug!(request_id = %request_id, "Deleting mailbox record");

        let request_id = request_id.to_string();
        
        self.conn
            .call(move |conn| {
                let changes = conn.execute(
                    "DELETE FROM mailbox_records WHERE request_id = ?1",
                    params![request_id],
                )?;
                Ok(changes > 0)
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to delete record: {}", e)))
    }

    async fn get_and_delete(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError> {
        debug!(request_id = %request_id, "Getting and deleting mailbox record");

        let request_id = request_id.to_string();
        
        self.conn
            .call(move |conn| {
                // Use a transaction for atomicity
                let tx = conn.transaction()?;
                
                // Get the record
                let record = {
                    let mut stmt = tx.prepare(
                        "SELECT request_id, model, created, status, response 
                         FROM mailbox_records WHERE request_id = ?1",
                    )?;
                    
                    let mut rows = stmt.query(params![&request_id])?;
                    
                    if let Some(row) = rows.next()? {
                        let response_str: Option<String> = row.get(4)?;
                        let response = response_str
                            .as_ref()
                            .and_then(|s| serde_json::from_str(s).ok());
                        
                        Some(MailboxRecord {
                            request_id: row.get(0)?,
                            model: row.get(1)?,
                            created: row.get::<_, i64>(2)? as u64,
                            status: row.get(3)?,
                            response,
                        })
                    } else {
                        None
                    }
                };
                
                // Delete if found
                if record.is_some() {
                    tx.execute(
                        "DELETE FROM mailbox_records WHERE request_id = ?1",
                        params![&request_id],
                    )?;
                }
                
                tx.commit()?;
                Ok(record)
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to get and delete record: {}", e)))
    }

    async fn cleanup_expired(&self, max_age_secs: u64) -> Result<usize, BackendError> {
        let cutoff = now_secs().saturating_sub(max_age_secs) as i64;
        debug!(cutoff = cutoff, "Cleaning up expired mailbox records");

        self.conn
            .call(move |conn| {
                let changes = conn.execute(
                    "DELETE FROM mailbox_records WHERE created < ?1",
                    params![cutoff],
                )?;
                Ok(changes)
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to cleanup records: {}", e)))
    }
}

/// SQLite queue backend using tokio-rusqlite for async operations.
pub struct SqliteQueueBackend {
    conn: Connection,
}

impl SqliteQueueBackend {
    /// Create a new SQLite queue backend at the given path.
    pub async fn new(path: &str) -> Result<Self, BackendError> {
        info!(path = %path, "Initializing SQLite queue backend");

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                BackendError::connection(format!("failed to create database directory: {}", e))
            })?;
        }

        let conn = Connection::open(path)
            .await
            .map_err(|e| BackendError::connection(format!("failed to open SQLite: {}", e)))?;

        // Initialize schema
        conn.call(|conn| {
            conn.execute(
                "CREATE TABLE IF NOT EXISTS queue_records (
                    id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    queued_at INTEGER NOT NULL,
                    request TEXT NOT NULL,
                    PRIMARY KEY (model, queued_at, id)
                )",
                [],
            )?;
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_queue_model ON queue_records(model, queued_at)",
                [],
            )?;
            Ok(())
        })
        .await
        .map_err(|e| BackendError::connection(format!("failed to create schema: {}", e)))?;

        info!("SQLite queue backend initialized");
        Ok(Self { conn })
    }
}

#[async_trait]
impl QueueBackendOps for SqliteQueueBackend {
    async fn enqueue(&self, record: QueueRecord) -> Result<(), BackendError> {
        debug!(id = %record.id, model = %record.model, "Enqueueing record");

        let request_json = serde_json::to_string(&record.request)
            .map_err(|e| BackendError::serialization(format!("failed to serialize request: {}", e)))?;

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO queue_records (id, model, queued_at, request) VALUES (?1, ?2, ?3, ?4)",
                    params![
                        record.id,
                        record.model,
                        record.queued_at as i64,
                        request_json,
                    ],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to enqueue record: {}", e)))?;

        Ok(())
    }

    async fn dequeue(&self, model: &str, max: usize) -> Result<Vec<QueueRecord>, BackendError> {
        debug!(model = %model, max = max, "Dequeueing records");

        let model = model.to_string();
        
        self.conn
            .call(move |conn| {
                // Use a transaction for atomic dequeue
                let tx = conn.transaction()?;
                
                // Select oldest records
                let records = {
                    let mut stmt = tx.prepare(
                        "SELECT id, model, queued_at, request 
                         FROM queue_records 
                         WHERE model = ?1 
                         ORDER BY queued_at ASC 
                         LIMIT ?2",
                    )?;
                    
                    let rows = stmt.query_map(params![&model, max as i64], |row| {
                        let request_str: String = row.get(3)?;
                        let request = serde_json::from_str(&request_str).unwrap_or_default();
                        
                        Ok(QueueRecord {
                            id: row.get(0)?,
                            model: row.get(1)?,
                            queued_at: row.get::<_, i64>(2)? as u64,
                            request,
                        })
                    })?;
                    
                    let mut records = Vec::new();
                    for row in rows {
                        records.push(row?);
                    }
                    records
                };
                
                // Delete selected records
                for record in &records {
                    tx.execute(
                        "DELETE FROM queue_records WHERE model = ?1 AND queued_at = ?2 AND id = ?3",
                        params![record.model, record.queued_at as i64, record.id],
                    )?;
                }
                
                tx.commit()?;
                Ok(records)
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to dequeue records: {}", e)))
    }

    async fn len(&self, model: &str) -> Result<usize, BackendError> {
        let model = model.to_string();
        
        self.conn
            .call(move |conn| {
                let count: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM queue_records WHERE model = ?1",
                    params![model],
                    |row| row.get(0),
                )?;
                Ok(count as usize)
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to count records: {}", e)))
    }

    async fn list(&self, model: Option<&str>) -> Result<Vec<QueueRecord>, BackendError> {
        let model = model.map(|s| s.to_string());
        
        self.conn
            .call(move |conn| {
                let records = if let Some(model) = model {
                    let mut stmt = conn.prepare(
                        "SELECT id, model, queued_at, request 
                         FROM queue_records 
                         WHERE model = ?1 
                         ORDER BY queued_at ASC",
                    )?;
                    
                    let rows = stmt.query_map(params![model], |row| {
                        let request_str: String = row.get(3)?;
                        let request = serde_json::from_str(&request_str).unwrap_or_default();
                        
                        Ok(QueueRecord {
                            id: row.get(0)?,
                            model: row.get(1)?,
                            queued_at: row.get::<_, i64>(2)? as u64,
                            request,
                        })
                    })?;
                    
                    let mut records = Vec::new();
                    for row in rows {
                        records.push(row?);
                    }
                    records
                } else {
                    let mut stmt = conn.prepare(
                        "SELECT id, model, queued_at, request 
                         FROM queue_records 
                         ORDER BY queued_at ASC",
                    )?;
                    
                    let rows = stmt.query_map([], |row| {
                        let request_str: String = row.get(3)?;
                        let request = serde_json::from_str(&request_str).unwrap_or_default();
                        
                        Ok(QueueRecord {
                            id: row.get(0)?,
                            model: row.get(1)?,
                            queued_at: row.get::<_, i64>(2)? as u64,
                            request,
                        })
                    })?;
                    
                    let mut records = Vec::new();
                    for row in rows {
                        records.push(row?);
                    }
                    records
                };
                
                Ok(records)
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to list records: {}", e)))
    }

    async fn clear(&self, model: &str) -> Result<usize, BackendError> {
        debug!(model = %model, "Clearing queue for model");

        let model = model.to_string();
        
        self.conn
            .call(move |conn| {
                let changes = conn.execute(
                    "DELETE FROM queue_records WHERE model = ?1",
                    params![model],
                )?;
                Ok(changes)
            })
            .await
            .map_err(|e| BackendError::other(format!("failed to clear queue: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_sqlite_mailbox_store_and_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_mailbox.db");
        let backend = SqliteMailboxBackend::new(path.to_str().unwrap())
            .await
            .unwrap();

        let record = MailboxRecord {
            request_id: "test-123".to_string(),
            model: "gpt-4".to_string(),
            created: now_secs(),
            status: "completed".to_string(),
            response: Some(serde_json::json!({"text": "hello"})),
        };

        backend.store(record.clone()).await.unwrap();

        let retrieved = backend.get("test-123").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().request_id, "test-123");
    }

    #[tokio::test]
    async fn test_sqlite_mailbox_get_and_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_mailbox_gad.db");
        let backend = SqliteMailboxBackend::new(path.to_str().unwrap())
            .await
            .unwrap();

        let record = MailboxRecord {
            request_id: "test-456".to_string(),
            model: "gpt-4".to_string(),
            created: now_secs(),
            status: "completed".to_string(),
            response: None,
        };

        backend.store(record).await.unwrap();

        let retrieved = backend.get_and_delete("test-456").await.unwrap();
        assert!(retrieved.is_some());

        // Should be gone now
        let again = backend.get("test-456").await.unwrap();
        assert!(again.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_queue_enqueue_dequeue() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_queue.db");
        let backend = SqliteQueueBackend::new(path.to_str().unwrap())
            .await
            .unwrap();

        let record1 = QueueRecord {
            id: "req-1".to_string(),
            model: "llama".to_string(),
            queued_at: 1000,
            request: serde_json::json!({"prompt": "hello"}),
        };
        let record2 = QueueRecord {
            id: "req-2".to_string(),
            model: "llama".to_string(),
            queued_at: 1001,
            request: serde_json::json!({"prompt": "world"}),
        };

        backend.enqueue(record1).await.unwrap();
        backend.enqueue(record2).await.unwrap();

        assert_eq!(backend.len("llama").await.unwrap(), 2);

        let dequeued = backend.dequeue("llama", 1).await.unwrap();
        assert_eq!(dequeued.len(), 1);
        assert_eq!(dequeued[0].id, "req-1"); // FIFO order

        assert_eq!(backend.len("llama").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_queue_clear() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_queue_clear.db");
        let backend = SqliteQueueBackend::new(path.to_str().unwrap())
            .await
            .unwrap();

        for i in 0..5 {
            backend
                .enqueue(QueueRecord {
                    id: format!("req-{}", i),
                    model: "test-model".to_string(),
                    queued_at: i as u64,
                    request: serde_json::json!({}),
                })
                .await
                .unwrap();
        }

        assert_eq!(backend.len("test-model").await.unwrap(), 5);

        let cleared = backend.clear("test-model").await.unwrap();
        assert_eq!(cleared, 5);
        assert_eq!(backend.len("test-model").await.unwrap(), 0);
    }
}
