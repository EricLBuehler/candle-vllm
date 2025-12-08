//! SurrealDB storage backend implementation.
//!
//! This backend uses embedded SurrealDB with RocksDB for persistent storage.
//! It provides ACID transactions and a flexible document model.

use async_trait::async_trait;
use surrealdb::engine::local::{Db, RocksDb};
use surrealdb::Surreal;
use tracing::{debug, error, info};

use crate::state::backend_traits::{
    BackendError, MailboxBackendOps, MailboxRecord, QueueBackendOps, QueueRecord, now_secs,
};

/// SurrealDB mailbox backend using embedded RocksDB storage.
pub struct SurrealMailboxBackend {
    db: Surreal<Db>,
}

impl SurrealMailboxBackend {
    /// Create a new SurrealDB mailbox backend at the given path.
    pub async fn new(path: &str) -> Result<Self, BackendError> {
        info!(path = %path, "Initializing SurrealDB mailbox backend");
        
        let db = Surreal::new::<RocksDb>(path)
            .await
            .map_err(|e| BackendError::connection(format!("failed to open SurrealDB: {}", e)))?;

        db.use_ns("candle_vllm")
            .use_db("mailbox")
            .await
            .map_err(|e| BackendError::connection(format!("failed to select namespace/db: {}", e)))?;

        info!("SurrealDB mailbox backend initialized");
        Ok(Self { db })
    }
}

#[async_trait]
impl MailboxBackendOps for SurrealMailboxBackend {
    async fn store(&self, record: MailboxRecord) -> Result<(), BackendError> {
        debug!(request_id = %record.request_id, "Storing mailbox record");
        
        let request_id = record.request_id.clone();
        
        // Upsert the record using request_id as the record ID
        let _: Option<MailboxRecord> = self
            .db
            .upsert(("mailbox", request_id))
            .content(record)
            .await
            .map_err(|e| BackendError::other(format!("failed to store record: {}", e)))?;

        Ok(())
    }

    async fn get(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError> {
        debug!(request_id = %request_id, "Getting mailbox record");
        
        let record: Option<MailboxRecord> = self
            .db
            .select(("mailbox", request_id))
            .await
            .map_err(|e| BackendError::other(format!("failed to get record: {}", e)))?;

        Ok(record)
    }

    async fn list(&self) -> Result<Vec<MailboxRecord>, BackendError> {
        debug!("Listing all mailbox records");
        
        let records: Vec<MailboxRecord> = self
            .db
            .select("mailbox")
            .await
            .map_err(|e| BackendError::other(format!("failed to list records: {}", e)))?;

        Ok(records)
    }

    async fn delete(&self, request_id: &str) -> Result<bool, BackendError> {
        debug!(request_id = %request_id, "Deleting mailbox record");
        
        let deleted: Option<MailboxRecord> = self
            .db
            .delete(("mailbox", request_id))
            .await
            .map_err(|e| BackendError::other(format!("failed to delete record: {}", e)))?;

        Ok(deleted.is_some())
    }

    async fn get_and_delete(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError> {
        debug!(request_id = %request_id, "Getting and deleting mailbox record");
        
        // Get the record first
        let record: Option<MailboxRecord> = self
            .db
            .select(("mailbox", request_id))
            .await
            .map_err(|e| BackendError::other(format!("failed to get record: {}", e)))?;

        if record.is_some() {
            // Delete it
            let _: Option<MailboxRecord> = self
                .db
                .delete(("mailbox", request_id))
                .await
                .map_err(|e| BackendError::other(format!("failed to delete record: {}", e)))?;
        }

        Ok(record)
    }

    async fn cleanup_expired(&self, max_age_secs: u64) -> Result<usize, BackendError> {
        let cutoff = now_secs().saturating_sub(max_age_secs);
        debug!(cutoff = cutoff, "Cleaning up expired mailbox records");

        // Query for old records and delete them
        let mut result = self
            .db
            .query("DELETE FROM mailbox WHERE created < $cutoff RETURN BEFORE")
            .bind(("cutoff", cutoff))
            .await
            .map_err(|e| BackendError::other(format!("failed to cleanup records: {}", e)))?;

        // Count deleted records
        let deleted: Vec<MailboxRecord> = result
            .take(0)
            .map_err(|e| BackendError::other(format!("failed to get deleted count: {}", e)))?;

        info!(deleted = deleted.len(), "Cleaned up expired mailbox records");
        Ok(deleted.len())
    }
}

/// SurrealDB queue backend using embedded RocksDB storage.
pub struct SurrealQueueBackend {
    db: Surreal<Db>,
}

impl SurrealQueueBackend {
    /// Create a new SurrealDB queue backend at the given path.
    pub async fn new(path: &str) -> Result<Self, BackendError> {
        info!(path = %path, "Initializing SurrealDB queue backend");
        
        let db = Surreal::new::<RocksDb>(path)
            .await
            .map_err(|e| BackendError::connection(format!("failed to open SurrealDB: {}", e)))?;

        db.use_ns("candle_vllm")
            .use_db("queue")
            .await
            .map_err(|e| BackendError::connection(format!("failed to select namespace/db: {}", e)))?;

        // Create index on model and queued_at for efficient queries
        let _ = db
            .query("DEFINE INDEX idx_queue_model_time ON TABLE queue COLUMNS model, queued_at")
            .await;

        info!("SurrealDB queue backend initialized");
        Ok(Self { db })
    }

    /// Generate a composite key for queue records.
    fn composite_key(record: &QueueRecord) -> String {
        format!("{}_{}_{}", record.model, record.queued_at, record.id)
    }
}

#[async_trait]
impl QueueBackendOps for SurrealQueueBackend {
    async fn enqueue(&self, record: QueueRecord) -> Result<(), BackendError> {
        debug!(
            id = %record.id,
            model = %record.model,
            "Enqueueing record"
        );

        let key = Self::composite_key(&record);
        let _: Option<QueueRecord> = self
            .db
            .create(("queue", key))
            .content(record)
            .await
            .map_err(|e| BackendError::other(format!("failed to enqueue record: {}", e)))?;

        Ok(())
    }

    async fn dequeue(&self, model: &str, max: usize) -> Result<Vec<QueueRecord>, BackendError> {
        debug!(model = %model, max = max, "Dequeueing records");

        // Convert to owned string to satisfy lifetime requirements
        let model_owned = model.to_string();

        // Select oldest records for this model
        let mut result = self
            .db
            .query("SELECT * FROM queue WHERE model = $model ORDER BY queued_at ASC LIMIT $max")
            .bind(("model", model_owned.clone()))
            .bind(("max", max))
            .await
            .map_err(|e| BackendError::other(format!("failed to query records: {}", e)))?;

        let records: Vec<QueueRecord> = result
            .take(0)
            .map_err(|e| BackendError::other(format!("failed to take records: {}", e)))?;

        // Delete the selected records
        for record in &records {
            let key = Self::composite_key(record);
            let _: Option<QueueRecord> = self
                .db
                .delete(("queue", &key))
                .await
                .map_err(|e| {
                    error!(id = %record.id, error = %e, "Failed to delete dequeued record");
                    BackendError::other(format!("failed to delete record: {}", e))
                })?;
        }

        debug!(model = %model, count = records.len(), "Dequeued records");
        Ok(records)
    }

    async fn len(&self, model: &str) -> Result<usize, BackendError> {
        let model_owned = model.to_string();
        
        let mut result = self
            .db
            .query("SELECT count() FROM queue WHERE model = $model GROUP ALL")
            .bind(("model", model_owned))
            .await
            .map_err(|e| BackendError::other(format!("failed to count records: {}", e)))?;

        #[derive(serde::Deserialize)]
        struct CountResult {
            count: usize,
        }

        let count: Option<CountResult> = result
            .take(0)
            .map_err(|e| BackendError::other(format!("failed to get count: {}", e)))?;

        Ok(count.map(|c| c.count).unwrap_or(0))
    }

    async fn list(&self, model: Option<&str>) -> Result<Vec<QueueRecord>, BackendError> {
        let records: Vec<QueueRecord> = if let Some(model) = model {
            let model_owned = model.to_string();
            
            let mut result = self
                .db
                .query("SELECT * FROM queue WHERE model = $model ORDER BY queued_at ASC")
                .bind(("model", model_owned))
                .await
                .map_err(|e| BackendError::other(format!("failed to query records: {}", e)))?;

            result
                .take(0)
                .map_err(|e| BackendError::other(format!("failed to take records: {}", e)))?
        } else {
            self.db
                .select("queue")
                .await
                .map_err(|e| BackendError::other(format!("failed to list records: {}", e)))?
        };

        Ok(records)
    }

    async fn clear(&self, model: &str) -> Result<usize, BackendError> {
        debug!(model = %model, "Clearing queue for model");

        let model_owned = model.to_string();

        let mut result = self
            .db
            .query("DELETE FROM queue WHERE model = $model RETURN BEFORE")
            .bind(("model", model_owned))
            .await
            .map_err(|e| BackendError::other(format!("failed to clear queue: {}", e)))?;

        let deleted: Vec<QueueRecord> = result
            .take(0)
            .map_err(|e| BackendError::other(format!("failed to get deleted count: {}", e)))?;

        info!(model = %model, deleted = deleted.len(), "Cleared queue");
        Ok(deleted.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_surreal_mailbox_store_and_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_mailbox.db");
        let backend = SurrealMailboxBackend::new(path.to_str().unwrap())
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
    async fn test_surreal_mailbox_get_and_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_mailbox_gad.db");
        let backend = SurrealMailboxBackend::new(path.to_str().unwrap())
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
    async fn test_surreal_queue_enqueue_dequeue() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_queue.db");
        let backend = SurrealQueueBackend::new(path.to_str().unwrap())
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
    async fn test_surreal_queue_clear() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_queue_clear.db");
        let backend = SurrealQueueBackend::new(path.to_str().unwrap())
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
