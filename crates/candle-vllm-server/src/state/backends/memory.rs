//! In-memory storage backend implementation.
//!
//! This is the default backend that stores all data in memory.
//! Data is lost when the server restarts.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};

use crate::state::backend_traits::{
    BackendError, MailboxBackendOps, MailboxRecord, QueueBackendOps, QueueRecord, now_secs,
};

/// In-memory mailbox backend using a HashMap protected by RwLock.
#[derive(Debug)]
pub struct MemoryMailboxBackend {
    records: RwLock<HashMap<String, MailboxRecord>>,
}

impl MemoryMailboxBackend {
    /// Create a new in-memory mailbox backend.
    pub fn new() -> Self {
        Self {
            records: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryMailboxBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MailboxBackendOps for MemoryMailboxBackend {
    async fn store(&self, record: MailboxRecord) -> Result<(), BackendError> {
        self.records
            .write()
            .insert(record.request_id.clone(), record);
        Ok(())
    }

    async fn get(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError> {
        Ok(self.records.read().get(request_id).cloned())
    }

    async fn list(&self) -> Result<Vec<MailboxRecord>, BackendError> {
        Ok(self.records.read().values().cloned().collect())
    }

    async fn delete(&self, request_id: &str) -> Result<bool, BackendError> {
        Ok(self.records.write().remove(request_id).is_some())
    }

    async fn get_and_delete(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError> {
        Ok(self.records.write().remove(request_id))
    }

    async fn cleanup_expired(&self, max_age_secs: u64) -> Result<usize, BackendError> {
        let cutoff = now_secs().saturating_sub(max_age_secs);
        let mut records = self.records.write();
        let initial_len = records.len();
        records.retain(|_, r| r.created >= cutoff);
        Ok(initial_len - records.len())
    }
}

/// In-memory queue backend using per-model VecDeques protected by RwLock.
#[derive(Debug)]
pub struct MemoryQueueBackend {
    queues: RwLock<HashMap<String, VecDeque<QueueRecord>>>,
}

impl MemoryQueueBackend {
    /// Create a new in-memory queue backend.
    pub fn new() -> Self {
        Self {
            queues: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryQueueBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl QueueBackendOps for MemoryQueueBackend {
    async fn enqueue(&self, record: QueueRecord) -> Result<(), BackendError> {
        self.queues
            .write()
            .entry(record.model.clone())
            .or_default()
            .push_back(record);
        Ok(())
    }

    async fn dequeue(&self, model: &str, max: usize) -> Result<Vec<QueueRecord>, BackendError> {
        let mut queues = self.queues.write();
        let queue = queues.entry(model.to_string()).or_default();
        
        let mut result = Vec::with_capacity(max.min(queue.len()));
        for _ in 0..max {
            if let Some(record) = queue.pop_front() {
                result.push(record);
            } else {
                break;
            }
        }
        Ok(result)
    }

    async fn len(&self, model: &str) -> Result<usize, BackendError> {
        Ok(self
            .queues
            .read()
            .get(model)
            .map(|q| q.len())
            .unwrap_or(0))
    }

    async fn list(&self, model: Option<&str>) -> Result<Vec<QueueRecord>, BackendError> {
        let queues = self.queues.read();
        if let Some(model) = model {
            Ok(queues
                .get(model)
                .map(|q| q.iter().cloned().collect())
                .unwrap_or_default())
        } else {
            Ok(queues
                .values()
                .flat_map(|q| q.iter().cloned())
                .collect())
        }
    }

    async fn clear(&self, model: &str) -> Result<usize, BackendError> {
        let mut queues = self.queues.write();
        if let Some(queue) = queues.get_mut(model) {
            let len = queue.len();
            queue.clear();
            Ok(len)
        } else {
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_mailbox_store_and_get() {
        let backend = MemoryMailboxBackend::new();
        
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
    async fn test_memory_mailbox_get_and_delete() {
        let backend = MemoryMailboxBackend::new();
        
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
    async fn test_memory_queue_enqueue_dequeue() {
        let backend = MemoryQueueBackend::new();
        
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
    async fn test_memory_queue_clear() {
        let backend = MemoryQueueBackend::new();
        
        for i in 0..5 {
            backend.enqueue(QueueRecord {
                id: format!("req-{}", i),
                model: "test-model".to_string(),
                queued_at: i as u64,
                request: serde_json::json!({}),
            }).await.unwrap();
        }

        assert_eq!(backend.len("test-model").await.unwrap(), 5);
        
        let cleared = backend.clear("test-model").await.unwrap();
        assert_eq!(cleared, 5);
        assert_eq!(backend.len("test-model").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_memory_mailbox_cleanup_expired() {
        let backend = MemoryMailboxBackend::new();
        let now = now_secs();
        
        // Add an old record
        backend.store(MailboxRecord {
            request_id: "old".to_string(),
            model: "test".to_string(),
            created: now - 3600, // 1 hour ago
            status: "completed".to_string(),
            response: None,
        }).await.unwrap();
        
        // Add a recent record
        backend.store(MailboxRecord {
            request_id: "new".to_string(),
            model: "test".to_string(),
            created: now,
            status: "completed".to_string(),
            response: None,
        }).await.unwrap();

        // Cleanup records older than 30 minutes
        let removed = backend.cleanup_expired(1800).await.unwrap();
        assert_eq!(removed, 1);
        
        // Old should be gone, new should remain
        assert!(backend.get("old").await.unwrap().is_none());
        assert!(backend.get("new").await.unwrap().is_some());
    }
}
