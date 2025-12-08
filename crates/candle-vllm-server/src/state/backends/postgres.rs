//! PostgreSQL storage backend implementation.
//!
//! This backend uses sqlx for async PostgreSQL operations.
//! It provides full ACID transactions with row-level locking for atomic dequeue.

use async_trait::async_trait;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use tracing::{debug, info};

use crate::state::backend_traits::{
    BackendError, MailboxBackendOps, MailboxRecord, QueueBackendOps, QueueRecord, now_secs,
};

/// PostgreSQL mailbox backend using sqlx connection pool.
pub struct PostgresMailboxBackend {
    pool: PgPool,
}

impl PostgresMailboxBackend {
    /// Create a new PostgreSQL mailbox backend with the given connection URL.
    pub async fn new(url: &str) -> Result<Self, BackendError> {
        info!("Initializing PostgreSQL mailbox backend");

        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(url)
            .await
            .map_err(|e| BackendError::connection(format!("failed to connect to PostgreSQL: {}", e)))?;

        // Initialize schema
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS mailbox_records (
                request_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                created BIGINT NOT NULL,
                status TEXT NOT NULL,
                response JSONB
            )
            "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| BackendError::connection(format!("failed to create schema: {}", e)))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_mailbox_created ON mailbox_records(created)",
        )
        .execute(&pool)
        .await
        .map_err(|e| BackendError::connection(format!("failed to create index: {}", e)))?;

        info!("PostgreSQL mailbox backend initialized");
        Ok(Self { pool })
    }
}

#[async_trait]
impl MailboxBackendOps for PostgresMailboxBackend {
    async fn store(&self, record: MailboxRecord) -> Result<(), BackendError> {
        debug!(request_id = %record.request_id, "Storing mailbox record");

        sqlx::query(
            r#"
            INSERT INTO mailbox_records (request_id, model, created, status, response)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (request_id) DO UPDATE SET
                model = EXCLUDED.model,
                created = EXCLUDED.created,
                status = EXCLUDED.status,
                response = EXCLUDED.response
            "#,
        )
        .bind(&record.request_id)
        .bind(&record.model)
        .bind(record.created as i64)
        .bind(&record.status)
        .bind(&record.response)
        .execute(&self.pool)
        .await
        .map_err(|e| BackendError::other(format!("failed to store record: {}", e)))?;

        Ok(())
    }

    async fn get(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError> {
        debug!(request_id = %request_id, "Getting mailbox record");

        let row = sqlx::query(
            r#"
            SELECT request_id, model, created, status, response
            FROM mailbox_records
            WHERE request_id = $1
            "#,
        )
        .bind(request_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| BackendError::other(format!("failed to get record: {}", e)))?;

        Ok(row.map(|r| MailboxRecord {
            request_id: r.get("request_id"),
            model: r.get("model"),
            created: r.get::<i64, _>("created") as u64,
            status: r.get("status"),
            response: r.get("response"),
        }))
    }

    async fn list(&self) -> Result<Vec<MailboxRecord>, BackendError> {
        debug!("Listing all mailbox records");

        let rows = sqlx::query(
            r#"
            SELECT request_id, model, created, status, response
            FROM mailbox_records
            ORDER BY created DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| BackendError::other(format!("failed to list records: {}", e)))?;

        Ok(rows
            .into_iter()
            .map(|r| MailboxRecord {
                request_id: r.get("request_id"),
                model: r.get("model"),
                created: r.get::<i64, _>("created") as u64,
                status: r.get("status"),
                response: r.get("response"),
            })
            .collect())
    }

    async fn delete(&self, request_id: &str) -> Result<bool, BackendError> {
        debug!(request_id = %request_id, "Deleting mailbox record");

        let result = sqlx::query("DELETE FROM mailbox_records WHERE request_id = $1")
            .bind(request_id)
            .execute(&self.pool)
            .await
            .map_err(|e| BackendError::other(format!("failed to delete record: {}", e)))?;

        Ok(result.rows_affected() > 0)
    }

    async fn get_and_delete(&self, request_id: &str) -> Result<Option<MailboxRecord>, BackendError> {
        debug!(request_id = %request_id, "Getting and deleting mailbox record");

        // Use RETURNING for atomic get-and-delete
        let row = sqlx::query(
            r#"
            DELETE FROM mailbox_records
            WHERE request_id = $1
            RETURNING request_id, model, created, status, response
            "#,
        )
        .bind(request_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| BackendError::other(format!("failed to get and delete record: {}", e)))?;

        Ok(row.map(|r| MailboxRecord {
            request_id: r.get("request_id"),
            model: r.get("model"),
            created: r.get::<i64, _>("created") as u64,
            status: r.get("status"),
            response: r.get("response"),
        }))
    }

    async fn cleanup_expired(&self, max_age_secs: u64) -> Result<usize, BackendError> {
        let cutoff = now_secs().saturating_sub(max_age_secs) as i64;
        debug!(cutoff = cutoff, "Cleaning up expired mailbox records");

        let result = sqlx::query("DELETE FROM mailbox_records WHERE created < $1")
            .bind(cutoff)
            .execute(&self.pool)
            .await
            .map_err(|e| BackendError::other(format!("failed to cleanup records: {}", e)))?;

        let deleted = result.rows_affected() as usize;
        info!(deleted = deleted, "Cleaned up expired mailbox records");
        Ok(deleted)
    }
}

/// PostgreSQL queue backend using sqlx connection pool.
pub struct PostgresQueueBackend {
    pool: PgPool,
}

impl PostgresQueueBackend {
    /// Create a new PostgreSQL queue backend with the given connection URL.
    pub async fn new(url: &str) -> Result<Self, BackendError> {
        info!("Initializing PostgreSQL queue backend");

        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(url)
            .await
            .map_err(|e| BackendError::connection(format!("failed to connect to PostgreSQL: {}", e)))?;

        // Initialize schema
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS queue_records (
                id TEXT NOT NULL,
                model TEXT NOT NULL,
                queued_at BIGINT NOT NULL,
                request JSONB NOT NULL,
                PRIMARY KEY (model, queued_at, id)
            )
            "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| BackendError::connection(format!("failed to create schema: {}", e)))?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_queue_model_time ON queue_records(model, queued_at)",
        )
        .execute(&pool)
        .await
        .map_err(|e| BackendError::connection(format!("failed to create index: {}", e)))?;

        info!("PostgreSQL queue backend initialized");
        Ok(Self { pool })
    }
}

#[async_trait]
impl QueueBackendOps for PostgresQueueBackend {
    async fn enqueue(&self, record: QueueRecord) -> Result<(), BackendError> {
        debug!(id = %record.id, model = %record.model, "Enqueueing record");

        sqlx::query(
            r#"
            INSERT INTO queue_records (id, model, queued_at, request)
            VALUES ($1, $2, $3, $4)
            "#,
        )
        .bind(&record.id)
        .bind(&record.model)
        .bind(record.queued_at as i64)
        .bind(&record.request)
        .execute(&self.pool)
        .await
        .map_err(|e| BackendError::other(format!("failed to enqueue record: {}", e)))?;

        Ok(())
    }

    async fn dequeue(&self, model: &str, max: usize) -> Result<Vec<QueueRecord>, BackendError> {
        debug!(model = %model, max = max, "Dequeueing records");

        // Use FOR UPDATE SKIP LOCKED for atomic dequeue in concurrent environments
        // This query selects and deletes in one statement using a CTE
        let rows = sqlx::query(
            r#"
            WITH selected AS (
                SELECT id, model, queued_at, request
                FROM queue_records
                WHERE model = $1
                ORDER BY queued_at ASC
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            DELETE FROM queue_records
            WHERE (model, queued_at, id) IN (SELECT model, queued_at, id FROM selected)
            RETURNING id, model, queued_at, request
            "#,
        )
        .bind(model)
        .bind(max as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| BackendError::other(format!("failed to dequeue records: {}", e)))?;

        let records = rows
            .into_iter()
            .map(|r| QueueRecord {
                id: r.get("id"),
                model: r.get("model"),
                queued_at: r.get::<i64, _>("queued_at") as u64,
                request: r.get("request"),
            })
            .collect();

        Ok(records)
    }

    async fn len(&self, model: &str) -> Result<usize, BackendError> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM queue_records WHERE model = $1")
            .bind(model)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| BackendError::other(format!("failed to count records: {}", e)))?;

        let count: i64 = row.get("count");
        Ok(count as usize)
    }

    async fn list(&self, model: Option<&str>) -> Result<Vec<QueueRecord>, BackendError> {
        let rows = if let Some(model) = model {
            sqlx::query(
                r#"
                SELECT id, model, queued_at, request
                FROM queue_records
                WHERE model = $1
                ORDER BY queued_at ASC
                "#,
            )
            .bind(model)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query(
                r#"
                SELECT id, model, queued_at, request
                FROM queue_records
                ORDER BY queued_at ASC
                "#,
            )
            .fetch_all(&self.pool)
            .await
        }
        .map_err(|e| BackendError::other(format!("failed to list records: {}", e)))?;

        Ok(rows
            .into_iter()
            .map(|r| QueueRecord {
                id: r.get("id"),
                model: r.get("model"),
                queued_at: r.get::<i64, _>("queued_at") as u64,
                request: r.get("request"),
            })
            .collect())
    }

    async fn clear(&self, model: &str) -> Result<usize, BackendError> {
        debug!(model = %model, "Clearing queue for model");

        let result = sqlx::query("DELETE FROM queue_records WHERE model = $1")
            .bind(model)
            .execute(&self.pool)
            .await
            .map_err(|e| BackendError::other(format!("failed to clear queue: {}", e)))?;

        let deleted = result.rows_affected() as usize;
        info!(model = %model, deleted = deleted, "Cleared queue");
        Ok(deleted)
    }
}

// Note: Integration tests for PostgreSQL require a running PostgreSQL instance.
// These tests are better run as integration tests with testcontainers or similar.
#[cfg(test)]
mod tests {
    use super::*;

    // This test requires a PostgreSQL instance and is marked ignore by default
    #[tokio::test]
    #[ignore = "requires PostgreSQL instance"]
    async fn test_postgres_mailbox_store_and_get() {
        let url = std::env::var("TEST_POSTGRES_URL")
            .unwrap_or_else(|_| "postgres://localhost/test_candle_vllm".to_string());
        
        let backend = PostgresMailboxBackend::new(&url).await.unwrap();

        let record = MailboxRecord {
            request_id: format!("test-{}", uuid::Uuid::new_v4()),
            model: "gpt-4".to_string(),
            created: now_secs(),
            status: "completed".to_string(),
            response: Some(serde_json::json!({"text": "hello"})),
        };

        backend.store(record.clone()).await.unwrap();

        let retrieved = backend.get(&record.request_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().request_id, record.request_id);

        // Cleanup
        backend.delete(&record.request_id).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "requires PostgreSQL instance"]
    async fn test_postgres_queue_enqueue_dequeue() {
        let url = std::env::var("TEST_POSTGRES_URL")
            .unwrap_or_else(|_| "postgres://localhost/test_candle_vllm".to_string());
        
        let backend = PostgresQueueBackend::new(&url).await.unwrap();
        let test_model = format!("test-model-{}", uuid::Uuid::new_v4());

        let record1 = QueueRecord {
            id: "req-1".to_string(),
            model: test_model.clone(),
            queued_at: 1000,
            request: serde_json::json!({"prompt": "hello"}),
        };
        let record2 = QueueRecord {
            id: "req-2".to_string(),
            model: test_model.clone(),
            queued_at: 1001,
            request: serde_json::json!({"prompt": "world"}),
        };

        backend.enqueue(record1).await.unwrap();
        backend.enqueue(record2).await.unwrap();

        assert_eq!(backend.len(&test_model).await.unwrap(), 2);

        let dequeued = backend.dequeue(&test_model, 1).await.unwrap();
        assert_eq!(dequeued.len(), 1);
        assert_eq!(dequeued[0].id, "req-1"); // FIFO order

        // Cleanup
        backend.clear(&test_model).await.unwrap();
    }
}
