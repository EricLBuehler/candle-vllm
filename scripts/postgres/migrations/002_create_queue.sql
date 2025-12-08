-- Migration: Create queue_records table
-- Version: 002
-- Description: Stores pending inference requests in a FIFO queue per model

-- Create the queue table with composite primary key for efficient ordering
CREATE TABLE IF NOT EXISTS queue_records (
    id TEXT NOT NULL,
    model TEXT NOT NULL,
    queued_at BIGINT NOT NULL,
    request JSONB NOT NULL,
    PRIMARY KEY (model, queued_at, id)
);

-- Index for efficient queue operations by model and time
CREATE INDEX IF NOT EXISTS idx_queue_model_time ON queue_records(model, queued_at);

-- Index for counting by model
CREATE INDEX IF NOT EXISTS idx_queue_model ON queue_records(model);

COMMENT ON TABLE queue_records IS 'FIFO queue for pending inference requests per model';
COMMENT ON COLUMN queue_records.id IS 'Unique request identifier';
COMMENT ON COLUMN queue_records.model IS 'Target model for the request';
COMMENT ON COLUMN queue_records.queued_at IS 'Unix timestamp when the request was queued';
COMMENT ON COLUMN queue_records.request IS 'Full request payload as JSON';

-- Function for atomic dequeue with SKIP LOCKED
-- Usage: SELECT * FROM dequeue_records('model_name', 10);
CREATE OR REPLACE FUNCTION dequeue_records(target_model TEXT, max_count INTEGER)
RETURNS SETOF queue_records
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH selected AS (
        SELECT id, model, queued_at, request
        FROM queue_records
        WHERE model = target_model
        ORDER BY queued_at ASC
        LIMIT max_count
        FOR UPDATE SKIP LOCKED
    )
    DELETE FROM queue_records
    WHERE (model, queued_at, id) IN (SELECT model, queued_at, id FROM selected)
    RETURNING *;
END;
$$;

COMMENT ON FUNCTION dequeue_records IS 'Atomically dequeues up to max_count records for a model using SKIP LOCKED for concurrency';
