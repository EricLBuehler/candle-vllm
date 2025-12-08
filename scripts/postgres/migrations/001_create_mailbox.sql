-- Migration: Create mailbox_records table
-- Version: 001
-- Description: Stores completed inference request results for async retrieval

-- Create the mailbox table
CREATE TABLE IF NOT EXISTS mailbox_records (
    request_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    created BIGINT NOT NULL,
    status TEXT NOT NULL,
    response JSONB
);

-- Index for cleanup operations (finding old records)
CREATE INDEX IF NOT EXISTS idx_mailbox_created ON mailbox_records(created);

-- Index for listing by model
CREATE INDEX IF NOT EXISTS idx_mailbox_model ON mailbox_records(model);

-- Index for status queries
CREATE INDEX IF NOT EXISTS idx_mailbox_status ON mailbox_records(status);

COMMENT ON TABLE mailbox_records IS 'Stores completed inference request results for async retrieval';
COMMENT ON COLUMN mailbox_records.request_id IS 'Unique identifier for the request (UUID)';
COMMENT ON COLUMN mailbox_records.model IS 'Model name used for inference';
COMMENT ON COLUMN mailbox_records.created IS 'Unix timestamp when the record was created';
COMMENT ON COLUMN mailbox_records.status IS 'Request status: completed, streaming, validation_error, model_error, internal_error';
COMMENT ON COLUMN mailbox_records.response IS 'JSON response payload (null for non-completed statuses)';
