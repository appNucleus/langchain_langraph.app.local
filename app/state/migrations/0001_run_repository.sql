CREATE TABLE IF NOT EXISTS app_runs (
    run_id UUID PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    execution_thread_id TEXT NOT NULL UNIQUE,
    request_hash CHAR(64) NOT NULL,
    request_hash_version SMALLINT NOT NULL,
    state_schema_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'pending', 'running', 'interrupted', 'completed', 'failed',
            'cancelled', 'expired', 'reconciling'
        )
    ),
    lease_owner UUID,
    lease_expires_at TIMESTAMPTZ,
    fencing_token BIGINT,
    checkpoint_id TEXT,
    response_payload JSONB,
    termination_reason TEXT,
    error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    history_committed_at TIMESTAMPTZ,
    resume_token_version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_app_runs_conversation_created
    ON app_runs (conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_runs_status_updated
    ON app_runs (status, updated_at);

CREATE TABLE IF NOT EXISTS app_conversation_leases (
    conversation_id TEXT PRIMARY KEY,
    run_id UUID NOT NULL,
    lease_owner UUID NOT NULL,
    fencing_token BIGINT NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_conversation_leases_expiry
    ON app_conversation_leases (lease_expires_at);

ALTER TABLE IF EXISTS app_conversation_messages
    ADD COLUMN IF NOT EXISTS run_id UUID;
ALTER TABLE IF EXISTS app_conversation_messages
    ADD COLUMN IF NOT EXISTS message_kind TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS uq_app_conversation_messages_run_kind
    ON app_conversation_messages (thread_id, run_id, message_kind)
    WHERE run_id IS NOT NULL AND message_kind IS NOT NULL;
