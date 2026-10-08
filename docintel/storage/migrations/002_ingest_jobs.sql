-- Queue bookkeeping across tenants (identifiers only, no document content), used to recover jobs after a
-- restart and to report queue depth. Tenant content tables keep row-level security.
CREATE TABLE ingest_jobs (
    document_id  uuid PRIMARY KEY,
    tenant_id    text NOT NULL,
    state        text NOT NULL DEFAULT 'queued',    -- queued | running
    attempts     int NOT NULL DEFAULT 0,
    enqueued_at  timestamptz NOT NULL DEFAULT now(),
    started_at   timestamptz
);
CREATE INDEX ingest_jobs_state ON ingest_jobs (state, enqueued_at);
