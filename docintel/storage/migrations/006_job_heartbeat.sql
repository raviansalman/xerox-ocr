-- Processing jobs report progress: a running job updates heartbeat_at every DOCINTEL_JOB_HEARTBEAT_SEC, so a job
-- whose worker died is requeued within minutes instead of after the whole task time limit.
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS heartbeat_at timestamptz;
