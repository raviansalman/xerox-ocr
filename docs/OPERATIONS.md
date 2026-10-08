# Operations

## Deploy with Docker Compose

```bash
cp deploy/.env.example deploy/.env       # passwords, DOCINTEL_METRICS_TOKEN; see docs/CONFIGURATION.md
docintel keygen                          # prints a key once and the sha256 to store
cp deploy/api_keys.example.json deploy/api_keys.json   # each key's sha256, tenant and roles
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
```

Compose refuses to start without the passwords and the metrics token. Only the API port is published. The API
applies database migrations at start (advisory-locked, so replicas starting together are safe; tested with four
processes on an empty database). `GET /health/live` reports the process; `GET /health/ready` checks the database,
the vector index, the embedder (model and dimension), that the object store is writable, and the job queue
(including stalled jobs), and returns 503 when any check fails.

**Images.** `deploy/Dockerfile` (API and workers: Ubuntu 24.04, Python 3.12, Tesseract 5, LibreOffice; runs as a
non-root user) and `deploy/Dockerfile.embedder` (embedding and optional reranker service). The embedder image takes
its models from `deploy/models/<key>/` when present (for networks without access to the model hub) or downloads
them, and verifies them against the checksums in `docintel/model_registry.yaml`. It installs CPU-only PyTorch from
the PyTorch index by default; `--build-arg TORCH_INDEX=` uses PyPI instead, which bundles CUDA libraries (about
3 GB more). Behind a TLS-intercepting proxy, pass its CA as a build secret (`--secret id=ca,src=ca.crt`); it is not
kept in the image.

**Verified on a clean machine:** CI builds both images from the repository on a fresh GitHub runner (the embedder
downloads and checksum-verifies its model), starts the whole compose stack and runs `scripts/verify_deployment.py`:
23 of 23 steps pass (upload native PDF, scanned PDF, image, DOCX, XLSX; multi-page processing; exact phrase,
identifier, typo, meaning with the real model, concept, entity, filter, count, sum, date range; evidence spans;
grounded answer; cross-tenant access; restarting PostgreSQL, Milvus, Redis and the embedder under a running
worker; reprocessing; deletion from every table and the vector index; worker metrics and readiness).

## Processing modes

| `DOCINTEL_TASK_MODE` | Use |
|---|---|
| `celery` | Production. Jobs go through Redis to worker containers; a job is acknowledged only after it finishes, so a lost worker's job is redelivered. Jobs wait in Redis while no worker runs. |
| `thread` | One machine without Redis. Jobs run in a thread pool in the API process; unfinished jobs are recovered from `ingest_jobs` at start. One API process only (`docintel serve` refuses `--workers` above 1): processes cannot see each other's in-memory queues, and at 50,000 documents two processes reaped each other's waiting backlog and processed 11% of it twice. |
| `inline` | Tests only. |

In every mode a reaper in the API requeues jobs that stopped making progress for longer than
`DOCINTEL_TASK_TIME_LIMIT_SEC` (a worker died, or a failure could not be recorded, for example on a full disk) and
marks a document `failed` with the reason after five attempts.

## Sizing and scaling

| Resource | Guidance |
|---|---|
| Workers | `docker compose ... up -d --scale worker=N`; each runs `DOCINTEL_WORKER_CONCURRENCY` processes, give it that many cores. |
| OCR | Scanned pages dominate processing: about 1 to 2 seconds per page per core at 250 DPI. |
| Embedder | CPU embedding is the next cost. One vector per passage, plus one per document with three or more passages. Run it on a GPU (`DOCINTEL_EMBEDDER_DEVICE=cuda`) for large backfills. |
| PostgreSQL | Measured at 50,000 short documents: postings 1.45 GB (6.3 million rows), units 245 MB, fields 70 MB, blocks 51 MB. Budget roughly 30 KB per short document, more for long ones. Raise `DOCINTEL_DB_POOL_MAX` with replicas; give `shared_buffers` enough memory for the postings index. Use a database with `C.UTF-8` collation (`deploy/postgres-init.sql` does). |
| Milvus | 768 floats per vector, about 3 KB plus the HNSW graph. |
| API | Stateless; add replicas. Queries never process documents in Celery mode. |

## Performance

Measured on one 4-core, 16 GB VM that runs everything (API, processing, PostgreSQL, Milvus, the CPU embedder and
the load generator itself), with 50,000 synthetic documents in one tenant. Latency is the server round trip. Raw
results are in [benchmarks/](benchmarks/); the scripts are `scripts/load_test.py` and `scripts/concurrency_test.py`.

**Ingestion**, 50,000 documents (a third each native PDF, DOCX and text; 2% scanned PDFs through OCR), real
`all-mpnet-base-v2` embeddings on CPU: **650 documents per minute**, 0 failures (76.9 minutes from the first
upload to the last document indexed). Scanned pages dominate cost (1 to 2 s per page per core), so a corpus with
more scans is slower; an earlier run of 2,000 documents with 10% scanned measured 593 per minute. Storage at
50,000 documents: PostgreSQL 2.3 GB (postings 1.6 GB in 6.3 million rows, 100,000 units 258 MB), originals 2.5 GB.

**One user**, p50 / p95 in ms, 50,000 documents:

| Question kind | Semantic on | Semantic off |
|---|---|---|
| Identifier | 84 / 219 | 74 / 219 |
| Quoted phrase | 172 / 215 | 181 / 211 |
| Keywords | 197 / 296 | 233 / 265 |
| Typo | 90 / 106 | 73 / 89 |
| Meaning (lexical only when semantic is off) | 224 / 315 | 237 / 283 |
| Concept | 498 / 588 | 455 / 498 |
| Count | 121 / 192 | 118 / 179 |
| Total over all invoices (16,667) | 359 / 396 | 323 / 355 |
| Filter | 91 / 164 | 95 / 131 |
| Field lookup | 335 / 417 | 320 / 356 |

**Concurrent users**, a mix of all question kinds, closed loop (each user asks again as soon as it has an
answer), 60 s per level after a 10 s warm-up, two API processes, 0 errors at every level:

| Users | Semantic on: queries/s | p50 / p95 / p99 ms | Semantic off: queries/s | p50 / p95 / p99 ms |
|---|---|---|---|---|
| 1 | 5.5 | 169 / 443 / 545 | 6.0 | 168 / 409 / 461 |
| 8 | 15.5 | 380 / 1,358 / 1,700 | 18.2 | 387 / 1,066 / 1,395 |
| 16 | 17.4 | 721 / 2,130 / 3,049 | 21.1 | 596 / 1,758 / 2,189 |
| 32 | 18.6 | 1,464 / 3,770 / 4,809 | 21.3 | 1,283 / 3,378 / 4,457 |
| 64 | 18.4 | 2,967 / 7,222 / 9,015 | 21.2 | 2,596 / 6,559 / 7,681 |
| 100 | 17.4 | 5,052 / 10,287 / 12,305 | 19.5 | 4,446 / 9,271 / 10,772 |

**Bottleneck: CPU, and within it PostgreSQL.** The host is at 87 to 98% CPU from 8 users on; after that,
throughput stays flat and latency grows with the queue. At saturation PostgreSQL uses about 2 of the 4 cores
(postings and structured queries), the API processes 1 to 1.3, the query embedder about 0.4 and Milvus under 0.1
(semantic on). Database connections are not the limit (10 to 12 active on average). So: one 4-core machine
serves about 15 to 21 questions per second; p95 stays under 500 ms only for a single user. Capacity grows with
cores for PostgreSQL first (a separate database host, or more cores and `shared_buffers` for the postings index),
then API replicas; the embedder and Milvus are not close to their limits at this size.

Reproduce:

```bash
python scripts/load_test.py --url http://localhost:8000 --key $KEY --count 50000 --scanned 0.02 --query-concurrency 1
python scripts/load_test.py --url http://localhost:8000 --key $KEY --queries-only --query-concurrency 1
python scripts/concurrency_test.py --url http://localhost:8000 --key $KEY --levels 1,8,16,32,64,100 --duration 60 \
    --proc api="multiprocessing.spawn" --proc postgres="postgres:" --proc embedder="docintel.embedder.app" \
    --container milvus=<milvus container> --pg-dsn <admin connection> --pg-db docintel
```

## Upgrading from v1

1. Back up PostgreSQL and the object store.
2. Deploy the new images. The API applies migrations 003 to 005 at start (`docintel migrate` does it by hand);
   existing documents keep their data and are reported `stale` ("processed before versioning").
3. Rebuild them into the canonical model and the postings index: `docintel reprocess --tenant T --stale --wait`
   per tenant. Until a document is reprocessed, it is not found by the v2 retrievers; set
   `DOCINTEL_RETRIEVAL_ENGINE=v1` (or `shadow`) during the backfill if that matters.
4. Run the evaluation set (`docintel eval run`) before switching the engine to `v2`.

Tested: a database at migration 002 with v1 data, upgraded and backfilled (`tests/integration/test_migrations.py`).
Every migration has a down script: `docintel rollback --to 004` (a prefix must name exactly one migration; `none`
reverts everything).

## Backups and recovery

* PostgreSQL and the object store are the system of record; back them up together.
* Milvus can be rebuilt by reprocessing. Changing the embedding model marks every document stale; reprocess them.
* After a crash, thread mode recovers unfinished jobs at start; Celery redelivers them; the reaper handles anything
  left behind.

## Monitoring

`GET /metrics` (bearer `DOCINTEL_METRICS_TOKEN`) on the API: queries by intent and outcome, query and retriever
latency, retriever failures, uploads, queue depth, answers by provider and outcome. Celery workers serve processing
metrics (documents and pages by kind and outcome, time per stage) on `DOCINTEL_WORKER_METRICS_PORT` (9100 in the
compose file, internal network only). Logs are JSON with request ids; document text and keys are not logged.

## Testing

```bash
pytest -q tests/unit                                   # no services
export DOCINTEL_TEST_INTEGRATION=1 DOCINTEL_TEST_DATABASE_URL=postgresql://docintel_app:...@localhost/docintel_test
export DOCINTEL_TEST_MILVUS_URI=http://localhost:19530           # omit for the in-memory index
export DOCINTEL_TEST_ADMIN_DATABASE_URL=postgresql://postgres:...@localhost/postgres   # migration tests
export DOCINTEL_TEST_REDIS_URL=redis://localhost:6379/15         # Celery test
export DOCINTEL_TEST_E2E=1                                       # browser tests
pytest -q                                                        # stand-in embedder
DOCINTEL_TEST_VECTORS=disabled pytest -q                         # no vector index at all
DOCINTEL_TEST_EMBEDDER_URL=http://localhost:8080 DOCINTEL_TEST_EMBEDDING_MODEL=all-mpnet-base-v2 pytest -q   # real model
```

The application test role must not be a superuser or have `BYPASSRLS`. CI runs the static checks, the unit tests,
the integration suite with stand-in vectors and with semantic search disabled, and the clean-machine deployment.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/health/ready` 503 with `object_store` failing | the storage volume is missing, read-only or owned by another user; the image creates `/data` for its user, so check bind mounts |
| Start fails with `invalid configuration: ...` | the message names the variable; production also refuses dev authentication, memory vectors, inline jobs and a missing metrics token |
| `database role is a superuser or has BYPASSRLS` | connect as the application role created by `deploy/postgres-init.sql` |
| Collation warning at start | prefix lookups expect a `C` collation database; create it as in `deploy/postgres-init.sql` |
| `embedder serves X, engine is configured for Y` | the embedding service and `DOCINTEL_EMBEDDING_MODEL` disagree; vectors from different models are never mixed |
| Documents stay `queued` | no worker consumes the queue (Celery) or the API process restarted (thread mode recovers at start); `queue.stalled` in readiness counts stuck jobs, which the reaper requeues |
| Documents `failed` | the `error` field says why (password-protected, corrupted, limits, disk full); fix the cause and `POST /api/v1/documents/{id}/reprocess` |
| Answers say "No sufficiently reliable evidence" | the engine abstained; `explain: true` shows the plan, the retrievers and their timings |
| Results after changing packs look old | documents are `stale`; `docintel reprocess --tenant T --stale` |
