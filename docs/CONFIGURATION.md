# Configuration

Every setting is an environment variable with the `DOCINTEL_` prefix, defined once in `docintel/config.py`. There
are no defaults for hosts, credentials or URLs: a required setting that is missing stops the process at start with
the variable named (`validate_for_startup`). A unit test checks that every setting is listed here.

## Environment profiles

`DOCINTEL_ENVIRONMENT` is `development`, `test`, `staging` or `production`. Staging and production refuse to start
with: authentication off (`AUTH_MODE=dev`), no API keys, the in-memory vector index, inline processing, URL
ingestion allowed to reach private addresses, or `/metrics` without a token. They also refuse a database role that
can bypass row-level security.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| **General** | | |
| `DOCINTEL_ENVIRONMENT` | `development` | profile, see above |
| `DOCINTEL_LOG_LEVEL` | `INFO` | |
| `DOCINTEL_LOG_JSON` | `true` | JSON logs with request ids |
| **Storage** | | |
| `DOCINTEL_DATABASE_URL` | required | PostgreSQL URL of the application role (not a superuser, no `BYPASSRLS`) |
| `DOCINTEL_DB_POOL_MIN`, `DOCINTEL_DB_POOL_MAX` | 2, 20 | connection pool per process |
| `DOCINTEL_STORAGE_DIR` | `./data/objects` (images: `/data/objects`) | object store for originals |
| `DOCINTEL_AUTO_MIGRATE` | `true` | the API applies pending migrations at start (advisory-locked) |
| **Vector index and embeddings** | | |
| `DOCINTEL_VECTOR_BACKEND` | `milvus` | `milvus`, `memory` (tests) or `disabled` (no semantic search; everything else works) |
| `DOCINTEL_MILVUS_URI`, `DOCINTEL_MILVUS_TOKEN` | required with Milvus | |
| `DOCINTEL_MILVUS_COLLECTION_PREFIX` | `docintel_chunks` | collection name prefix; the model and dimension are appended |
| `DOCINTEL_MILVUS_CONSISTENCY` | `Bounded` | Milvus read consistency |
| `DOCINTEL_EMBEDDER_URL` | required unless disabled | the embedding service |
| `DOCINTEL_EMBEDDING_MODEL` | `all-mpnet-base-v2` | registry key; the service must serve the same model and dimension |
| `DOCINTEL_EMBED_BATCH_SIZE`, `DOCINTEL_EMBED_TIMEOUT_SEC` | 32, 120 | |
| `DOCINTEL_RERANKER_MODEL` | empty (off) | registry key of a cross-encoder served by the embedding service |
| `DOCINTEL_RERANK_TOP_N` | 40 | results considered by the reranker |
| **Answers** | | |
| `DOCINTEL_ANSWER_PROVIDER` | `extractive` | `extractive`, `anthropic` or `disabled` |
| `DOCINTEL_ANSWER_MODEL` | `claude-opus-5-5` | Claude model for the `anthropic` provider |
| `DOCINTEL_ANSWER_API_KEY` | empty | key for the provider; when empty the Anthropic SDK uses its own environment (`ANTHROPIC_API_KEY`, a profile) |
| `DOCINTEL_ANSWER_BASE_URL` | empty | API gateway or proxy URL |
| `DOCINTEL_ANSWER_EFFORT` | `medium` | model effort level |
| `DOCINTEL_ANSWER_FALLBACKS` | `true` | server-side refusal fallback (Claude API only; set false behind gateways that do not support it) |
| `DOCINTEL_ANSWER_TIMEOUT_SEC` | 60 | |
| `DOCINTEL_ANSWER_MAX_EVIDENCE` | 12 | evidence items given to the answer layer |
| **Jobs** | | |
| `DOCINTEL_TASK_MODE` | `thread` | `celery` (multi-node), `thread` (one machine) or `inline` (tests) |
| `DOCINTEL_REDIS_URL` | required with Celery | broker |
| `DOCINTEL_WORKER_THREADS` | CPU count | thread-mode workers |
| `DOCINTEL_INGEST_QUEUE` | `docintel.ingest` | Celery queue |
| `DOCINTEL_TASK_TIME_LIMIT_SEC` | 1800 | Celery hard limit per document |
| `DOCINTEL_JOB_HEARTBEAT_SEC` | 30 | A running job marks itself alive this often; one silent for 4 beats (at least 2 minutes) is requeued |
| **Processing** | | |
| `DOCINTEL_OCR_LANGUAGES` | `eng` | Tesseract languages (the language packs must be installed) |
| `DOCINTEL_OCR_DPI` | 250 | render resolution for scanned pages |
| `DOCINTEL_OCR_PAGE_TIMEOUT_SEC` | 180 | OCR time limit per page; a page over it fails the document with the reason |
| `DOCINTEL_OCR_WORKERS` | CPU count | pages OCR'd in parallel per document |
| `DOCINTEL_SIGNATURE_DETECTION` | `true` | detect signature marks on scans |
| `DOCINTEL_MAX_UPLOAD_BYTES` | 200 MB | per file |
| `DOCINTEL_MAX_REQUEST_BYTES` | 1 GB | per upload request, refused before the body is spooled to disk (other requests: 1 MB) |
| `DOCINTEL_MAX_PAGES`, `DOCINTEL_MAX_SHEET_ROWS` | 2000, 20000 | |
| `DOCINTEL_MAX_IMAGE_PIXELS` | 120 million | decompression-bomb guard |
| `DOCINTEL_MAX_ARCHIVE_UNCOMPRESSED_BYTES`, `DOCINTEL_MAX_ARCHIVE_RATIO` | 1 GB, 200 | Office container guards |
| `DOCINTEL_MAX_ATTACHMENT_DEPTH` | 2 | nested e-mail attachments |
| `DOCINTEL_SOFFICE_PATH`, `DOCINTEL_CONVERT_TIMEOUT_SEC` | `soffice`, 180 | LibreOffice conversion |
| `DOCINTEL_URL_FETCH_ENABLED` | `false` | URL ingestion |
| `DOCINTEL_URL_ALLOWED_HOSTS` | empty | comma-separated host allow-list for URL ingestion |
| `DOCINTEL_URL_ALLOW_PRIVATE` | `false` | allow private addresses (development only; refused in production) |
| **Search** | | |
| `DOCINTEL_RETRIEVAL_ENGINE` | `v2` | `v2`, `v1` (previous engine) or `shadow` (serve v1, compare v2 in the logs) |
| `DOCINTEL_RESULT_LIMIT` | 20 | default results per query (at most 100) |
| `DOCINTEL_VECTOR_TOP_K`, `DOCINTEL_LEXICAL_TOP_K` | 100, 200 | candidates per retriever |
| `DOCINTEL_STRUCTURED_ID_LIMIT` | 5000 | documents a filter may pass to the other retrievers as an explicit list |
| `DOCINTEL_LEXICAL_AVG_UNIT_TERMS` | 120 | BM25 length normalization reference |
| `DOCINTEL_QUERY_TIMEOUT_MS` | 5000 | retrieval budget per query |
| `DOCINTEL_DEFAULT_PACKS` | `business,legal,finance` | domain packs for tenants without their own setting; checked at start |
| **Authentication** | | |
| `DOCINTEL_AUTH_MODE` | `keys` | `dev` disables authentication (refused in staging and production) |
| `DOCINTEL_API_KEYS_FILE` / `DOCINTEL_API_KEYS` | required | hashed API keys (file, or inline JSON) |
| `DOCINTEL_DEV_TENANT` | `default` | tenant used in `dev` mode |
| `DOCINTEL_CORS_ORIGINS` | empty | origins allowed to call the API from a browser |
| **Observability** | | |
| `DOCINTEL_METRICS_ENABLED` | `true` | serve `/metrics` |
| `DOCINTEL_METRICS_TOKEN` | required in production | bearer token for `/metrics` |
| `DOCINTEL_WORKER_METRICS_PORT` | empty (off) | Celery workers serve their processing metrics on this port (internal network; needs `PROMETHEUS_MULTIPROC_DIR`); the compose file uses 9100 |

Outside the `DOCINTEL_` prefix: `DOCINTEL_PACKS_DIR` (extra domain packs), `DOCINTEL_MODEL_REGISTRY` (a replacement
model registry), `PROMETHEUS_MULTIPROC_DIR` (metrics across API worker processes). The embedding service reads
`DOCINTEL_EMBEDDING_MODEL`, `DOCINTEL_EMBEDDER_CACHE` (model folder), `DOCINTEL_EMBEDDER_DEVICE` (`cpu` or `cuda`),
`DOCINTEL_EMBEDDER_THREADS`, `DOCINTEL_RERANKER_MODEL` and `DOCINTEL_RERANKER_PATH`.

## Files

| File | Tracked | Purpose |
|---|---|---|
| `.env.example` | yes | development settings template (copy to `.env`) |
| `deploy/.env.example` | yes | compose deployment template (copy to `deploy/.env`) |
| `deploy/api_keys.example.json` | yes | API key file template |
| `.env`, `deploy/.env`, `deploy/api_keys.json` | never | real values |
| `docintel/model_registry.yaml` | yes | embedding and reranker models with pinned checksums and calibrated floors |
