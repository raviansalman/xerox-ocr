# Deployment and observability architecture

Status: design for review.

## 1. Today

Five compose files and several shell scripts. Production and staging receive the developer's working tree by
`rsync --delete`, `docker cp` into running containers, and source bind mounts (`./src:/app/src` in every service
of `docker-compose.ultimate.yml`); production configuration (`docker_byoc*.env`) lives outside git (`FORENSICS.md`
F3, `HARDCODED_DATA_AUDIT.md`). Phase 1 added Redis authentication, private datastore binding, API key mounting and
liveness/readiness checks (`SECURITY.md`).

Consequence: what runs in production cannot be reproduced from the repository, which is exactly why production is
UNKNOWN today.

## 2. Target principles

* **Immutable, versioned images** built in CI from a tagged commit; the image is the release. No source bind
  mounts, no `docker cp`, no rsync of working trees.
* **Configuration from the environment**, validated at startup by typed settings; secrets from mounted files or a
  secret manager; a non-secret startup report lists effective configuration and versions.
* **Same artefacts in every environment**; differences only in configuration.
* **Air-gap ready:** all images and model artefacts can be loaded from an internal registry; no build-time or
  run-time internet dependency (pinned wheels, mirrored model weights with checksums).
* **Customer-controlled hosting** (on-prem, private cloud, sovereign cloud) as the default target.

## 3. Components and topology

| Component | Image | Scales by | State |
|---|---|---|---|
| `api` | `die-api` | requests (stateless, N replicas behind a reverse proxy) | none |
| `worker-parse` / `worker-ocr` / `worker-understand` / `worker-index` | `die-worker` (one image, role by command) | queue depth per stage | none |
| `embedder` | `die-embedder` (model baked or mounted read-only from the model store) | embedding throughput; GPU optional | none |
| `reranker` (optional) | `die-reranker` | query rate | none |
| `llm` (optional, on-prem) | customer-approved runtime | query rate | none |
| `postgres` | upstream image | vertical, then read replicas | registry, fields, lexical index, audit |
| `milvus` | upstream image (standalone → cluster) | vectors | vectors |
| `redis` | upstream image | small | queue and caches |
| `object-store` | MinIO or customer S3-compatible / filesystem | capacity | originals, page images |
| `reverse-proxy` | customer standard (nginx/Traefik/ingress) | | TLS termination |

Profiles:

* **Single node (PoC, on-prem):** one compose file, all components, volumes on local disks, GPU optional.
* **Production:** Kubernetes (Helm chart) or compose across hosts; Postgres and Milvus per customer HA standards;
  workers autoscaled by queue depth.

## 4. Configuration

* `die/config`: typed settings (pydantic-settings) grouped by module; every variable documented with default,
  allowed values and whether it is secret. One source per value (fixes the duplicate and conflicting defaults in
  `HARDCODED_DATA_AUDIT.md` D1 to D6).
* **Model registry** file (versioned): for each model, name, revision, sha256, task, dimension, normalization,
  max tokens, licence. Startup verifies that the embedding model matches the vector collection's recorded model
  and dimension, and refuses to serve otherwise.
* **Tenant configuration** (taxonomy, field schemas, identifier patterns, synonyms, thresholds, retention, language
  set) in Postgres, versioned and audited; changes trigger the relevant reprocessing.

## 5. Release and migration

* CI: lint, unit tests, integration tests (Milvus, Postgres, Redis), golden evaluation (stand-in embedder),
  security tests, image build, image scan, SBOM; tag → signed images.
* Database migrations (Alembic) are forward-only, run as a separate job before rolling out new API/worker
  versions; each migration is backward compatible with the previous release (expand/contract).
* Milvus collection changes create new collections with an alias switch after backfill; never destructive in
  place (consistent with the KD-MLV-01 fix).
* Rollback = previous image tag + compatible schema.

## 6. Observability

| Signal | Content |
|---|---|
| Logs | JSON; `ts`, `level`, `service`, `version`, `request_id`, `query_id`, `tenant_id` (hashed in shared sinks if required), `document_id`, `version_id`, `run_id`, `job_id`, `stage`, `duration_ms`, `model`, `error_class`. No document text or secrets at info level |
| Metrics (Prometheus) | Request rate/latency/error per endpoint and answer kind; per-retriever latency and candidate counts; fusion/rerank latency; planner source (rules/LLM) and fallback rate; abstention rate; ingest throughput; per-stage processing time (parse, OCR, layout, understanding, embedding, indexing); OCR confidence distribution; queue depth and age; worker health; dependency health; security events (final-check drops) |
| Traces (OpenTelemetry) | One trace per query (plan → retrievers → fusion → rerank → answer) and per ingest run (stage spans), propagated through Celery |
| Health | `/health/live`, `/health/ready` (existing pattern) extended with Postgres, object store, reranker, LLM; non-critical components report degraded |
| Diagnostics | `explain` for a query id: plan, retriever candidates (ids and scores), fusion tiers, final ranking, evidence; operator role only |

Every query and every document is diagnosable end to end from its id.

## 7. Backup and recovery

Postgres (PITR), object store (versioned buckets or snapshots), Milvus (backups or rebuild from canonical chunks:
vectors are derived data and can be regenerated from Postgres + object storage). Recovery drills are part of the
production checklist; restoring must preserve tenant boundaries and audit trails.

## 8. Sizing notes (measured so far)

* Real all-mpnet-base-v2 on 4 CPU cores embedded about 2 chunks per second (`SEARCH_FORENSICS.md`); ingest
  capacity needs GPU or more cores for large backfills.
* Current vector search p50 about 55 to 70 ms at 2k to 30k chunks per tenant locally (scale probe); the legacy
  full scan reached 17 s and truncated at 16,384 rows, so full scans are excluded from the query path by design.
