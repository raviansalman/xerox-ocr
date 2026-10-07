# xerox-ocr security model (Phase 1)

What this branch enforces, how to configure it, and what is still open. Defect IDs refer to
`docs/ASSESSMENT.md`; every control below has tests in `ultimate/tests`.

## 1. Identity and tenancy

* Every API route except `GET /` (static UI page) and `GET /health` requires an API key, sent as
  `X-API-Key: <key>` or `Authorization: Bearer <key>`. Missing or unknown key: **401**.
* **The tenant (Milvus `user_id`) comes from the key, not from the request body.**
  * A key bound to a tenant always acts for that tenant. If the body names a different `userId`,
    the request is refused with **403**.
  * A `service` key (the trusted backend that serves many end users) acts for the `userId` it
    sends. That id must match `^[A-Za-z0-9_.:@-]{1,128}$`, otherwise **400**.
* No keys configured, or an unreadable key file: **503** on every protected route (fail closed).
* `AUTH_DISABLED=true` turns all of this off for local development and logs a warning. Never set it
  anywhere reachable by other people.

| Role | Can do | Implies |
|---|---|---|
| `reader` | `/search`, `/task-status` (own tasks only) | |
| `uploader` | `/process`, `/process-file`, `/delete-document` (own tenant only) | reader |
| `service` | everything above for any tenant it names | uploader |
| `admin` | `/admin/*` (stats, stuck jobs, queues, route test, purge) | service |

### Configuring keys

```bash
cd ultimate
python -m src.security generate          # prints a new key and its key_sha256
mkdir -p secrets && $EDITOR secrets/api_keys.json   # gitignored
```

```json
[
  {"name": "storagechain-backend", "key_sha256": "<hex>", "roles": ["service"]},
  {"name": "xerox-demo-reader",    "key_sha256": "<hex>", "tenant": "xerox-demo", "roles": ["reader"]},
  {"name": "xerox-demo-ingest",    "key_sha256": "<hex>", "tenant": "xerox-demo", "roles": ["uploader"]},
  {"name": "ops",                  "key_sha256": "<hex>", "roles": ["admin"]}
]
```

Only hashes are stored. The compose files mount `${API_KEYS_HOST_FILE:-./secrets/api_keys.json}` read-only
at `/run/secrets/xocr_api_keys.json` and set `API_KEYS_FILE` to it. To rotate: edit the file and restart
the API container. Invalid entries (bad hash, unknown role, a reader/uploader key without a tenant,
duplicates) stop authentication with a logged error rather than being skipped.

The bundled UI has an "API key" field; the key is kept in that browser tab's `sessionStorage` and sent
on every call.

### Changes for existing API clients

| Before | Now |
|---|---|
| No credentials | API key required |
| `userId` defaulted to `user000` | `service` keys must send `userId`; tenant keys need not send it |
| `/delete-document` with only `file_id` deleted across tenants, and retried without bucket/path/connection filters | Deletes only inside the caller's tenant; no unscoped retry. Unknown file in that tenant: 404 |
| Admin routes open unless `VECTOR_STATS_ADMIN_KEY` was set | `admin` role required; `X-Admin-Key` / `VECTOR_STATS_ADMIN_KEY` no longer used |
| `/task-status` tried a hardcoded list of test users | Tenant keys see only their own tasks (others: 404) |
| `/process` validation errors: 500 with traceback | 4xx with a message; 500s say "see server logs" |
| CORS `*` with credentials | Off unless `CORS_ALLOW_ORIGINS` lists origins |

## 2. Network boundaries

```
 internet / office network
          │  HTTPS (terminate TLS at a reverse proxy; not part of this repo yet)
          ▼
   ┌───────────────┐   only this port is published (8000)
   │  API (uvicorn) │
   └──────┬────────┘
          │ private Docker network only
   ┌──────┼──────────────┬──────────────┬───────────────┐
   ▼      ▼              ▼              ▼               ▼
 Redis  Milvus      embedder(s)   Celery workers   (outbound: fileUrl fetches, policy in §4)
 (password)  (no auth configured)
```

| Component | Exposure (monolith `docker-compose.ultimate.yml`) | Exposure (processing/search split) |
|---|---|---|
| API :8000 | Published; put TLS in front | Published on both servers |
| Redis :6379 | **Not published**; `--requirepass` | Bound to `127.0.0.1` unless `DATASTORE_BIND_ADDR` is set to a **private** address; password required |
| Milvus :19530 / :9091 | Not published | Same as Redis. Milvus authentication is not enabled, so the private network is the only protection |
| Embedders :8080 | Not published | Not published |

Split deployment checklist (search server reads the processing server's Redis and Milvus):
1. Put both servers on a private network (same VPC/subnet). Never use public IPs.
2. On the processing server set `DATASTORE_BIND_ADDR=<its private IP>`.
3. Firewall / security group: allow TCP 6379, 19530 **only from the search server**.
4. On the search server set `MILVUS_HOST=<private IP>` and
   `REDIS_URL=redis://:<REDIS_PASSWORD>@<private IP>:6379/0`. Both are required; there are no defaults.

`ultimate/tests/test_redis_and_serialization.py::test_compose_never_publishes_data_stores_on_all_interfaces`
fails the build if any compose file publishes 6379, 19530 or 9091 on all interfaces or reintroduces the
old public IPs.

## 3. Redis

* **Password:** set `REDIS_PASSWORD` (URL-safe, e.g. `openssl rand -hex 32`). Compose interpolates it
  from the shell or from `ultimate/.env` (gitignored), **not** from the `env_file`. Compose refuses to
  start without it. All services receive `REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0`.
* **No pickle:** the metadata index synced through Redis and the disk cache is now tagged JSON
  (`dump_metadata_state` / `load_metadata_state`). Anything that is not our versioned JSON is ignored and
  never deserialized; the Redis key changed (`metadata_index_json:<tenant>`) so legacy pickle blobs are
  not even read. The embedding cache uses `.npz` with `allow_pickle=False`. No `pickle.load(s)` remains.
* **What Redis holds:** Celery broker messages and results (file ids, tenant ids, file URLs), job
  registry, heartbeats, processing locks, and (once the metadata index is re-enabled) per-tenant full
  document text for up to 30 days. Treat Redis as a sensitive data store: back it up and restrict it like
  the database.

Every read/write path (audited for this phase):

| Module | Keys | Data | Format |
|---|---|---|---|
| `ultimate_celery_app` | Celery queues `ultimate_*`, results | task kwargs (url, file_id, user_id) | JSON serializer only (`accept_content=["json"]`) |
| `ultimate_tasks` | `process_lock:{user}:{file}` | task id | string |
| `scalability_utils` | `heartbeat:{task}` | progress | JSON |
| `job_registry` | `job:{user}:{file}`, `job_index:{user}`, `task_lookup:{task}` | job status | JSON |
| `semantic_components.MetadataIndex` | `metadata_index_json:{user}`, `metadata_index_changed_users` | inverted index incl. full text | zlib + tagged JSON |
| `ultimate_tasks.rebuild_user_metadata_index_task` | `ultimate:meta_rebuild_lock:{user}` | lock | redis-py lock |

## 4. Outbound fetches (`fileUrl`)

`src/net_safety.py`, used by `/process` (HEAD requests) and the worker download:

* `http`/`https` only, no credentials in the URL.
* The host must resolve only to public addresses; loopback, private, link-local (including
  `169.254.169.254`), reserved and multicast are refused. Redirects are followed manually and every hop
  is re-checked.
* Optional allowlist `URL_FETCH_ALLOWED_HOSTS` (e.g. `.amazonaws.com`). Recommended for production.
* `MAX_DOWNLOAD_BYTES` (default 200 MB) on declared and streamed size.
* `/process` rejects a disallowed URL with 400 before anything is queued.
* Logs show URLs without query strings (presigned signatures are not logged).
* `URL_FETCH_ALLOW_PRIVATE=true` exists for local development only.

Residual risk: DNS can change between the check and the connection (rebinding). Add egress firewall
rules on the worker hosts for defence in depth.

## 5. Uploads and stored data

* `/process-file` streams uploads to `UPLOAD_DIR` and returns 413 above `MAX_UPLOAD_BYTES` (default 200 MB).
* Milvus expressions escape every user-supplied value (KD-SEC-01).
* Existing collections are never dropped automatically; a dimension or schema mismatch stops the
  process with an error (KD-MLV-01).
* UI output is HTML-escaped, including document text, filenames, ids and server messages (KD-SEC-06).

## 6. Secrets

* No secrets in the repository: `.gitignore` excludes `*.env` (except `.env.example`) and
  `ultimate/secrets/`. Configuration is environment-based (`.env.example` lists every setting).
* **Rotate now:** the original StorageChain env files contained `EXTERNAL_ACCESS_KEY`,
  `EXTERNAL_STOR_API_KEY`, `WORKFLOW_ACCESS_KEY`, `WORKFLOW_STOR_API_KEY` and a `JWT_SECRET_KEY`.
  They were never committed here, but they existed in the shared zip.
* `JWT_SECRET_KEY`, `CORS_ORIGINS`, `MAX_FILE_SIZE` and `RATE_LIMIT_*` in the template are **not read**
  by the code; they are marked as such.
* The repository `raviansalman/xerox-ocr` is currently **public**, and this documentation describes the
  original system's weaknesses. Make it private.

## 7. Not covered yet (later phases)

| Gap | Phase |
|---|---|
| TLS termination and a reverse proxy | 8 |
| Milvus authentication (`common.security.authorizationEnabled`) | 8 |
| Rate limiting, per-key quotas | 10 |
| Audit log of who searched/ingested/deleted what | 7 |
| OIDC / SSO instead of static API keys; key expiry | 7 |
| `/health` still reports Celery/Redis without checking (KD-OPS-02) | 1 (next) |
| Event-loop blocking under slow searches (KD-OPS-01) | 1 (next) |
| Shared metadata index must become per-tenant before `threading` is imported (KD-SEC-09) | 5 |
| Embedder services have no auth (internal network only) | 8 |
| Prompt-injection handling for OCR text once RAG exists | 6 |
| Encryption at rest for Milvus/Redis volumes | 10 |
