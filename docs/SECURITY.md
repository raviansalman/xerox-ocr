# Security model

## Identity and tenants

* Every API call needs an `X-API-Key`. Keys are stored only as SHA-256 hashes (`DOCINTEL_API_KEYS_FILE`, mounted as
  a secret in the compose file). Create one with `docintel keygen`; it prints the key once and the hash to store.
* A key is bound to one tenant and one or more roles. The tenant always comes from the key, never from the request
  body or query string.
* Roles: `reader` (ask, read documents) < `uploader` (+ upload, delete, reprocess) < `admin`. A `service` key is for
  a trusted backend serving several tenants; it must name the tenant in `X-Tenant-Id` on every call.
* `DOCINTEL_AUTH_MODE=dev` turns authentication off for local work and is refused when
  `DOCINTEL_ENVIRONMENT=production`. If the key configuration cannot be read, every call fails with 503 rather
  than falling back to open access.

## Tenant isolation, in depth

1. **Database.** Every tenant table has a forced row-level security policy. The application sets the tenant for
   each transaction; a connection with no tenant sees no rows, and a row cannot be written under another tenant.
   The application role must not be a superuser and must not have `BYPASSRLS`: the engine checks this at start
   and refuses to run in production otherwise. `deploy/postgres-init.sql` creates the role correctly.
2. **Vector index.** Milvus searches are filtered by tenant (the partition key) and the candidate documents are
   then re-read through PostgreSQL, so a vector hit cannot surface a document the tenant cannot see.
3. **Lexical index and vocabulary.** The postings and the vocabulary used for typo correction are tenant tables
   under the same forced row-level security, so a correction can never suggest another tenant's word. Every
   retriever receives only a tenant-scoped connection.
4. **Object store.** Originals are stored under the tenant's prefix and served only after the document row has
   been read with the caller's tenant.
5. **Tests.** The evaluation harness asks every labelled question as every other tenant (154 queries, leakage must
   be 0). `tests/integration/test_security_adversarial.py` (55 tests) tries injection through questions and list
   filters, tenant headers on tenant-bound keys, malformed tenants on service keys, path manipulation on every
   document route, other tenants' document ids on every route, cross-tenant probes of postings, vocabulary typo
   correction, entities, structured counts and vectors, and settings and cache isolation. Migration tests check
   that every table with a `tenant_id` column forces row-level security.

## Input handling

* File types are detected from content. Unsupported, empty and oversized files (`DOCINTEL_MAX_UPLOAD_BYTES`,
  default 200 MB) are rejected; page and sheet-row limits bound the work per file.
* Request bodies are limited while they stream (uploads `DOCINTEL_MAX_REQUEST_BYTES`, everything else 1 MB) and
  refused with 413 before anything is spooled to disk. OCR has a per-page time limit, images a per-frame pixel
  limit checked before decoding.
* Office conversion runs LibreOffice headless with a timeout in a private temporary profile and its own process
  group, so a timeout stops every process it started.
* URL ingestion allows only http and https, refuses credentials in URLs and any host that resolves to a private,
  loopback, link-local or multicast address (including private IPv4 addresses inside IPv4-mapped and NAT64 IPv6
  addresses; 6to4 is refused), re-checks every redirect, ignores proxy settings from the environment, caps the
  download size while streaming and bounds the whole download in time. An optional
  host allow-list (`DOCINTEL_URL_ALLOWED_HOSTS`) narrows it further. It is off by default in the compose file.
  Residual risk: the address is checked before connecting, so a hostile DNS server could still rebind between the
  check and the connection. Keep it off, or set an allow-list, where that matters.
* SQL is always parameterized. The few interpolated fragments are code constants or values checked against a
  closed set at the call site (reviewed with bandit, see `pyproject.toml`). Milvus filter values are validated.
* Query plans from any source other than the deterministic planner (for example a language model) are parsed with
  a strict schema that has no tenant, document id, SQL or code fields, and every value is checked against the
  tenant's packs (`docintel/query/validation.py`).
* Stored file names lose path components, control characters and quotes. Originals are always served as
  attachments with `Content-Security-Policy: default-src 'none'; sandbox`, so uploaded HTML never runs on the
  engine's origin. Query bodies reject unknown fields.
* Client request ids are echoed and logged only when they match `[A-Za-z0-9._:-]{1,64}`.

## Documents as untrusted input to language models

The optional Claude answer provider sees only the question and the evidence, with markup escaped and each passage
in a delimited element; it has no tools and no access to data or identifiers beyond the evidence numbering.
Sentences that read like instructions to an AI system are withheld from the model. Its reply must match a JSON
schema, and every sentence is verified against the evidence it cites before it is returned. Computed answers and
authorization never involve a model. Tests use a fake Messages server and a document containing a prompt injection.

## Web UI and HTTP

* Strict Content Security Policy (`script-src 'self'`, no inline script), `X-Frame-Options: DENY`,
  `nosniff`, `Referrer-Policy: no-referrer`. The UI inserts every value as text, never as HTML.
* The API key is kept in the browser's local storage for convenience. Use reader keys for people who only ask
  questions, and serve the UI over HTTPS (terminate TLS at a reverse proxy).
* CORS is closed unless `DOCINTEL_CORS_ORIGINS` lists origins.

## Operations

* Only the API port is published by the compose file; PostgreSQL, Milvus, Redis and the embedder are on the
  internal network. Redis requires a password.
* Logs are JSON with a request id; document text and API keys are never logged. URLs are logged with credentials
  and query strings removed.
* `/metrics` requires a bearer token in production; metric labels carry no tenant, user or document data. Worker
  metrics are served on the internal network only.
* The embedding and reranker models are pinned by SHA-256 (and revision when downloaded) and loaded offline; the
  service refuses weights that do not match.
* Images run as a non-root user; a proxy CA needed at build time is passed as a build secret and not kept.
* Production refuses the stand-in test models (`test-*`) for embeddings and reranking.
* The API, worker and embedder containers run with `no-new-privileges` and no Linux capabilities.
* CI runs with a read-only token and actions pinned to commit SHAs.
* `scripts/check_hardcoded.py` scans every tracked file except tests and documentation (application, packs, UI,
  Dockerfiles, compose, templates, SQL, scripts, CI) for fixture and customer names, addresses, URLs, credentials,
  key-shaped strings, passwords in URLs and absolute paths. Exceptions name the rule they waive and the reason.
* Dependencies are checked with `pip-audit` (clean at release: Pillow 12.3, Starlette 1.3.1+) and code with
  `bandit` (no high findings).
* Never commit `deploy/.env` or `deploy/api_keys.json`; only the `.example` templates are tracked.

## Known residual risks

* **Git history.** Commits from the earlier service (before the v2 rewrite) contain infrastructure details that were
  later deleted from the tree: public IP addresses and SSH targets of the hosts it ran on. Removing them needs a
  history rewrite and a force-push, which is the repository owner's decision; in any case treat those hosts as
  known and rotate their keys and addresses. Keep the repository private until then.
* **DNS rebinding** for URL ingestion (above).
* **API keys in browser storage** for the web UI (above).
* **Base images** are pinned by tag, not by digest, and Python dependencies by range, not by lock file: a rebuild
  can pick up newer versions. Pin digests and use a lock file where builds must be reproducible bit for bit.
* **One database role** owns the schema and runs the application (row-level security is forced, so the owner is
  still filtered). A separate migration role would narrow what a compromised API process could change.
