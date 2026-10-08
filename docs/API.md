# API

Base path `/api/v1`. Every call needs `X-API-Key: <key>` (or `Authorization: Bearer <key>`); the key decides the
tenant and the roles. A `service` key must also send `X-Tenant-Id`. Interactive documentation: `/api/docs`.
Errors are JSON `{"detail": ...}` with 400 (bad input), 401 (no or unknown key), 403 (role or tenant mismatch),
404 (not found, including another tenant's ids), 422 (invalid body or unknown field), 503 (authentication not
configured). Every response carries `X-Request-ID`.

## Documents

`POST /documents` (uploader), multipart `files` (one or many) and optional `collection`:

```json
{"documents": [{"id": "…", "filename": "contract.pdf", "status": "queued", "duplicate": false, "…": "…"},
               {"filename": "archive.zip", "status": "rejected", "error": "unsupported archive or corrupted Office file"}]}
```

Identical content in the same tenant returns the existing document with `"duplicate": true`.

`POST /documents/url` (uploader, when `DOCINTEL_URL_FETCH_ENABLED`) `{"url", "filename"?, "collection"?}`.

`GET /documents?status=&doc_type=&q=&collection=&limit=&offset=` (reader): `{"documents", "total", "limit", "offset"}`.

`GET /documents/{id}` (reader): status, error, kind, page count, title, type with confidence and method, language,
signature flag, `stale` (reasons the document should be reprocessed), `fields`, `entities`, `clauses`,
`relations`, `tables`, `versions` (each with status and the components that produced it).

`GET /documents/{id}/pages/{n}`: page text, OCR confidence, word boxes, `blocks` (type, text, character offsets,
bounding box) and `tables` (rows of cells). `GET /documents/{id}/pages/{n}/image`: PNG rendering.
`GET /documents/{id}/original`: the uploaded file, always as a sandboxed attachment.

`DELETE /documents/{id}` (uploader): removes the original, every derived row, the postings and the vectors.
`POST /documents/{id}/reprocess` (uploader): queues the document again; a new version replaces the current one.

## Query

`POST /query` (reader):

```json
{"q": "How many contracts expire in 2027?", "limit": 20, "explain": false, "answer": false}
```

Response:

| Field | Content |
|---|---|
| `answer` | `kind` (`number`, `table`, `fields`, `figure`, `documents`, `none`) and `text`; for computations `calculation` and `records`; `note`, `degraded` when a dependency failed |
| `results` | per document: rank, id, file name, title, type, `tier` (1 exact to 4 meaning only), `confidence`, `retrievers`, `match_types`, `explanation`, `evidence` (page, `char_start`, `char_end`, `text`, match type, retriever), key `fields`, `version` |
| `total`, `intent`, `engine`, `terms`, `timings_ms` | |
| `plan` (with `explain`) | the validated plan: intent, text, phrases, identifiers, filters, concepts, entities, strategies, packs |
| `generated_answer` (with `answer`) | `provider`, `status` (`answered`, `computed`, `abstained`, `disabled`), `sentences` (text, citation ids, spans for quotes), `citations`, `dropped`, `warnings`, `degraded` |

See [QUERIES.md](QUERIES.md) for the questions it understands.

## Tenant settings

`GET /settings` (reader): `packs` (the tenant's choice, or null for the deployment default), `effective_packs`
(including `core` and dependencies), `available_packs` and `examples` (`[{"label", "q"}]`).
`PUT /settings` (admin): `{"packs": ["legal"], "examples": [{"label": "Expiring", "q": "Which agreements expire next year?"}]}`
(at most 30 examples; unknown packs are refused; the change is audited).

## Other

`GET /stats` (counts by status and type), `GET /taxonomy` (document types of the tenant's packs), `GET /me`
(principal, tenant, roles). Unauthenticated: `GET /health/live`, `GET /health/ready`. `GET /metrics`: Prometheus,
bearer `DOCINTEL_METRICS_TOKEN`.

## Compatibility with v1

Every v1 endpoint and field is kept. Additions: `answer` on queries and `generated_answer`; `tier`, `retrievers`,
`evidence`, `explanation` and `version` on results; `calculation` and `records` on computed answers; `stale`,
`relations`, `tables` and `versions` on documents; `blocks` and `tables` on pages; the settings endpoints. One
deliberate change: unknown fields in a query body are now rejected with 422 instead of ignored.
