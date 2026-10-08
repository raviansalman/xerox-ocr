# Document Intelligence Engine

Ingest documents in the common office and scan formats, keep every document whole and addressable, and ask
questions about them in plain English. Every result carries its evidence (document, page, character span and the
retriever that found it); every computed answer carries its calculation and the records it was computed from; and
when the evidence is not good enough the engine says so instead of guessing.

* **Formats:** PDF (native, scanned and mixed), images (PNG, JPEG, TIFF including multi-page, GIF, BMP), Word
  (DOCX, DOC, ODT), spreadsheets (XLSX, XLS, ODS, CSV), presentations (PPTX, PPT, ODP), e-mail (EML, with
  attachments), HTML, RTF, plain text. See [docs/INGESTION.md](docs/INGESTION.md).
* **Canonical model:** pages, typed blocks (headings, paragraphs, tables, key-value lines) with bounding boxes and
  OCR confidence, tables with cells, retrieval units, fields, entities, clauses and relations, each with its page
  and exact character span; a version per processing run with the parser, OCR engine, embedding model and domain
  packs that produced it. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
* **Retrieval:** eight independent retrievers (exact, lexical, fuzzy and OCR-tolerant, semantic, entity,
  contextual, metadata, structured) fused deterministically, with exact evidence always ranked first. **Exact search
  does not need the vector index**: CI runs the whole suite with semantic search switched off.
* **Questions:** identifiers in any spelling, quoted phrases, names with typos, OCR errors, meaning, concepts worded
  differently ("cancel" finds termination clauses), filters (type, governing law, dates, amounts, parties,
  signatures, clauses), counts, percentages, sums and averages per currency, group-bys, period comparisons, field
  lookups. See [docs/QUERIES.md](docs/QUERIES.md).
* **Domain packs:** the core engine knows no business vocabulary. Document types, fields, clause types, concepts
  and relation patterns come from packs (`business`, `legal`, `finance`, or your own), chosen per tenant. See
  [docs/DOMAIN_PACKS.md](docs/DOMAIN_PACKS.md).
* **Grounded answers (optional):** an extractive answer quoting the evidence, or a Claude-written answer whose every
  sentence is verified against the evidence it cites; documents are treated as untrusted data. The engine works
  fully without a language model.
* **Multi-tenant:** the tenant always comes from the API key; PostgreSQL row-level security is forced on every
  tenant table and the vector index is partitioned by tenant. Adversarial tests probe every index for leakage. See
  [docs/SECURITY.md](docs/SECURITY.md).

Built with FastAPI, PostgreSQL 16, Milvus 2.6 (HNSW), Tesseract, LibreOffice and sentence-transformers
(`all-mpnet-base-v2`). Nothing leaves your infrastructure unless you enable the optional Claude answer provider.

## Quick start (Docker)

```bash
cp deploy/.env.example deploy/.env       # set POSTGRES_PASSWORD, DOCINTEL_DB_PASSWORD, DOCINTEL_REDIS_PASSWORD, DOCINTEL_METRICS_TOKEN
docintel keygen                          # or: python -m docintel.security generate; prints a key and its sha256
cp deploy/api_keys.example.json deploy/api_keys.json   # put the sha256, a tenant name and roles in it
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
```

Open http://localhost:8000, enter the API key, drop files on the Documents tab and ask questions on the Ask tab.
Building, sizing, scaling, backups and upgrades are in [docs/OPERATIONS.md](docs/OPERATIONS.md); every setting is
in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## API

All endpoints take `X-API-Key`. Interactive documentation is served at `/api/docs`. Details in
[docs/API.md](docs/API.md).

| Method and path | Role | Purpose |
|---|---|---|
| `POST /api/v1/documents` (multipart `files`) | uploader | Upload files; duplicates are detected by content |
| `POST /api/v1/documents/url` | uploader | Ingest from a URL (off by default) |
| `GET /api/v1/documents` | reader | List and filter documents |
| `GET /api/v1/documents/{id}` | reader | Status, staleness, type, fields, entities, clauses, relations, tables, versions |
| `GET /api/v1/documents/{id}/pages/{n}`, `/pages/{n}/image`, `/original` | reader | Page text, blocks and tables; rendered page; the original file |
| `DELETE /api/v1/documents/{id}` | uploader | Delete a document from every store and index |
| `POST /api/v1/documents/{id}/reprocess` | uploader | Process again |
| `POST /api/v1/query` `{"q", "limit", "explain", "answer"}` | reader | Ask a question |
| `GET`, `PUT /api/v1/settings` | reader, admin | The tenant's domain packs and example questions |
| `GET /api/v1/stats`, `/api/v1/taxonomy`, `/api/v1/me` | reader | Counts, document types, caller identity |
| `GET /health/live`, `/health/ready`, `/metrics` | none, none, bearer token | Liveness, dependency readiness, Prometheus metrics |

```bash
curl -s -H "X-API-Key: $KEY" -F files=@contract.pdf -F files=@invoice.docx http://localhost:8000/api/v1/documents
curl -s -H "X-API-Key: $KEY" -H 'content-type: application/json' \
     -d '{"q": "How many contracts expire in 2027?", "explain": true}' http://localhost:8000/api/v1/query
```

## Quality and performance

Measured, not claimed; how to reproduce each number is in [docs/EVALUATION.md](docs/EVALUATION.md) and
[docs/OPERATIONS.md](docs/OPERATIONS.md#performance). The validation of this release, what is and is not verified,
and the remaining limitations are in [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

* **Synthetic evaluation set** (80 labelled questions over 83 documents in 3 tenants): v2 passes 80 of 80 with the
  real embedding model and 73 of 73 with semantic search switched off (v1: 77 and 69); MRR 1.0, exact recall 100%,
  computed answers 100% correct, no false positives on unanswerable questions, no cross-tenant results in 160
  isolation queries. These numbers catch regressions; they do not measure quality on your documents, which still
  has to be established with real labelled documents ([docs/EVALUATION.md](docs/EVALUATION.md)).
* **50,000 documents on one 4-core machine** running everything: ingestion at 650 documents per minute with real
  embeddings (2% scanned), 0 failures. One user: identifiers 84 ms, quoted phrases 172 ms, keywords 197 ms, typos
  90 ms, counts 121 ms, filters 91 ms, totals over 16,667 invoices 359 ms, lookups 335 ms, concept questions
  498 ms (p50, semantic on). Under concurrent load the machine saturates at about 15 to 21 questions per second
  (CPU, mostly PostgreSQL); from 8 users p95 is above one second. Details and the bottleneck analysis in
  [docs/OPERATIONS.md](docs/OPERATIONS.md#performance).
* **Clean-machine deployment** is checked in CI: both images built from the repository, the compose stack started,
  and 23 end-to-end steps (every upload format, every kind of question, evidence, cross-tenant access, dependency
  restarts, reprocessing, deletion) pass.

## Development

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,answer]"
sudo apt-get install tesseract-ocr libreoffice-writer-nogui        # OCR and legacy office formats
pytest -q                                                          # unit tests, no services needed
ruff check docintel scripts tests && python scripts/check_hardcoded.py
```

Integration, browser, migration and Celery tests need services; see [docs/OPERATIONS.md](docs/OPERATIONS.md#testing).
Run the suite three ways: with a deterministic stand-in embedder (default), with semantic search disabled
(`DOCINTEL_TEST_VECTORS=disabled`) and with the real model.

## Repository layout

```
docintel/
  api/            HTTP API and web UI
  ingest/         registration, processing pipeline, versions, job dispatch and recovery
  processing/     type detection, file safety limits, parsers, OCR, signatures, chunking into retrieval units
  understanding/  fields, entities, clauses, relations, classification, language
  packs/          domain packs (core, business, legal, finance) as YAML
  indexing/       embedding client, vector store
  retrieval/      retriever contract, postings index, the eight retrievers, fusion, evidence, reranking
  search/         planner and the v1 engine (kept for comparison and rollback)
  query/          v2 engine, plan validation, deterministic computations
  answering/      optional grounded answers (extractive or Claude) with verification
  evaluation/     evaluation harness: datasets, metrics, gates, comparison
  storage/        PostgreSQL (migrations with row-level security), repositories, object store
  embedder/       the embedding and reranking service
deploy/           Dockerfiles, compose file, database init, configuration templates
scripts/          model placement, load test, hardcoded-data scanner
tests/            unit, integration, adversarial, browser tests; synthetic corpus and golden dataset
docs/             architecture, ingestion, queries, API, configuration, security, operations, evaluation, packs
  design/         the v2 design documents the implementation follows
  history/        analysis of the earlier service
```
