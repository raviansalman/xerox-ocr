# Architecture

One Python package (`docintel`) deployed as a modular monolith: the API, ingestion workers and an embedding service,
around PostgreSQL (system of record), Milvus (vectors only), an object store (originals) and Redis (job queue in
multi-node mode). The design that this implements, and the reasoning behind it, is in [design/](design/README.md);
where the code differs from the design, this document describes the code.

```
           ┌────────────┐                    ┌───────────────────────────────────────────────────────┐
  browser ─┤  Web UI    ├─── upload / ask ──►│ API: auth → tenant → planner → validation → engine    │
  clients ─┤  REST API  │◄── answer+evidence─│      retrievers ─► fusion ─► evidence ─► (answers)    │
           └────────────┘                    └─────┬──────────────────┬─────────────────┬────────────┘
                                                   │ jobs             │ SQL (forced RLS) │ vectors (tenant partition)
                                            ┌──────▼──────┐   ┌───────▼──────┐   ┌──────▼─────┐
                                            │ Workers     │──►│ PostgreSQL   │   │ Milvus     │
                                            │ Celery or   │   │ canonical    │   │ unit       │
                                            │ threads     │   │ model,       │   │ embeddings │
                                            └──────┬──────┘   │ postings,    │   └────────────┘
                                                   │          │ fields ...   │
                                            ┌──────▼──────┐   └──────────────┘   ┌────────────┐
                                            │ Embedder    │                      │ Object     │
                                            │ (+reranker) │                      │ store      │
                                            └─────────────┘                      └────────────┘
```

## Modules

| Module | Responsibility |
|---|---|
| `api` | HTTP API and web UI; request ids, security headers, metrics endpoint |
| `security` | API keys (SHA-256), roles, tenant from identity |
| `ingest` | Registration (content hash, dedupe, object store, job record), processing pipeline, versions, staleness, dispatch and recovery |
| `processing` | Type detection from content, container and image limits, parsers per format into the canonical model, OCR abstraction (Tesseract), signature marks, chunking into retrieval units |
| `understanding` | Dates by role, amounts, identifiers, people, organizations, jurisdictions, clauses, relations, classification, language; all driven by the tenant's domain packs |
| `packs` | Domain packs (YAML): document types, fields, clause types, concepts, relation patterns, lookup terms |
| `indexing` | Embedding client (model and dimension verified), Milvus store (in-memory store for tests) |
| `retrieval` | The retriever contract, the postings index, eight retrievers, fusion, evidence verification, optional reranking |
| `search` | The query planner and plan types; the v1 engine, kept behind the engine flag |
| `query` | The v2 engine, plan validation (including plans from untrusted sources) and deterministic computations |
| `answering` | Optional grounded answers: extractive, or Claude with verification |
| `evaluation` | Datasets, metrics, gates, report comparison |
| `storage` | Connection pool with tenant transactions, migrations (up and down), repositories, object store |
| `embedder` | The embedding and reranking HTTP service (offline, checksum-verified models) |

## The six representations

| Representation | Where | What |
|---|---|---|
| Original | object store, under the tenant and the SHA-256 | the uploaded bytes, never modified |
| Canonical model | PostgreSQL: `documents`, `document_versions`, `pages`, `blocks`, `doc_tables`, `table_cells` | text per page as the concatenation of typed blocks (so every block, unit and annotation is an exact span of the page text), bounding boxes, OCR confidence, table cells with headers |
| Lexical index | PostgreSQL: `unit_terms` postings and `vocabulary` | per retrieval unit: exact words, English stems, OCR-folded forms, canonical identifiers, reversed digit tokens (suffix search) and file-name tokens; B-tree indexed so lookups stay index scans under row-level security |
| Vector index | Milvus | one vector per passage (and per document with three or more passages), tenant as partition key; can be rebuilt from PostgreSQL |
| Structured store | PostgreSQL: `fields`, `entities`, `clauses`, `relations` | typed values with role, unit and confidence; normalized entity names; clause types; subject-predicate-object relations with qualifiers |
| Provenance | every row above | document, version, page, block and character span; the version records pipeline, parser, OCR engine, embedding model and packs |

Retrieval units (`chunks`) are of three kinds: `passage` (sentence-bounded spans of a page with heading context),
`table_row` (a row with its header labels) and `document` (title, type, file name and key fields).

Every tenant table forces row-level security on `current_setting('docintel.tenant')`. The engine sets it per
transaction from the caller's key; a connection without it sees nothing. `ingest_jobs` (identifiers only) is the
one cross-tenant table, read only by the job system.

## Ingestion

1. **Register** (in the API request): stream to the object store while hashing, detect the type from the bytes,
   reject empty, unsupported and oversized files, dedupe by content per tenant, insert the document and a job.
2. **Process** (worker): start a version; parse into pages and blocks (native text where it exists, OCR per page
   where it does not, so mixed PDFs OCR only their scanned pages); understand with the tenant's packs; build units;
   embed passages; replace all derived rows and postings in one transaction (serialized per document); upsert
   vectors; mark the version current and the document `indexed`.
3. **Failure** marks the version and the document `failed` with a reason; a dependency outage is retried. A document
   is never `indexed` with partial content.
4. **Staleness**: a document whose version was produced by another pipeline, pack set or embedding model is
   reported `stale` with the reasons; `docintel reprocess --tenant T --stale` rebuilds those.

Safety limits (container entries and ratios, image pixels, pages, sheet rows, attachment depth, conversion
timeouts) are in [INGESTION.md](INGESTION.md).

## Query answering

```
question ─► planner ─► QueryPlan ─► validation against the tenant's packs ─┐
                                                                           │
   search / fact ─► retrievers in plan.strategies (semantic in parallel) ─► fusion ─► evidence ─► results
   count / sum / average / min / max / percentage / group / compare ─► SQL over structured rows ─► answer + calculation + records
   lookup ─► search, then the field's value and span from the best evidence-backed documents
                                                                           │
                                                optional: rerank, grounded answer (verified)
```

**Planner.** Deterministic (no model). It recognizes intents, document types and concepts from the enabled packs,
jurisdictions, parties, signers, date expressions and roles, amounts with currencies and operators, clause types,
group-by keys, period comparisons and quoted phrases, and chooses the retrieval strategies. A plan from any other
source (for example a language model) goes through `plan_from_untrusted`: a strict schema with no tenant, document
or SQL fields, and vocabulary checked against the packs.

**Retrievers** all implement `Retriever.retrieve(context, plan, scope, budget) -> [Candidate]`; the context holds
a tenant-scoped connection, so no retriever can read another tenant's rows.

| Retriever | Finds | Tier |
|---|---|---|
| exact | quoted phrases, identifiers in any spelling, exact words, file names | 1-2 |
| lexical | all terms (BM25 over postings), most terms, joined or split forms | 2-3 |
| fuzzy | OCR-tolerant forms, partial identifiers, typo corrections from the tenant's vocabulary | 2-3 |
| semantic | units above the model's similarity floor and within a margin of the best | 4 |
| entity | people, organizations and places named in the question | 2 |
| contextual | concepts from the packs: clauses of that type, relations with that predicate, every way the concept is written | 2 |
| metadata | the document unit (title, type, file name, key fields) | 2 |
| structured | documents satisfying the plan's filters, with the matching fields as evidence | 1 |

**Fusion** groups candidates by document. Tier 1 evidence always comes first; for natural-language questions strong
evidence and strong semantic similarity then compete on reciprocal-rank fusion, followed by tolerant and weak
matches. Rules remove documents: identifier-only queries need lexical evidence, quoted phrases must appear, typo
corrections apply only when nothing matched otherwise, and when the question names things that occur in none of the
tenant's documents, documents matched only by concept or weak similarity are dropped.

**Evidence** is re-read from the stored text and checked against what the retriever claimed (an identifier match
must contain the identifier); each item has page, character span, match type and retriever. With no reliable
evidence the answer is an abstention.

**Computations** never estimate: counts, sums, averages, extremes, percentages, group-bys and period comparisons
are SQL over extracted fields, per currency (never added across currencies), and return the calculation (operation,
field, filters, periods, documents considered and with values) and the supporting records with spans.

**Engine flag.** `DOCINTEL_RETRIEVAL_ENGINE=v2` (default), `v1` (the previous engine, unchanged) or `shadow` (serve
v1, run v2 alongside and log the differences).

## Optional layers

* **Reranker** (`DOCINTEL_RERANKER_MODEL`): a cross-encoder reorders the head of the fused list within fusion groups
  outside the exact tier. Off by default: on the fixtures it changes no metric (see [EVALUATION.md](EVALUATION.md)).
* **Grounded answers** (`"answer": true`): extractive by default; with `DOCINTEL_ANSWER_PROVIDER=anthropic`, a
  Claude model writes the answer from escaped, delimited evidence with no tools, and every sentence is verified
  (citations exist, numbers and quotes appear in the cited evidence, most words are supported). Instruction-like
  sentences in documents are withheld from the model. Computed answers are never rewritten by a model.

## Scaling

* Workers are stateless; in Celery mode add worker containers. A job is acknowledged after it finishes, so a lost
  worker's job is redelivered; thread mode recovers unfinished jobs from `ingest_jobs` at start.
* OCR runs pages in parallel per document; the embedder micro-batches concurrent requests and can run on a GPU.
* The lexical index is a B-tree postings table (rarest term first, then intersection), so exact search cost grows
  with the number of matching units, not the corpus.
* The API never processes documents in Celery mode; query latency is independent of ingestion load except for
  shared hardware.
