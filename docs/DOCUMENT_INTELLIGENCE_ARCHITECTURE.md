# Document Intelligence Engine: target architecture

Status: **design for review. Nothing here is implemented** unless `CAPABILITY_MATRIX.md` says WORKING.

`xerox-ocr` becomes a generic, reusable Document Intelligence Engine (DIE). Xerox is one deployment of it. The
engine turns arbitrary enterprise documents into a structured, searchable, explainable knowledge base, and answers
natural-language questions over the documents a user is authorized to see, with verifiable evidence.

Document set:

| Document | Covers |
|---|---|
| This document | Principles, layers, modules, technology roles, data flows, decisions |
| `CAPABILITY_MATRIX.md` | Current vs target, per capability, with evidence |
| `DOCUMENT_MODEL.md` | Canonical document model and storage schema |
| `INGESTION_ARCHITECTURE.md` | Intake, preservation, OCR, parsing, understanding, indexing |
| `SEARCH_ARCHITECTURE.md` | Retrieval: exact, lexical, semantic, structured, fusion, ranking |
| `QUERY_PLANNER.md` | Natural language to typed plans, execution, answers, abstention |
| `SECURITY_ARCHITECTURE.md` | Tenancy, authorization, data-layer enforcement, AI-specific risks |
| `DEPLOYMENT_ARCHITECTURE.md` | Images, configuration, on-prem topology, observability |
| `EVALUATION_STRATEGY.md` | Datasets, metrics, gates |
| `MIGRATION_PLAN.md` | S0 to S15 with entry/exit criteria |
| `HARDCODED_DATA_AUDIT.md` | What must leave production code and where it goes |
| `SEARCH_QUERY_EXAMPLES.md` | 39 specified questions with plans and expected answers |
| `SEARCH_FORENSICS.md`, `ASSESSMENT.md`, `FORENSICS.md`, `SECURITY.md` | Evidence about the current system |

## 1. Principles

1. **The user asks; the engine decides how to answer.** No user-selected "exact / semantic / hybrid".
2. **Understand once, at ingest; answer many times.** Structure, fields, entities, classes and regions are
   extracted when a document arrives and stored with provenance, so questions run against indexes and tables, not
   against raw text in an LLM context.
3. **No single index answers every question.** Lexical, vector, structured and visual representations of the same
   canonical document, each with a clear job.
4. **Deterministic components compute; models interpret.** Counts, sums, filters, dates and authorization are code
   and SQL. Language models may parse questions, classify, extract and phrase answers, and every model output is
   validated and labelled with confidence.
5. **Tenant authorization is part of every data access**, not a post-filter, with a final defence-in-depth check.
6. **Evidence for everything.** Every result and every number traces to document, version, page, region or span,
   extraction method, model version and confidence.
7. **Precision, recall, confidence and abstention together.** The engine says "insufficient evidence" or
   "7 high-confidence, 4 need review" rather than guessing. It does not promise perfect results.
8. **Generic by construction.** No customer, corpus, person, place or document literal in production code;
   domain knowledge is configuration or data (`HARDCODED_DATA_AUDIT.md`).
9. **Model and provider agnostic.** OCR, embeddings, rerankers, layout and LLMs sit behind interfaces with
   versioned outputs; on-prem/private deployment is the default assumption.
10. **Recover behaviour, not old implementations.** Legacy code is evidence of intent only.

## 2. Layers

```
                          DOCUMENT INTELLIGENCE ENGINE
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │ API (REST)  ·  AuthN/AuthZ  ·  Request context  ·  Observability            │
   ├───────────────┬──────────────────────────────┬───────────────────────────────┤
   │ INGESTION     │ UNDERSTANDING                │ KNOWLEDGE                     │
   │ intake        │ format detection, parsing    │ canonical document model      │
   │ preservation  │ OCR (pluggable, versioned)   │ document registry             │
   │ dedupe/hash   │ language detection           │ structured store (fields,     │
   │ job control   │ layout: pages, blocks,       │   entities, clauses, regions) │
   │               │   tables, reading order      │ lexical index                 │
   │               │ classification               │ vector index                  │
   │               │ entity / field / clause      │ object storage (originals,    │
   │               │   extraction                 │   page images)                │
   │               │ visual regions (signature,   │                               │
   │               │   stamp, handwriting)        │                               │
   ├───────────────┴──────────────────────────────┴───────────────────────────────┤
   │ QUERY: understanding → planner → orchestrator                               │
   │   exact/lexical retrieval · semantic retrieval · structured query · visual   │
   │   → candidate fusion → rerank → authorization re-check                      │
   │   → result list | aggregation | grounded answer (RAG) → evidence/citations   │
   └──────────────────────────────────────────────────────────────────────────────┘
```

## 3. Modules (modular monolith first)

One deployable application with strict internal boundaries; services are extracted later only where scaling or
isolation needs it (section 7). Proposed package layout (new code; the current `ultimate/` stays runnable until
S15):

```
die/
  api/            HTTP routes, request/response schemas, versioning
  security/       AuthContext, key/OIDC verification, policy checks, audit events
  config/         typed settings, model registry, taxonomy and rule configuration loading
  ingestion/      intake, URL fetch (SSRF guard), hashing, job orchestration, status
  storage/        object store, registry repository, structured store repositories
  processing/     format detection, per-format parsers, page rendering
  ocr/            OcrEngine interface + engines (Tesseract first), preprocessing
  layout/         blocks, reading order, tables, regions
  understanding/  classification, entity/field/clause extraction, visual detectors (plugins)
  indexing/       chunking (structure-aware), lexical indexing, vector indexing, outbox consumer
  retrieval/      lexical, vector, structured, visual retrievers (all take AuthContext)
  query/          query understanding, planner (rule + LLM), plan validation
  ranking/        fusion, reranking, calibration
  aggregation/    deterministic executors (SQL)
  answer/         result assembly, RAG generation, evidence/citations, abstention
  observability/  logging, metrics, tracing, correlation ids
  shared/         ids, errors, clocks, typed value objects (no business logic)
```

Dependency rules (enforced by an import-linter contract in CI):

* `api` → `query`, `ingestion`, `security`; never the reverse.
* `retrieval`, `aggregation` → `storage` and `security` only; they never import `api` or `query`.
* `understanding` plugins depend on `shared` and their own interface; they do not touch storage directly (the
  pipeline persists their outputs).
* No module keeps per-request state in globals or singletons; a `RequestContext` (auth, deadline, ids) is passed
  explicitly. Shared process objects (clients, model handles) are stateless with respect to tenants.

Each module has a public interface (`Protocol` classes), its own tests, and no circular imports.

## 4. Technology roles

| Component | Responsibility | Choice | Status |
|---|---|---|---|
| Object storage | Original files, page images, OCR artefacts | S3-compatible (MinIO on-prem) or a mounted volume behind the same interface | Proposed |
| Document registry | Identity, versions, tenant, ACL, status, processing provenance | PostgreSQL | Proposed (approved direction) |
| Structured store | Fields, entities, clauses, regions, classifications; filters and aggregation | PostgreSQL | Proposed (approved direction) |
| Lexical index | Exact, phrase, identifier, keyword, Arabic stemming, trigram tolerance | PostgreSQL FTS + `pg_trgm` (spiked) | Proposed; revisit at scale (SEARCH_ARCHITECTURE 18) |
| Vector index | Semantic retrieval | Milvus (existing) | Kept |
| Queue / workers | Ingestion and processing jobs | Celery + Redis (existing) | Kept; outbox pattern added |
| Cache | Short-lived, tenant-keyed | Redis | Kept |
| OCR | Text, boxes, confidence | Tesseract behind `OcrEngine`; others pluggable | Interface proposed |
| Embeddings | Dense vectors | all-mpnet-base-v2 now; multilingual model decision open | Model registry proposed |
| Reranker | Precision on fused candidates | Cross-encoder behind `Reranker`; multilingual option open | Proposed |
| Query planner | Question → typed plan | Rule planner; optional on-prem LLM | Proposed |
| Answer generation | Grounded prose with citations | Optional on-prem LLM | Proposed |
| Evidence layer | Provenance for every result and number | Part of the canonical model | Proposed |

Why PostgreSQL is introduced (the justification required before adding a database): counting, summing, grouping,
date and amount filters, classification with confidence, document versions and ACLs need a transactional,
relational store with row-level security. That need exists regardless of which engine does keyword search. The
spike showed it also covers phrase/identifier search, Arabic stemming and fuzzy matching at PoC scale, so one new
component replaces several (registry, structured store, lexical index, audit log). Elasticsearch/OpenSearch is
not introduced: it would be a second search cluster to secure and operate without a requirement Postgres cannot
meet at the planned scale. The decision is revisited if lexical latency targets are missed at 1M+ chunks per
tenant.

## 5. Main data flows

**Ingest** (details in `INGESTION_ARCHITECTURE.md`):

```
upload / URL ─► authz ─► hash ─► object store (original) ─► registry: document + version (status=received)
   ─► job: detect format ─► parse/render pages ─► OCR where needed (versioned) ─► layout
   ─► canonical document (pages, blocks, tables, words with boxes, confidence)
   ─► understanding plugins: language, classification, entities, fields, clauses, regions (each with provenance)
   ─► chunking (structure-aware, page-anchored) ─► outbox events
   ─► indexers: lexical (Postgres), vector (Milvus), structured (Postgres) ─► status=indexed
```

**Query** (details in `QUERY_PLANNER.md`, `SEARCH_ARCHITECTURE.md`):

```
question + AuthContext ─► query understanding ─► planner ─► validated plan
   ─► orchestrator runs retrievers in parallel, each with tenant/ACL filters
   ─► fusion (exact tier first) ─► rerank ─► authorization re-check
   ─► documents | passages | number/table (deterministic) | grounded answer
   ─► evidence + confidence + explain(plan)
```

## 6. Confidence, precision and abstention

* Every result carries a `match_type` (EXACT, NORMALIZED_EXACT, OCR_TOLERANT, LEXICAL, SEMANTIC, ENTITY,
  STRUCTURED, VISUAL, CLASSIFICATION) and a calibrated confidence for that type. Weak fuzzy matches are never merged
  silently with exact ones.
* Every extracted value carries confidence and method; aggregations report high-confidence, uncertain and pending
  buckets separately.
* Answers are labelled FACT (directly supported by evidence), INFERENCE (derived, with the evidence shown) or
  UNCERTAIN; below thresholds the engine abstains.

## 7. From modular monolith to services

Extraction happens only when a measured need exists. Likely first candidates, because their resource profile
differs from the API:

| Candidate service | Reason to extract | Interface already defined by |
|---|---|---|
| Document processing / OCR workers | CPU/GPU heavy, bursty | `ocr/`, `processing/` job contracts |
| Embedding service | GPU, shared by ingest and query (already separate today) | `/embed` contract + model registry |
| Understanding plugins (layout, signature detection) | Model-specific runtimes | plugin interface |
| Retrieval / query API | Latency-sensitive, scales with users | `retrieval/`, `query/` |

Rules: no shared database access across extracted services except through the owning module's repository
interface; events (outbox) rather than synchronous chains for ingest; AuthContext propagated, never re-derived from
untrusted headers.

## 8. Cross-cutting

* **Configuration:** one typed settings object per process, validated at startup; secrets only from mounted
  files or a secret manager; tenant-level configuration (taxonomy, field schemas, synonyms, thresholds) in the
  database with versioning.
* **Observability:** correlation ids for request, query, document, job; per-stage timings; model and processing
  versions in every log line and result (`DEPLOYMENT_ARCHITECTURE.md`).
* **Versioning:** every derived artefact (OCR output, extraction, embedding, classification) stores the producer
  name and version, so reprocessing with a better model coexists with the old provenance until promoted.

## 9. Decisions

| Decision | Status |
|---|---|
| Do not restore the legacy search stack | Approved |
| PostgreSQL + Milvus with separated roles (section 4) | Approved direction; schema in `DOCUMENT_MODEL.md` for review |
| Modular monolith first, extraction on measured need | Proposed |
| Exact/identifier tier ahead of fusion | Proposed (spike evidence) |
| LLM only plans/phrases; never tenant, never authoritative numbers | Proposed |
| Object storage for originals (MinIO vs volume) | **Open**: depends on Xerox infrastructure |
| Multilingual embedding model | **Open**: decide on real Arabic documents |
| On-prem LLM for planning/answers (model, hardware) | **Open** |
| Signature/layout detector models (licence, accuracy) | **Open**: needs labelled pages |
