# Migration plan: from `ultimate/` to the Document Intelligence Engine

Status: plan for review. Each phase is approved separately, ends with tests and a recorded comparison against the
baseline, and never removes working functionality before its replacement is proven. The current `/search` and
ingest paths stay available until S14.

Rules for every phase: no restoration of the legacy search stack; no in-place destructive change to the existing
Milvus collection or embeddings; tenant leakage = 0 gate; hardcoded-data scanner counts may only go down.

## Phase overview

| Phase | Goal | Changes current behaviour? |
|---|---|---|
| S0 | Baseline and production reconciliation | No |
| S1 | Configuration, hardcoded-data removal, repository restructuring skeleton, observability basics | Config only (same behaviour) |
| S2 | Canonical document model (code) and request context | No |
| S3 | Ingestion/OCR pipeline on the canonical model (originals preserved, pages, per-page OCR) | Ingest, behind a flag |
| S4 | Document registry and structured store in PostgreSQL (dual write, backfill) | Additive |
| S5 | Lexical retrieval (Postgres FTS, exact/identifier tier) | New path only |
| S6 | Semantic retrieval on the new interfaces (configurable k, model registry) | New path only |
| S7 | Query planner (rules first) | New path only |
| S8 | Hybrid retrieval (tiered fusion) | New path only |
| S9 | Reranking and calibration (abstention) | New path only |
| S10 | Classification, extraction, aggregation | New capability |
| S11 | Visual capabilities (signatures, stamps, handwriting as plugins) | New capability |
| S12 | Clauses, RAG with evidence | New capability |
| S13 | Shadow comparison old vs new | No |
| S14 | Production cut-over per tenant | Yes, opt-in then default |
| S15 | Remove legacy implementation | Deletion |

## Details

### S0 Baseline and reconciliation
* **Do:** obtain the deployed tree and run `scripts/forensics/compare_deployed.py`; answer P1 to P5
  (`SEARCH_ARCHITECTURE.md` section 0); receive the customer evaluation set; agree metric targets
  (`EVALUATION_STRATEGY.md` section 3).
* **Exit:** production differences documented; targets signed off; golden baselines current.
* **Owner dependency:** previous team (deployed tree), customer (documents and questions).

### S1 Configuration, hardcoded data, skeleton
* **Do:** typed settings module and model registry (one source per value; conflicting defaults removed);
  `HARDCODED_DATA_AUDIT.md` items B1 to B6, C1 (fail closed on missing tenant), C2, D1 to D7, A9, E2; move legacy
  scripts to `tools/legacy/`; create the `die/` package skeleton with module interfaces and an import-linter
  contract (`DOCUMENT_INTELLIGENCE_ARCHITECTURE.md` section 3); JSON logging with request ids; remove source bind
  mounts from compose; CI builds versioned images.
* **Tests:** settings validation tests; scanner check in CI (zero person/org/company-domain literals outside
  `tests/`, `tools/legacy/` and the legacy search modules); full existing suite unchanged; golden maps unchanged.
* **Exit:** same behaviour, measurable configuration, reproducible images.

### S2 Canonical document model and request context
* **Do:** `DOCUMENT_MODEL.md` types in code; `AuthContext`/`RequestContext` passed explicitly (removes per-request
  state from singletons, KD-SRCH-03, which also allows concurrent searches later); single `normalize()`.
* **Tests:** model round-trip tests; normalization tests (English, Arabic, identifiers, OCR fold).
* **Exit:** model reviewed; no behaviour change.

### S3 Ingestion and OCR pipeline
* **Do:** object storage for originals; magic-byte detection; per-page triage (fixes mixed PDFs); `OcrEngine` with
  Tesseract words/boxes/confidence and `eng+ara`; deterministic page order; page numbers; explicit job states;
  per-format tests for every listed format; new pipeline writes the canonical model (behind `INGEST_PIPELINE=v2`).
* **Tests:** extraction suite per format; OCR CER on labelled pages; the KD-OCR strict-xfail tests flip to pass and
  their markers are removed in the same change.
* **Exit:** v2 ingest produces canonical documents for the golden corpus with correct pages.

### S4 Registry and structured store
* **Do:** PostgreSQL schema (`DOCUMENT_MODEL.md`), RLS policies, migrations; dual write from ingest; outbox to
  indexers; backfill existing documents from Milvus chunk text (marked `source=backfill`, page numbers unknown
  until reprocessed from originals where available).
* **Tests:** RLS tests; cross-tenant tests; dual-write consistency job; duplicate re-ingest is a no-op.
* **Exit:** every Milvus chunk has a registry row; counts reconcile per tenant.

### S5 Lexical retrieval
* **Do:** `retrieval.lexical` over Postgres FTS (per-language configurations, identifier canonical forms, phrase,
  trigram on OCR-folded text); exact/identifier tier.
* **Tests:** golden exact/identifier/Arabic/OCR cases on the new path; needle case at 10k/100k chunks.
* **Exit:** exact recall@10 ≥ 0.99 on golden and scale tenants.

### S6 Semantic retrieval
* **Do:** `retrieval.vector` with `AuthContext`, configurable k and ef, model registry check; multilingual model
  evaluation on customer Arabic data (decision record; re-embedding only if approved, into a new collection).
* **Exit:** semantic metrics recorded against the MPNet baseline.

### S7 Query planner
* **Do:** rule planner, plan schema and validation, `explain`; optional LLM planner behind a flag (on-prem).
* **Tests:** planner golden (question → plan), adversarial plans.

### S8 Hybrid retrieval
* **Do:** orchestrator, tiered fusion (exact first, RRF within tiers), result model with `match_type`.
* **Exit:** golden and customer metrics at or above targets for retrieval queries; no regression vs v1 on any
  query type beyond the agreed tolerance.

### S9 Reranking and calibration
* **Do:** reranker with health status; calibrated floors per match type; `none` answers.
* **Exit:** no-result precision and false-positive targets met.

### S10 Classification, extraction, aggregation
* **Do:** taxonomy configuration, classifier, field/entity extractors with provenance, aggregation executor,
  confidence buckets; `[golden xdemo]` aggregation questions become runnable.
* **Exit:** aggregation accuracy and classification/extraction targets met.

### S11 Visual capabilities
* **Do:** region detection plugins (signature first), page images, signer association with uncertainty.
* **Exit:** precision/recall measured on labelled pages; capability flagged experimental until targets met.

### S12 Clauses and RAG
* **Do:** clause segmentation/typing; grounded generation with citations and FACT/INFERENCE/UNCERTAIN labels;
  citation post-check.
* **Exit:** citation precision and answer accuracy targets met.

### S13 Shadow comparison
* **Do:** both paths on the same queries; per-type comparison reports; fix or accept differences explicitly.

### S14 Cut-over
* **Do:** per-tenant opt-in, then default; `/search` request contract kept as a compatibility adapter onto the
  new engine (including `bucketId`/`connectionId` mapping to collections).
* **Exit:** customer acceptance; rollback path tested.

### S15 Remove legacy implementation
* **Do:** delete the legacy search code (`ultimate_ui.py` search block, `semantic_pipeline.py`,
  `query_enhancement.py`, `constraint_ranking.py`, `MetadataIndex`, dead fuzzy layer, corpus JSON files) and the old
  ingest path; hardcoded-data scanner reaches zero for corpus vocabulary in production code.
* **Exit:** only the engine remains; documentation updated; strict-xfail pins for legacy defects removed with the
  code they described.

## Recommended next milestone

**S1 only** (plus the S0 requests running in parallel): it changes no search or ingest behaviour, removes the
production coupling to sample data and developer machines, and creates the module skeleton every later phase
builds on. The existing suites and golden baselines must stay identical at the end of S1.
