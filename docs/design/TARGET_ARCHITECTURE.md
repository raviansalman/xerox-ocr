# Target architecture

## Layers

```
 sources: upload, URL, folder, connector                      callers: UI, REST, SDK, RAG clients
        │                                                                 │
 ┌──────▼─────────────────────────────┐                   ┌───────────────▼──────────────────────┐
 │ INGESTION                          │                   │ QUERY INTELLIGENCE                   │
 │ register → store original → detect │                   │ normalize → understand → plan →      │
 │ → job                              │                   │ validate plan                        │
 └──────┬─────────────────────────────┘                   └───────────────┬──────────────────────┘
        │ job                                                             │ QueryPlan
 ┌──────▼─────────────────────────────┐                   ┌───────────────▼──────────────────────┐
 │ DOCUMENT PROCESSING                │                   │ RETRIEVAL (parallel, independent)    │
 │ parse (native) │ OCR │ layout/vision│                  │ exact │ lexical │ fuzzy/OCR │ semantic│
 │        ↓ normalize                 │                   │ contextual │ entity │ metadata │      │
 │ CANONICAL DOCUMENT MODEL           │                   │ structured                           │
 │ blocks, tables, spans, provenance  │                   └───────────────┬──────────────────────┘
 │        ↓ understand                │                                   │ Candidates
 │ entities, fields, relations, types │                   ┌───────────────▼──────────────────────┐
 └──────┬─────────────────────────────┘                   │ FUSION → RERANK → EVIDENCE CHECK →   │
        │ canonical model (versioned)                     │ ABSTAIN?                             │
 ┌──────▼─────────────────────────────┐                   └───────────────┬──────────────────────┘
 │ INDEXING (one writer per index)    │                                   │
 │ lexical │ vector │ structured │    │◄──── read ────────────────────────┤
 │ relation                           │                   ┌───────────────▼──────────────────────┐
 └────────────────────────────────────┘                   │ ANSWERING                            │
                                                          │ results │ computed answers │ grounded│
 cross-cutting: security & tenancy, configuration &       │ answers (RAG, optional) with         │
 domain packs, model registry, observability, evaluation  │ verified citations                   │
                                                          └──────────────────────────────────────┘
```

The canonical document model is the hinge. Processing writes it once, versioned. Every index is derived from it
and can be rebuilt from it. Retrieval never reads raw files.

## Modules and contracts

Each module has one responsibility, a typed input and output, and no knowledge of the modules beside it except
through those types. "Today" names the v1 code it grows from.

| Module | Responsibility | Input → output | Today |
|---|---|---|---|
| `ingestion` | Accept sources, store the original immutably, dedupe, detect type, create jobs | stream + metadata → `DocumentRef`, job | `ingest/pipeline.register`, `processing/detect` |
| `parsers` | Format-specific extraction of native content with positions | file → `RawDocument` (pages, blocks, tables, images, positions) | `processing/parsers` (no blocks or tables yet) |
| `ocr` | Text with word boxes and confidence for image regions | page image → `OcrPage` | `processing/ocr` |
| `layout` | Reading order, block types (heading, paragraph, list, table, caption, header/footer), table structure, figure regions; optional vision model | `RawDocument` + `OcrPage`s → laid-out pages | partly in `processing/chunking`, `signatures` |
| `normalization` | Unicode, whitespace, hyphenation, OCR folding, identifier canonical forms; one implementation used by indexing and querying alike | text → normalized forms | `text.py` |
| `canonical_model` | Types, invariants, persistence and versioning of the document model | laid-out pages → `CanonicalDocument` | `models.py`, `storage/repo` |
| `extraction` | Generic entities, values, identifiers, sections, key-value pairs, table records, relations; domain packs add their own | `CanonicalDocument` → annotations with spans | `understanding/*` |
| `classification` | Document type and domain, from pack definitions | `CanonicalDocument` → labels with confidence | `understanding/classify` |
| `indexing/lexical` | Exact, phrase, token, identifier, fuzzy, OCR-folded representations | canonical units → lexical index | chunk `tsv*`, `idents`, trigram |
| `indexing/vector` | Embeddings at several granularities, model-versioned | canonical units → vector index | `indexing/vectors`, `embeddings` |
| `indexing/structured` | Typed fields, entities, mentions, table cells | annotations → relational tables | `fields`, `entities`, `clauses` |
| `indexing/relations` | Subject–predicate–object edges with evidence | relations → edge table | (none) |
| `query/understanding` | Normalize, detect language, identifiers, quotes, entities, dates, amounts, concepts | question → `ParsedQuestion` | `search/planner` |
| `query/planning` | Choose intent and retrieval strategies, build filters and computations | `ParsedQuestion` → `QueryPlan` | `search/planner`, `plan` |
| `query/validation` | Check a plan against the schema, the tenant's packs and limits | `QueryPlan` → valid plan or error | (none) |
| `retrieval/*` | One retriever per mechanism, all with the same contract | `QueryPlan` + `AuthContext` → `Candidate`s | `search/retrieval`, `engine._vector` |
| `ranking` | Fusion, tiering, reranking, calibration | `Candidate`s → ranked `Result`s | `engine._rank` |
| `evidence` | Re-check each result's span against the canonical text, compute highlights, decide abstention | `Result`s → verified `Result`s | (implicit) |
| `aggregation` | Deterministic computations over structured data | plan → `ComputedAnswer` with contributing documents | `engine._count`, `_amounts`, `_group` |
| `answering` (RAG) | Optional grounded answers from verified evidence, with citation checking | question + evidence → `GroundedAnswer` | (none) |
| `security`, `tenancy` | Identity, roles, tenant context, row-level security, audit | request → `AuthContext` | `security.py`, `storage/db` |
| `config`, `packs`, `model_registry` | Settings, domain packs, model versions | environment / database → typed config | `config.py`, `resources/*` |
| `observability` | Structured logs, metrics, traces, query explanation | events → sinks | `logging_setup` (logs only) |
| `evaluation` | Datasets, runners, metrics, reports, gates | dataset + engine → report | tests (no metrics yet) |
| `api` | HTTP contract and UI | HTTP → module calls | `api/` |

Rules that keep the boundaries:

* Retrievers do not call each other, and fusion does not know which algorithm a retriever uses.
* Only `indexing/*` writes indexes, and only `canonical_model` writes canonical data.
* Domain knowledge enters only through `packs`. A grep for a domain term in core modules is a CI failure (see
  DOMAIN_PACKS.md).
* Every cross-module type is versioned. Changes are additive, or come with a migration.

## Target package layout

```
docintel/
  api/              ingestion/        parsers/          ocr/              layout/
  normalization/    canonical_model/  extraction/       classification/
  indexing/         lexical/  vector/  structured/  relations/
  query/            understanding/  planning/  validation/
  retrieval/        exact/  lexical/  fuzzy/  semantic/  contextual/  entity/  metadata/  structured/
  ranking/          evidence/         aggregation/      answering/
  security/         tenancy/          storage/          config/   packs/   model_registry/
  observability/    evaluation/
packs/              business/  legal/  finance/  medical/  supply_chain/  hr/   (configuration + optional models)
evaluation/         datasets (outside the package; never imported by production code)
tests/fixtures/     synthetic test data only
```

The move is mechanical and is done module by module (MIGRATION_PLAN.md, phase 2). Import paths of the v1 code
keep working through thin re-exports until the move is complete.

## Deployment

The services stay few. A module becomes its own process only when it scales differently:

| Process | Contains | Scales with |
|---|---|---|
| API | api, security, query intelligence, retrieval, ranking, evidence, aggregation | queries per second (stateless replicas) |
| Processing workers | parsers, layout, normalization, extraction, classification, canonical model writes | documents per hour (CPU) |
| OCR workers | ocr | scanned pages per hour (CPU or GPU) |
| Indexing workers | lexical, vector, structured and relation index writes | write volume; isolates index outages from processing |
| Embedding service | embedding models from the registry, batched | text volume (GPU recommended) |
| Reranker service | cross-encoder reranking, batched | queries per second (GPU recommended) |
| Answer service (optional) | grounded answer generation and citation checking | RAG queries |
| PostgreSQL, object storage, Milvus, queue (Redis now, a durable broker optional) | state | data size |

Work flows through queues between processing, OCR and indexing. That lets each stage retry on its own,
backpressure the one before it, and replay from the canonical model.

## Non-functional targets

| Area | Target |
|---|---|
| Tenant isolation | Zero cross-tenant results, counts or evidence: enforced in the database, in every vector search filter and in object storage, and tested on every query type |
| Exact retrieval | 100% recall of exact phrases and identifiers present in the canonical text (invariant tests) |
| Latency, interactive search | p95 under 500 ms up to 1 million blocks per tenant on recommended hardware, without the optional reranker and answer service |
| Ingestion | Linear in workers; per-document cost reported per stage |
| Reproducibility | Every derived row records extractor, model and configuration versions |
| Observability | Per-stage metrics and traces for ingestion and queries; query explanations on request |
| Security | Least-privilege roles, secrets from the environment or a secret manager, audit of every write and every administrative action |
