# Indexing strategy

## Six representations of every document

| # | Representation | Store | Purpose | Rebuildable from |
|---|---|---|---|---|
| 1 | Original file | object storage, content-addressed, per-tenant prefix | proof, download, reprocessing | (source of truth) |
| 2 | Canonical model | PostgreSQL | provenance, evidence, structure | 1 |
| 3 | Lexical | PostgreSQL full-text and trigram (OpenSearch at large scale) | exact, phrase, token, identifier, fuzzy, OCR | 2 |
| 4 | Vector | Milvus (HNSW, tenant partition key) | meaning, paraphrase, concepts | 2 + embedding model |
| 5 | Structured | PostgreSQL typed tables | filters, lookups, aggregation | 2 |
| 6 | Relations | PostgreSQL edge table | contextual and relational questions | 2 |

Representations 3 to 6 are projections. Losing one means an outage of that retriever, never data loss, and the
other retrievers keep answering.

## Lexical representation

For each block, passage, table row and file name:

| Field | Content | Serves |
|---|---|---|
| `exact` | tokens after Unicode NFKC, case folding, punctuation split, CamelCase split; positions kept | exact token and phrase |
| `stem` | language stemmer per detected language | lexical recall |
| `fold` | OCR-folded tokens (rn→m, vv→w, 0/O and 1/l/I in words, ligatures) applied identically to query and text | OCR-tolerant exact |
| `idents` | canonical identifiers (letters and digits only, lowercase) plus each identifier's digit and letter groups | identifiers in any spelling, partial identifiers |
| `trigram` | normalized text | fuzzy, substring, prefix and suffix |
| `labels` | heading, section path, key-value keys | field boosts |
| `vocabulary` | per-tenant term statistics (term, document frequency), refreshed incrementally | typo correction against the tenant's own words, IDF |

Exact retrieval reads only `exact`, `idents` and file names. The fold and trigram forms widen recall but are
labelled as their own match types.

**Scale path (decision D3).** PostgreSQL serves up to roughly 5 to 10 million blocks per deployment with the
index-usable lookup path (D1). Beyond that, or when BM25 relevance tuning, many languages or very high query
rates are needed, the same fields move to OpenSearch. The tenant is the routing key and a mandatory filter, and
the retriever contract does not change. The switch is a configuration change validated by the evaluation suite.

## Vector representation

* **Granularities.** Passage (always), section and table row (when present), and document (title, type, key
  fields, and optional summary). Each is a separate Milvus field or collection with a `unit_type`, so a query can
  target the right granularity.
* **Context.** The embedded text is the unit plus a short header: document type, title and section path. This
  is stored as `context_text` so it can be reproduced.
* **Models.** From the model registry: name, revision, checksum, dimension, normalization, max tokens, language
  coverage, calibrated floor. Collections are named by model and dimension, and a document version records the
  model it was embedded with. Changing a model creates a new collection, filled in the background; queries switch
  when it is complete.
* **Tenancy.** Tenant as partition key plus a mandatory filter in the single query builder, with tenant and
  identifier values validated.
* **Optional hybrid sparse vectors** (learned sparse, for example SPLADE) are a later addition that can sit
  beside dense vectors in the same collection when evaluation shows a gain.

## Structured representation

* `values`: every typed value with its span and role, indexed by (tenant, kind, role, value), for dates,
  amounts with currency, quantities with units, percentages, durations and identifiers.
* `fields`: named fields from packs, pointing at values.
* `mentions`, `entities`, `entity_links`: entity search with aliases, plus cross-document identity
  ("Acme Corp" in 40 documents).
* `table_cells` with header links: questions over tables ("which items cost more than 100") become structured
  queries.
* All computations used by aggregation run here, in SQL, and return the contributing documents and spans.

## Relation representation

* `relations(subject, predicate, object, qualifiers, span, confidence, extractor)` with the predicate vocabulary
  defined by packs (`signed`, `party_to`, `may_terminate`, `ships_to`, `prescribed`, `reports_to` ...).
* It starts in PostgreSQL. A graph database is considered only if multi-hop queries over large graphs become a
  measured need.

## Write path and consistency

```
processing → canonical model (version N+1, not current)
           → indexing jobs: lexical, structured, relations (same transaction or same job)
           → vector job (separate; embedding service)
           → all done? → mark version N+1 current, retire version N → purge N's index rows after the retention period
```

* Each index write is idempotent and keyed by (unit_id, version), so retries and replays are safe.
* An index outbox table records pending index work, so an index that was down is caught up automatically.
* A reconciliation job compares counts per document and version across stores, and re-queues mismatches.

## Capacity planning (per 1 million passages)

| Store | Approximate size |
|---|---|
| PostgreSQL text, lexical and structured | 3 to 6 times the extracted text |
| Milvus, 768-dimension float vectors with HNSW | about 3 GB of vectors plus about 1 GB of graph; scalar quantization optional |
| Object storage | the originals, plus page images when rendering is cached |
