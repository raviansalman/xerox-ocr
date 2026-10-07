# Target search architecture: natural-language document query engine

> Part of the Document Intelligence Engine design (`DOCUMENT_INTELLIGENCE_ARCHITECTURE.md`). This document covers
> retrieval and answering; the document model, ingestion, planner, security, deployment, evaluation and migration
> each have their own document. **Phase labels below (S1 to S7) are from the first draft;** the authoritative
> numbering is `MIGRATION_PLAN.md` (S0 to S15), mapped in section 24.

Status: **design only.** Nothing in this document is implemented. The current search code is frozen: no
restoration of `LOCATION_PEERS`, `threading` or the legacy query-understanding stack, no Milvus schema, chunking or
embedding changes. Companion documents: `docs/SEARCH_FORENSICS.md` (what exists today),
`docs/SEARCH_QUERY_EXAMPLES.md` (39 specified queries).

Guiding principles, agreed in review:

* **Search is query-driven, not index-driven.** The user asks; the system decides which retrievers, filters and
  computations answer it.
* **Recover the behaviour, not the old implementation.** The legacy StorageChain code tells us what was intended;
  it is not the reference architecture (restoring it leaked across tenants, `SEARCH_FORENSICS.md` section 7).
* **Deterministic components compute; the language model interprets and presents.** Counts, sums, filters and
  date logic run against indexes and tables, never inside an LLM.
* **Tenant authorization is an input to every retriever**, plus a final defence-in-depth check.

Evidence labels as in `SEARCH_FORENSICS.md`: VERIFIED, TRACED, INFERRED, UNKNOWN. Design choices that rest on
the scratch spikes described in section 18 say so.

---

## 0. Production reconciliation (blocking input)

Production is UNKNOWN (`SEARCH_FORENSICS.md` section 1). The design below does not depend on the answer, but the
migration and the risk statement do. Requested from the previous team:

| # | Item | Decides |
|---|---|---|
| P1 | `docker exec <api> tar c -C /app src ultimate_ui.py search_api.py` from every search and worker host | Run `ultimate/scripts/forensics/compare_deployed.py`; which of the missing names exist in production |
| P2 | `docker_byoc.env` / `docker_byoc_staging.env` with secrets removed | Production tuning (`MILVUS_SEARCH_EF`, `SEARCH_MIN_SCORE`, `CHUNK_SIZE`, strictness) for a faithful baseline |
| P3 | Does the search container load `data/metadata_cache/` (MetadataIndex)? | Whether production has the shared-index tenant exposure (do **not** assume it until P1/P3 answer) |
| P4 | Production `/search` responses for the 27-query suite in `scripts/compare_staging_prod_search.py`, run by them on their own tenant | Whether production performs exact/lexical retrieval (content-scan supplement) |
| P5 | Milvus server version and collection row counts per tenant (no content) | Migration effort and scale targets |

Until P1 to P3 are answered, nobody states that production has, or does not have, the tenant problem.

## 1. Current search architecture (summary)

`POST /search` → tenant from the API key → `enhance_query` fails (ImportError) → **vector search only** in the
default `both` mode (semantic path gated on a flag that is never set) → best chunk per file + substring word-overlap
bonus → sort by (query is a substring of the chunk, filename token hits, score) → zero-result fallbacks.
`searchMethod=semantic` runs a separate pipeline with a cross-encoder and many regex gates. Full detail:
`SEARCH_FORENSICS.md` section 2.

## 2. Current limitations (measured)

| Limitation | Evidence |
|---|---|
| No exact-text retrieval: exact phrases only reorder the ≤ 50 vector candidates | Needle ranked 61/61 by MPNet cosine, never returned (VERIFIED) |
| Exact recall collapses with tenant size, even with a lexical-biased embedder | Scale probe, stand-in embedder, 20 planted exact phrases: 20/20 at 2k chunks, 4/20 at 10k, 4 to 14/20 at 25k (VERIFIED, section 19) |
| The legacy full-scan exact approach cannot scale | `query_all_chunks` at 25k chunks: truncated to 16,384 rows, 17.3 s (VERIFIED) |
| Not hybrid: no lexical candidate generator, no fusion | VERIFIED |
| No structured fields: dates, parties, amounts, jurisdiction, document type are not stored | KD-DATA-01 (VERIFIED) |
| No page numbers | KD-OCR-02 (VERIFIED); blocks page-level evidence |
| No aggregation, classification or visual understanding | TRACED |
| English-centric embedding (all-mpnet-base-v2); Arabic only through tokens shared with the query | TRACED; cross-language untested |
| "No results" impossible in default mode | KD-SRCH-06 (VERIFIED) |
| Per-request state on singletons; searches serialized per process | KD-SRCH-03 |

## 3. Search capability matrix: current vs target

| Capability | Current | Target mechanism | Phase |
|---|---|---|---|
| Exact phrase | BROKEN (re-ordering only) | Lexical index phrase query → exact tier | S2 |
| Exact identifier (`INV-2026-00481`, `17/2024`) | Works on tiny corpora via vectors | Identifier tokens + canonical form + n-gram substring | S2 |
| Normalized match (case, whitespace, punctuation, Unicode) | PARTIAL | One `normalize()` used at ingest and query; stored normalized field | S1/S2 |
| Joined/split forms (`Storage Chain` / `StorageChain`) | MISSING | Compound-split and join variants in the analyzer | S2 |
| Ordered words / proximity | MISSING | Phrase with slop / proximity query | S2 |
| Word boundary | PARTIAL | Token-based matching only (no raw substring scoring) | S2 |
| OCR-tolerant match | MISSING | OCR-confusion folded field + bounded trigram similarity, flagged | S3 |
| Semantic retrieval | WORKING (capped at 50) | Dense retrieval, configurable k, multilingual model decision | S2 |
| Hybrid fusion | MISSING | Per-retriever candidates → tiered fusion → rerank | S2 |
| Rerank | PARTIAL (semantic mode only; 1.0 fallback) | Cross-encoder on fused top N, health-reported | S3 |
| Filename search | PARTIAL (fallback only) | Structured field, exact and token | S2 |
| Entity search (person, org, location) | BROKEN | Extracted entities table + lexical fallback | S4 |
| Metadata filters (type, date, amount, jurisdiction) | MISSING | Structured fields with provenance | S4 |
| Document classification | MISSING | Ingest-time classifier with confidence | S4 |
| Clause search | MISSING (dead regexes) | Clause segmentation + clause type index | S5 |
| Aggregation (count, sum, group by, percentage) | MISSING | SQL over structured tables | S4 |
| Signature / visual | MISSING | Page-region detection, stored with bbox and confidence | S6 |
| Question answering (RAG) | MISSING | Evidence-grounded generation over retrieved passages | S6 |
| Evidence and provenance | PARTIAL (file, chunk text) | document, page, span, bbox, match type, scores | S1 |
| Tenant isolation | WORKING (Milvus filter) | Filter in every retriever + DB row-level security + final check | S1 |
| Arabic | PARTIAL (tokens only) | Arabic analyzer (normalization, light stemming); multilingual embeddings TBD | S2/S3 |

Phases S1 to S6 are defined in section 24.

## 4. Query taxonomy

A query is classified into one **answer kind** and a set of **operators**:

| Answer kind | Meaning | Example |
|---|---|---|
| `documents` | ranked documents with evidence | `FOR IMMEDIATE RELEASE` |
| `passages` | ranked passages (clause, paragraph) | `termination clause in the Riyadh contract` |
| `number` | a deterministic count, sum, average, min, max or ratio | `How many NDAs do we have?` |
| `table` | grouped numbers | `contracts expiring per month in 2027` |
| `quoted_figure` | a figure stated in a document, returned with its source, not computed | `How many employees were paid in 2024?` with only a summary document |
| `generated_answer` | prose synthesized from cited evidence | `What notice period do our termination clauses require?` |
| `none` | nothing matched | `zzqxunknownzzq` |

Operators (a query may need several): `EXACT`, `LEXICAL`, `SEMANTIC`, `FILENAME`, `ENTITY`, `METADATA`,
`DATE`, `NUMERIC`, `CLASSIFICATION`, `CLAUSE`, `VISUAL`, `AGGREGATE`, `COMPARE`, `RAG`.

## 5. Query routing design

Routing produces a typed **QueryPlan**, not a mode string. Two planners behind one interface:

1. **Rule-based planner (always available, deterministic).** Recognizers for quoted phrases, identifiers
   (letter-digit patterns, `No.`, `#`, `/` and `-` numbers), dates and ranges, amounts with currency, known
   document types (from the configured taxonomy), counting cues (`how many`, `number of`, `percentage`, `total`),
   person-name shape, Arabic script. Covers the exact/lexical/semantic/filename/date cases on its own.
2. **LLM planner (optional, on-prem model).** Only for composite questions the rules cannot parse. Input: the
   question and the **schema** of available fields and document types. Never document text, never other tenants'
   data. Output: QueryPlan JSON, validated against a strict schema; invalid or low-confidence plans fall back to the
   rule planner + hybrid retrieval. The planner cannot set or change the tenant: the executor injects it.

Composite queries are a conjunction (or explicit union) of operator nodes; the plan records which retriever
serves each node so results remain independently measurable.

```
QueryPlan {
  answer_kind: documents | passages | number | table | quoted_figure | generated_answer
  text_query: { phrases[], identifiers[], terms[], semantic_text, language }
  filters:    [ { field, op, value, source: "explicit" | "inferred", confidence } ]
  doc_types:  [ ... ]                     # from the configured taxonomy
  aggregate:  { op: count|sum|avg|min|max|ratio, field?, group_by?, distinct_on: "document_id" }?
  retrievers: [ lexical, vector, structured, visual ]   # which ones to run
  limit, explain: true
}
```

## 6. Exact search architecture

Requirements: exact retrieval never depends on vector similarity; the reason for every match is recorded.

* **Index:** a lexical index over chunk text with positions (phrase queries), plus a document-level index over
  filename and title.
* **Analyzer** (identical at ingest and query): Unicode NFKC, case folding, whitespace collapse, punctuation
  tokens removed (the Milvus ICU spike showed spaces and `#`/`-` emitted as tokens without a filter), identifiers
  kept whole **and** split (`inv-2026-00481` → `inv-2026-00481`, `inv`, `2026`, `00481`, canonical
  `inv202600481`), dotted numbers kept (`12.4`), slash numbers kept (`17/2024`).
* **Match types, in precedence order:** `exact_phrase` → `exact_identifier` → `normalized_identifier` →
  `filename_exact` → `all_terms` (BM25) → `ocr_tolerant` (section 13).
* **Substring fallback for identifiers** (partial ids such as `00481`): n-gram index (Milvus `NGRAM` / Postgres
  `pg_trgm`), only for tokens with digits or ≥ 4 characters.
* **Precedence rule:** a document with an `exact_phrase` or `exact_identifier` hit for a quoted or identifier-shaped
  query is ranked above documents without one. Reason: in the spike, plain RRF fusion of dense + BM25 ranked the
  needle **second**, behind a heron note (VERIFIED). Exact intent needs an explicit tier, not just a fusion weight.

## 7. Semantic search architecture

* Dense retrieval in Milvus as today (all-mpnet-base-v2, 768, COSINE, HNSW M=32/efConstruction=200), tenant
  filtered. Keep the model initially so existing vectors stay valid.
* Raise the per-retriever candidate count from the hard 50 to a configured k (for example 200 chunks) with `ef`
  ≥ k, measured for latency (section 19).
* Chunk → document aggregation by max (or top-2 mean) chunk score, keeping the best passages as evidence.
* **Multilingual decision (open):** all-mpnet-base-v2 is English-centric. For Arabic and cross-language queries,
  evaluate a multilingual model (for example a multilingual E5 or BGE-M3 class model) on real Xerox documents
  before changing anything: it would require re-embedding every chunk and a new collection.

## 8. Hybrid search architecture

```
QueryPlan ──► AuthContext(tenant, user, groups)
   │
   ├─► Lexical retriever   (tenant-filtered)  ── candidates L (k_L) with match_type, bm25
   ├─► Vector retriever    (tenant-filtered)  ── candidates V (k_V) with cosine
   ├─► Structured retriever(tenant-filtered)  ── candidate set S (filters, entities, types)
   └─► Visual retriever    (tenant-filtered)  ── candidate set G (page regions)
            │
            ▼
   Candidate fusion (section 17) ─► Rerank (top N) ─► Authorization re-check ─► Results / Aggregation / RAG
```

* Each retriever returns typed candidates `{document_id, chunk_id?, page?, score, match_type, retriever}`;
  candidate lists are logged per query (without text) for evaluation.
* Structured filters are **constraints** applied in each retriever's query (pre-filter), not post-filters on a
  truncated list.

## 9. Structured query architecture

A **document registry** and **field store** become the system of record for documents (they do not exist today;
Milvus only holds chunks):

| Table | Key columns |
|---|---|
| `documents` | `document_id`, `tenant_id`, `acl_groups[]`, `filename`, `sha256`, `mime`, `pages`, `language`, `doc_type`, `doc_type_confidence`, `classifier_version`, `ingested_at`, `status` |
| `pages` | `document_id`, `page_number`, `ocr_used`, `ocr_confidence`, `image_ref` |
| `chunks` | `chunk_id`, `document_id`, `page_number`, `char_start`, `char_end`, `text`, `text_norm`, `text_ocrfold`, `milvus_pk` |
| `fields` | `document_id`, `field` (`issue_date`, `expiry_date`, `amount`, `currency`, `party`, `jurisdiction`, `invoice_number`, …), `value_text`, `value_num`, `value_date`, `confidence`, `extractor_version`, `page_number`, `span` / `bbox` |
| `entities` | `document_id`, `type` (person, org, location), `surface`, `normalized`, `page_number`, `span`, `confidence` |
| `clauses` | `document_id`, `clause_type`, `clause_ref` (`12.4`), `page_number`, `span`, `confidence` |
| `regions` | `document_id`, `page_number`, `region_type` (`signature`, `handwriting`, `stamp`, `table`), `bbox`, `confidence`, `detector_version` |

Every row carries `tenant_id`; every extracted value carries provenance (page, span or bbox) and confidence.
Re-ingesting a file replaces its rows (fixes duplicate chunks, KD-DATA-02, via `sha256` + `document_id`).

## 10. Aggregation architecture

* Aggregations execute as **SQL over `documents` / `fields` / `regions`**, always with `tenant_id = :tenant`
  (and row-level security, section 14), `DISTINCT document_id` by default.
* The executor returns the number **and** the contributing document ids (evidence), plus an `uncertain` set
  (confidence between a lower and upper threshold) and a `pending` count (documents not yet classified/extracted).
  Example answer: "2 NDAs (plus 1 possible, 0 pending)".
* Currency sums group by currency unless a configured rate table exists; the answer states which.
* "Quoted figures" (a number stated inside a document) are returned as citations, not recomputed.
* The LLM may phrase the result; it never produces the number.

## 11. Document classification architecture

* Configured taxonomy per deployment (Xerox: invoice, purchase order, NDA, service contract, employment agreement,
  lease, policy, government form, tax document, financial statement, letter, other), with an explicit
  "contract family" grouping used by aggregations.
* Ingest-time classifier, layered: (1) high-precision rules on title/first page (`MUTUAL NON-DISCLOSURE AGREEMENT`,
  `TAX INVOICE`, `فاتورة ضريبية`); (2) embedding nearest-prototype classifier using the existing embedder;
  (3) optional on-prem LLM for the remainder. Output: label, confidence, method, version.
* Stored in `documents`; reclassification is a background job keyed by `classifier_version`.
* Evaluation: per-class precision/recall on a labelled Xerox sample; counts are only as good as this number, and
  the answer exposes the uncertain bucket.

## 12. Visual and signature search architecture

Text search cannot see a handwritten signature. Target capability (not implemented):

* Page images are already rendered for OCR; persist page renders (or regenerate on demand) with `page_number`.
* **Region detection** per page: signature, handwriting, stamp/seal, table, using a document-layout / signature
  detector model run on-prem. Store `bbox`, `confidence`, `detector_version` in `regions`.
* **Signer association:** link a signature region to the nearest person entity or "Signed by/Name:" line on the
  same page; store as a separate low-confidence relation, never as fact.
* Distinguish in answers: `signature_detected` (visual), `signature_line_text` (typed name on a signature line),
  `no_signature_detected` (never "unsigned" as a certainty).
* Requirements before build: detector model selection and licence check, a labelled sample of Xerox pages
  (signed, unsigned, stamped, faxed), measured precision/recall, page-number fix (KD-OCR-02).

## 13. OCR-aware retrieval

* Store three text forms per chunk: `text` (as extracted), `text_norm` (normalized), `text_ocrfold` (normalized,
  then OCR confusions folded: `rn→m`, `vv→w`, `0→o` and `1/l/|→i` inside alphabetic tokens, `5→s` and `8→b` inside
  alphabetic tokens). The query is folded the same way. This makes `California` match `Califomia`
  deterministically (both fold to `califomia`). Trigram similarity alone scored that pair 0.55, below the 0.6
  default threshold (Postgres spike, VERIFIED), so thresholds alone are not enough.
* Bounded fuzzy matching (trigram or edit distance) only for tokens of ≥ 6 characters, only when exact and folded
  matches return fewer than k results, and always labelled `ocr_tolerant` / `fuzzy_name`.
* Use OCR confidence: matches inside low-confidence OCR pages are labelled and ranked below clean matches of the
  same type.
* Never rewrite the stored source text (KD-OCR-01 showed the current cleaner corrupts uppercase words).

## 14. Tenant authorization model

```
API key / (later) OIDC token ─► AuthContext { tenant_id, user_id, roles, acl_groups }   (immutable, per request)
```

* **Injected, not inferred:** retrievers take `AuthContext` as a required argument. No retriever has a code path
  without it (the current `search_similar` silently drops the filter when `user_id` is empty; the target code
  raises).
* **At candidate generation:** Milvus `user_id` filter (partition key on a future collection); SQL with
  `tenant_id = :tenant` **and** Postgres row-level security policies bound to a per-transaction setting, so a
  missing predicate returns nothing instead of everything.
* **Within a tenant:** optional `acl_groups` on documents, filtered the same way, for department-level access.
* **Final defence in depth:** the response builder drops (and logs as a security event) any item whose tenant
  differs from the context. The golden test already enforces this externally.
* **Aggregations, counts, filenames, classifications, RAG:** all derive from tenant-filtered queries only. Counts
  of other tenants are never computed in the same query.
* **No shared mutable per-tenant state in process memory** (the cause of the E2 leak): caches keyed by
  `(tenant_id, …)`, bounded, invalidated on ingest; no singleton index objects.
* **Consistency across stores:** `tenant_id` is written to Milvus and the registry in the same ingest step; a
  reconciliation job flags rows whose tenant differs between stores.

## 15. Evidence and provenance model

Unified result item (evaluated against the current response; existing fields stay for compatibility):

```
{
  document_id, source_file, title, doc_type,
  page_number, chunk_id, span: {start, end}, bbox?,
  snippet, highlights[],
  match_type: exact_phrase | exact_identifier | normalized_identifier | filename_exact | entity |
              structured_filter | bm25 | semantic | ocr_tolerant | fuzzy_name | visual_region,
  scores: { lexical?, semantic?, rerank?, fused },   # never compared across types
  confidence,                                         # calibrated per match type
  retrievers: [lexical, vector, structured, visual],
  metadata: {...}, evidence_ids: [...]
}
```

Answers of kind `number`/`table`/`generated_answer` return the list of evidence items that produced them. The
current response's internal `query_meta` debug field becomes an opt-in `explain` block for operators only.

## 16. Query planner design

* Interface: `plan(question, schema, locale) -> QueryPlan`; `execute(plan, auth) -> Result`.
* Rule planner first; LLM planner only when rules flag "composite/unparsed". Planner output is logged with the
  final result for evaluation.
* The plan is shown in `explain` mode ("searched for the exact phrase …, filtered doc_type = NDA, counted
  distinct documents"), which is also the basis of the Xerox demo narrative.
* Prompt-injection boundary: the planner sees only the user's question and field schema. Generation (RAG) sees
  evidence text and is instructed to treat it as data; it cannot call tools or change the plan.

## 17. Candidate fusion and ranking strategy

1. **Tiering by match type** (exact intent queries): `exact_phrase` / `exact_identifier` tier first, then
   `filename_exact` / `entity` / `structured_filter`, then everything else. Within a tier, order by fused score.
2. **Fusion within a tier: reciprocal rank fusion** (k ≈ 60) over the lexical and vector lists. RRF needs no score
   calibration, which matters because BM25, cosine and reranker scales are not comparable.
3. **Rerank** the fused top N (for example 50 passages) with a cross-encoder; the reranker may reorder within a
   tier, never move a non-exact item above an exact one for exact-intent queries. Arabic needs a multilingual
   reranker (bge-reranker-base is English/Chinese-centred; evaluate a multilingual one).
4. **No-result rule:** if no retriever passes its own calibrated floor, return `none`.
5. Weights, k values and floors are configuration, tuned only against the golden and Xerox evaluation sets.

## 18. Recommended indexes and storage

Evaluated against: current stack, tenant isolation, Arabic, structured queries and aggregation, on-prem/Saudi
deployment, operational complexity. Scratch spikes (outside the repository, nothing committed) were run for the
two credible lexical options:

| Check (spike) | Milvus 2.6.3, ICU analyzer, BM25 + `PHRASE_MATCH` + `NGRAM` | PostgreSQL 16, `tsvector` + `pg_trgm` |
|---|---|---|
| Needle phrase among 60 near-duplicates | top 1 (BM25 and `PHRASE_MATCH`) | top 1 (`phraseto_tsquery`) |
| `FOR IMMEDIATE RELEASE` vs firmware note | phrase match returns only the press release | same |
| `INV-2026-00481`, `Article 12.4`, partial `00481` | all found (`NGRAM` `LIKE`) | all found (`ILIKE` + trigram) |
| Arabic `إنهاء العقد`, `المادة 12.4` | found (ICU tokenization, **no stemming**) | found; `arabic` config stems and normalizes (`انهاء`, `عقد`, `اشعار`) |
| OCR `California` vs `Califomia` | not matched | trigram 0.55 (below default threshold) |
| Tenant-filtered search and counts | correct | correct (with `WHERE tenant`; RLS not spiked) |
| Dense + sparse fusion in one call | `hybrid_search` + RRF works; needle ranked 2nd (see section 6) | n/a (vectors stay in Milvus) |
| Aggregation (`COUNT`, `SUM`, `GROUP BY`) over extracted fields | very limited | native |

**Recommendation:**

* **Milvus stays the vector index** (no schema change in the first phases). The 2.3.1 → 2.6 upgrade is
  attractive later (BM25, phrase, n-gram, partition-key tenancy) but requires a new collection and re-indexing,
  and does not provide aggregation over structured fields.
* **Add PostgreSQL as the document registry, structured field store, lexical index and aggregation engine.**
  Justification, as required before introducing a database: the query requirements (counts, sums, dates,
  classification, clauses, signatures with provenance) need a relational store with transactions and row-level
  security regardless of which engine does BM25. One component then covers lexical (FTS with Arabic stemming and
  trigram fuzzy), structured filters, aggregation, document registry (fixing duplicates and missing page numbers),
  and audit. It runs on-prem, is widely supported in Saudi enterprise environments, and is operationally simpler
  than Elasticsearch/OpenSearch for this scale.
* **Not recommended now:** Elasticsearch/OpenSearch (a second search cluster to secure and operate without a
  requirement Postgres cannot meet at PoC scale); an LLM reranker (latency and on-prem cost; revisit after
  measurement).
* **Revisit trigger:** if a tenant exceeds what Postgres FTS serves within the latency target (measure at 1M+
  chunks), move lexical retrieval to Milvus 2.6 BM25 or a dedicated engine behind the same retriever interface.
* Redis remains for Celery and short-lived caches only.

Open points the spikes did not settle: Postgres `ts_rank` is not BM25 (acceptable for tiering + RRF, measure
quality); Milvus ICU emitted punctuation tokens and needs a filter; Arabic normalization of hamza/alef variants in
Milvus was not tested.

## 19. Performance considerations

Measured with `ultimate/scripts/forensics/scale_probe.py` against the current code (stand-in embedder; local
Milvus 2.3.1; 4 CPU cores; 20 planted exact phrases per tenant; limit 10):

| Tenant chunks | Exact-phrase recall@10 (both / vector / semantic) | Search p50 / p95 (both) | Full scan (`query_all_chunks`, cap 16,384) |
|---|---|---|---|
| ~2,400 | 20/20 · 20/20 · 20/20 | 0.055 s / 0.068 s | 2,400 rows, 0.56 s |
| ~12,000 | 4/20 · 4/20 · 4/20 | 0.070 s / 0.119 s | 12,000 rows, 4.8 s |
| ~30,000 (est.) | 4/20 · 9/20 · 14/20 | 0.053 s / 0.059 s | **16,384 rows (truncated)**, 17.3 s |

Reading: the stand-in is lexical by construction, so these recalls are an **upper bound** for MPNet. The
variance between modes at the largest size was not investigated (probably approximate search right after bulk
ingest); the conclusion does not depend on it. Exact recall from vector search fails well before enterprise scale, and the legacy content
scan cannot be the fix (17 s and truncated). Real MPNet embedding ran at about 2 chunks/s on this CPU, so
ingest capacity (GPU or batching) is a separate sizing item.

Targets (to confirm with Xerox): search p50 ≤ 300 ms and p95 ≤ 1 s at 1M chunks per tenant for `documents`
answers; aggregation p95 ≤ 2 s; planner overhead ≤ 50 ms (rules) / ≤ 1.5 s (LLM). Ingest-time classification and
extraction are asynchronous; answers report `pending` documents.

## 20. Failure modes

| Failure | Behaviour |
|---|---|
| Lexical store down | Vector-only results, response flagged `degraded: lexical_unavailable`; exact-intent queries say exact search is unavailable instead of returning near matches as if exact |
| Vector store or embedder down | Lexical/structured only, flagged |
| Reranker unavailable | Fused order, flagged (no 1.0 fallback) |
| LLM planner unavailable or invalid plan | Rule planner + hybrid retrieval |
| Classification/extraction backlog | Aggregations report `pending` counts |
| Stores disagree on a document's tenant | Item dropped, security event |
| Timeouts | Per-retriever budgets; partial results labelled |

## 21. Security risks

| Risk | Mitigation |
|---|---|
| Cross-tenant retrieval through shared in-memory state (proven in E2) | No singleton per-tenant state; tenant-keyed caches; final check |
| Missing tenant predicate in a new query | Postgres RLS; retriever signatures require AuthContext; golden tenant invariant in CI |
| Inference through counts, filenames, classifications, "did you mean" | All derived from tenant-filtered data; suggestions drawn from the tenant's own vocabulary only |
| Prompt injection from document text (OCR'd content) | Planner never sees document text; generation treats evidence as data; no tool use from generation |
| LLM data egress | On-prem model only for Saudi deployment; no external API calls with document content |
| Filter-expression injection | Typed parameters (SQL bind variables, validated Milvus values); the current raw interpolation of non-string scope values is replaced |
| Embedding inversion / vector export | Milvus not exposed; authentication enabled (Phase 8) |
| Over-broad fuzzy matching revealing near-identifiers | Fuzzy limited to long tokens, labelled, within tenant |

## 22. Xerox-specific PoC plan

1. **Data:** 200 to 500 real Xerox documents per demo tenant (English and Arabic; invoices, POs, NDAs, service
   contracts, letters; scanned and digital), plus 50 questions with expected answers written by Xerox staff.
2. **Scenarios** (each shows the plan in `explain` mode): exact identifier lookup (`INV-…`, `Contract No.`),
   Arabic clause lookup, semantic clause search, "How many NDAs" with evidence, contracts expiring in a year,
   OCR-noisy scan retrieval, tenant isolation demonstration (same question, two tenants, different counts).
3. **Out of PoC scope unless data allows:** signature detection (S6), cross-language retrieval.
4. **Deployment:** on-prem Docker stack (API, workers, embedder, Milvus, Postgres, Redis), no external calls.
5. **Exit criteria:** the metrics in section 23 on the Xerox set, tenant leakage 0.

## 23. Golden evaluation methodology

Sets:

* **Synthetic golden (CI):** `ultimate/tests/golden/cases.py`, now 86 documents, 6 tenants, 61 cases, all three
  current modes, stand-in and MPNet baselines. Hard invariant: no foreign-tenant result.
* **Xerox set (private, not in git):** real documents and questions from Xerox, stored outside the repository.
* **Scale tenant:** `scale_probe.py` at 10k, 100k and 1M chunks with planted exact phrases and identifiers.
* **Aggregation set:** questions with exact numeric answers computed from labelled data (the `[golden xdemo]`
  examples in `SEARCH_QUERY_EXAMPLES.md` are the first ones).

Metrics (per query type and overall):

| Metric | Definition | Target (initial) |
|---|---|---|
| Exact-match recall@10 | share of exact/identifier queries whose matching documents all appear in the top 10 | ≥ 0.99 |
| Top-1 accuracy | share of single-answer queries with the expected document first | ≥ 0.90 exact, ≥ 0.75 semantic |
| Top-5 recall | share of relevant documents found in the top 5 | ≥ 0.90 |
| Semantic recall@10 | graded relevance on paraphrase queries | ≥ 0.80 |
| MRR | mean reciprocal rank | tracked |
| Aggregation accuracy | exact numeric answer and correct evidence set | ≥ 0.95 |
| No-result precision | unknown queries returning `none` | ≥ 0.95 |
| Tenant leakage | foreign-tenant items, counts or evidence | **0, hard gate** |
| Latency | p50/p95 per answer kind at each scale point | section 19 |

Every change to retrieval runs the full set; baselines are re-recorded only deliberately (the existing golden test
already fails when a case starts or stops passing).

## 24. Migration strategy from current search

Superseded by `MIGRATION_PLAN.md`. Mapping from the phase labels used in this document:

| Label here | `MIGRATION_PLAN.md` |
|---|---|
| S0 (reconciliation, Xerox data, metrics) | S0 |
| S1 (AuthContext, page numbers, registry, dual write, normalized fields) | S1, S2, S3, S4 |
| S2 (new retrieval package, rule planner, exact tier, shadow mode) | S5, S6, S7, S8 (shadow comparison: S13) |
| S3 (reranker, OCR-tolerant matching, Arabic analyzer, no-result floor, opt-in cut-over) | S5 (analyzer, OCR fold), S9 (reranker, floors), S14 (cut-over) |
| S4 (classification, extraction, structured retriever, aggregation) | S10 |
| S5 (clauses, RAG) | S12 |
| S6 (visual regions, LLM planner, optional Milvus 2.6) | S11, S7 (LLM planner), later (Milvus upgrade) |
| S7 (remove legacy search code) | S15 |

Not done at any point without separate approval: restoring the legacy names, changing the existing collection
schema or embedding model in place, mixing tenants in any shared index.
