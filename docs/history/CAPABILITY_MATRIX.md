# Capability matrix: current vs target

Status of every capability **today** (branch `claude/inspiring-bohr-3kphnv`), with the evidence behind it, and
the target from `DOCUMENT_INTELLIGENCE_ARCHITECTURE.md`. Statuses:

* **WORKING**: verified by an executed test or measurement
* **PARTIAL**: works in a limited or degraded form (limitation stated)
* **BROKEN**: implemented and reachable, but wrong
* **MISSING**: not implemented
* **LEGACY**: code exists but is dead or unsafe, and is not part of the target (forensic evidence only)
* **UNKNOWN**: code exists but has never been exercised by a test, or depends on production we cannot see

Evidence references: KD-* in `ASSESSMENT.md` appendix A, `SEARCH_FORENSICS.md` (SF), tests in `ultimate/tests`.
Phase = first migration phase that delivers the target (`MIGRATION_PLAN.md`).

## Ingestion and storage

| Capability | Current | Evidence | Target | Phase |
|---|---|---|---|---|
| Upload API, URL ingest with SSRF guard and size cap | WORKING | `test_input_safety.py`, `test_api_contract.py` | Same, plus streaming to object storage | S3 |
| Preserve original document | MISSING | Uploads and downloads deleted after processing (`ultimate_ui.py:2524`, `ultimate_tasks.py:1023`) | Content-addressed object storage, never deleted by processing | S3 |
| Content hash, idempotent re-ingest | MISSING | Re-ingest duplicates chunks (KD-DATA-02) | sha256 per version, replace semantics | S3/S4 |
| Document registry (identity, status, versions) | MISSING | Only Milvus chunks and Celery task state | Postgres `documents` / `document_versions` | S4 |
| Text PDF | WORKING | `test_text_pdf_extraction` | Same, with layout | S3 |
| Scanned PDF / image OCR (English) | WORKING | `test_scanned_image_ocr_english`; 14/14 key terms (ASSESSMENT B) | Same, with word boxes and confidence | S3 |
| Mixed PDF (digital + scanned pages) | BROKEN | Scanned appendix skipped (KD-OCR-05) | Per-page native/scanned decision | S3 |
| Digital/scanned/mixed/image classification of the file | PARTIAL | 30-words-per-page heuristic for the whole PDF | Per page, stored | S3 |
| DOCX (with tables as text) | WORKING | `test_docx_extraction_includes_tables` | Structure-preserving (tables as tables) | S3 |
| TXT, HTML | WORKING | `test_plain_text_and_html_are_read_verbatim` | Same | S3 |
| DOC, PPT/PPTX, XLS/XLSX/ODS, CSV, RTF, MD | UNKNOWN | Extractors exist (`ultimate_search_processor.py:1993-2600`), no tests | Tested per format | S3 |
| TIFF (multi-page), WEBP, GIF, BMP | UNKNOWN | Mapped in the type table, untested | Tested | S3 |
| Email (EML/MSG with attachments) | MISSING | Not in the supported list | Optional extractor | S3+ |
| Page numbers per chunk | BROKEN | Always 0 (KD-OCR-02, strict xfail) | Page, block, char offsets | S2/S3 |
| Page order | BROKEN | Thread completion order (KD-OCR-03) | Deterministic | S3 |
| Failed extraction surfaced | BROKEN | Indexed as success (KD-OCR-04); duplicate ingest reported as success (KD-OPS-07) | Explicit failed/partial/skipped states | S3 |

## OCR and document understanding

| Capability | Current | Evidence | Target | Phase |
|---|---|---|---|---|
| OCR provider abstraction | MISSING | Tesseract calls inline | `OcrEngine` interface, versioned results | S3 |
| Arabic OCR in the pipeline | BROKEN | 0 Arabic characters through the pipeline; Tesseract `ara` works directly (KD-OCR-06) | Language-aware OCR | S3 |
| Language detection | MISSING | Single configured language | Per page/block | S3 |
| OCR confidence | BROKEN | Not a real confidence (KD-OCR-08) | Word/line confidence from the engine | S3 |
| Word/region coordinates | MISSING | `image_to_string` only | Boxes for words and blocks | S3 |
| OCR text cleaning | BROKEN | Corrupts uppercase words (KD-OCR-01); `_normalize_text` corrupts text (KD-OCR-07) | Raw text kept; normalization only in derived fields | S2/S3 |
| Layout: blocks, headings, reading order | MISSING | Text flattened | Canonical model blocks | S3 |
| Table extraction (structured) | MISSING | Text only | Table cells with coordinates | S3+ |
| Handwriting recognition | MISSING | | Pluggable capability, evaluated before claimed | S11 |
| Image captioning (BLIP) | UNKNOWN | Code exists, disabled in tests (`SKIP_IMAGE_CAPTIONING_IN_PROCESSOR`) | Optional plugin | S11 |
| PII redaction (README claims Presidio) | MISSING | No Presidio code in the repository | Optional policy-driven plugin | later |
| Document classification | MISSING | | Configured taxonomy, confidence, version | S4+ |
| Entity extraction persisted (people, orgs, places) | BROKEN | Computed at ingest, discarded (KD-DATA-01) | `entities` with provenance | S4+ |
| Dates, amounts, identifiers persisted | BROKEN | Same | `fields` with provenance | S4+ |
| Clause extraction | LEGACY | Regex clause detection only inside dead query code | Clause segmentation and typing | S12 |
| Signature / stamp / checkbox detection | MISSING | | Visual region plugins | S11 |
| Metadata index (`MetadataIndex`) | LEGACY | Cannot be constructed; restoring it leaked across tenants (SF E2) | Replaced by the structured store | S15 (delete) |

## Indexing and retrieval

| Capability | Current | Evidence | Target | Phase |
|---|---|---|---|---|
| Vector index (MPNet 768, COSINE, HNSW) | WORKING | Golden MPNet baseline | Same, configurable k, model registry | S6 |
| Vector candidate count | PARTIAL | Hard cap 50 chunks (SF) | Configurable, measured | S6 |
| Lexical index over content | MISSING | Lexical scan matches filenames only | Postgres FTS (phrase, Arabic stemming, trigram) | S5 |
| Exact phrase / identifier retrieval | BROKEN | Re-ordering among ≤ 50 vector hits only; needle 61/61 (SF) | Exact tier independent of vectors | S5/S8 |
| Exact recall at scale | BROKEN | 20/20 at ~2.4k chunks, 4/20 at ~12k (scale probe, stand-in upper bound) | ≥ 0.99 at 1M chunks | S8 |
| Filename search | PARTIAL | Lexical tiers only as zero-result fallback | Structured field, exact and token | S5 |
| Normalization (case, Unicode, punctuation) | PARTIAL | Lowercase and whitespace only | One analyzer for ingest and query | S5 |
| Joined/split forms, ordered words, proximity | MISSING | SF section 6 | Analyzer variants, phrase slop | S5 |
| OCR-tolerant matching | MISSING | No live fuzzy code | OCR-folded field + bounded trigram, labelled | S5/S8 |
| Semantic retrieval | WORKING | Golden: paraphrase cases pass on small corpora | Same, plus multilingual decision | S6 |
| Multilingual / cross-language semantic | UNKNOWN | English-centric model; untested on real Arabic | Decided on Xerox data | S6 |
| Hybrid fusion | MISSING | `both` mode runs vector only (KD-SRCH-13) | Tiered RRF over independent candidates | S8 |
| Reranking | PARTIAL | Semantic mode only; 1.0 fallback (KD-SRCH-08); production reranker not reproducible here | Health-reported cross-encoder, multilingual option | S9 |
| Structured filters (type, date, amount, party, jurisdiction) | MISSING | | SQL filters in every retriever | S4/S8 |
| Legacy query understanding (`enhance_query`) | LEGACY | ImportError on every call (KD-SRCH-11) | Replaced by the query planner | S15 (delete) |
| Legacy supplements (content scan, NDA, entity) | LEGACY | Dead; unsafe if revived (SF E1) | Replaced | S15 (delete) |
| Search modes chosen by the user (`searchMethod`) | PARTIAL | Exposed in the API | Planner decides; parameter kept for compatibility | S7 |

## Query understanding, answers and evidence

| Capability | Current | Evidence | Target | Phase |
|---|---|---|---|---|
| Query planner (typed plans) | MISSING | | Rule planner + optional LLM planner, validated | S7 |
| Aggregation (count, sum, avg, min, max, %, group by) | MISSING | | SQL executor with evidence ids | S10 |
| Multi-condition queries | MISSING | | Composite plans | S7/S10 |
| RAG / generated answers | MISSING | | Evidence-grounded, citations, FACT/INFERENCE/UNCERTAIN | S12 |
| Evidence: document, chunk text | PARTIAL | Response has file id and best chunk text | Page, span, bbox, match type, versions | S2/S12 |
| Match reason per result | PARTIAL | `search_method` only (vector/semantic) | `match_type` taxonomy | S8 |
| Abstention / "no results" | BROKEN | Default mode always returns the closest documents (KD-SRCH-06) | Calibrated floors, `none` answer | S8/S9 |
| Confidence | BROKEN | Scores capped and mixed across scales | Calibrated per match type | S9 |

## Security, operations and deployment

| Capability | Current | Evidence | Target | Phase |
|---|---|---|---|---|
| API-key authentication, roles, tenant from identity | WORKING | `test_auth.py` | Plus OIDC/SSO, key expiry | S1/S14 |
| Tenant isolation on live search paths | WORKING | 0 leaks across golden runs (SF section 8) | Data-layer enforcement + final check, 0 leakage gate | every phase |
| Document-level ACL within a tenant | MISSING | | ACL groups | S4 |
| Tenant fallback `default_user` for rows without tenant | BROKEN | `vector_db_milvus_server.py:404` | Fail closed | S1 |
| Redis authentication, no pickle | WORKING | `test_redis_and_serialization.py` | Same | done |
| Milvus authentication, TLS | MISSING | SECURITY.md section 8 | Enabled | S14 |
| Audit log (who searched/ingested/deleted) | MISSING | | Append-only audit table | S4+ |
| Health: liveness / readiness | WORKING | `test_health.py`, `test_operational.py` | Same, extended to new stores | done |
| Search isolation from the event loop | WORKING | `test_search_isolation.py` | Per-request context, concurrent searches | S2 |
| Structured logging, request/query/document ids | MISSING | `logging.basicConfig` text logs | JSON logs, correlation ids | S1 |
| Metrics and tracing (latency per stage, model versions) | MISSING | | Prometheus/OpenTelemetry | S1+ |
| Immutable deployments | BROKEN | Source bind mounts in compose; rsync deploys (FORENSICS F3) | Versioned images, no bind mounts | S1 |
| Configuration | PARTIAL | 95 env vars with scattered defaults, 2 conflicting (`HARDCODED_DATA_AUDIT.md`) | Typed settings, one source | S1 |
| Hardcoded corpus/customer data | BROKEN | `HARDCODED_DATA_AUDIT.md` | Zero in production code | S1/S15 |
| Production code known | UNKNOWN | Deployed tree not provided (SF section 1) | Reconciled | S0 |
