# Evaluation strategy

Status: design for review. Existing assets: the golden search suite (`ultimate/tests/golden`, 86 documents,
6 tenants, 61 cases, three search modes, stand-in and MPNet baselines, hard no-foreign-tenant invariant), the
scale probe (`ultimate/scripts/forensics/scale_probe.py`), and 178 unit / 205 integration tests with strict-xfail
pins for known defects.

Principle: every capability is claimed only with a measurement, and every change is compared with the recorded
baseline before it ships.

## 1. Datasets

| Set | Content | Where | Used for |
|---|---|---|---|
| Synthetic golden | Generated documents and tenants, including OCR scans, Arabic, identifiers, near-duplicates, cross-tenant traps | `tests/golden` (in git) | CI regression on every change |
| Customer evaluation set (for example Xerox) | 200 to 500 real documents per demo tenant, 50+ questions with expected answers written by the customer | **outside git**, encrypted storage, access-controlled | Acceptance and model choices |
| Labelled OCR pages | Page images with ground-truth text (English, Arabic, mixed, low quality, handwriting) | outside git | CER/WER, engine comparison |
| Labelled extraction set | Documents with expected fields, entities, classes, clauses, signature regions | outside git (synthetic subset in git) | Field/class/region precision and recall |
| Scale tenants | Generated at 10k, 100k, 1M chunks with planted exact phrases and identifiers | generated on demand | Recall and latency at scale |
| Adversarial set | Cross-tenant probes, injection strings, prompt injection inside documents, malformed files | `tests/` (in git) | Security gates |

Customer data never enters the repository, CI logs or model training without written consent.

## 2. Metrics

Retrieval (per query type: exact, identifier, lexical, semantic, entity, filename, OCR-noisy, Arabic, mixed):

| Metric | Definition |
|---|---|
| Recall@K (K = 1, 5, 10, 50) | share of relevant documents in the top K |
| Precision@K | share of the top K that is relevant |
| Exact recall@10 | for exact/identifier queries, all documents containing the exact string appear in the top 10 |
| MRR | mean reciprocal rank of the first relevant document |
| nDCG@10 | graded relevance (0 to 3) |
| Top-1 accuracy | single-answer queries with the expected document first |
| False-positive rate | share of returned items judged not relevant, per match type (weak fuzzy matches tracked separately) |
| No-result precision | unknown/unanswerable queries that return `none` |

Answers and structure:

| Metric | Definition |
|---|---|
| Aggregation accuracy | exact numeric answer and identical evidence set |
| Answer accuracy (RAG) | judged correct and fully supported by cited evidence |
| Citation precision | cited spans that exist and support the claim |
| Abstention accuracy | abstains when evidence is insufficient, answers when it is sufficient |
| Planner accuracy | plan equals the expected plan (fields, operators, values) |
| Classification | per-class precision/recall/F1, calibration of confidence |
| Extraction | per-field precision/recall, value accuracy after normalization, provenance accuracy (right page/span) |
| OCR | character and word error rate per language and quality band; layout reading-order accuracy |
| Visual regions | precision/recall at IoU ≥ 0.5 per region type; signer association accuracy |

Operations: p50/p95/p99 latency per answer kind and per stage, ingest throughput, error rate, cost per 1,000
pages (CPU/GPU time).

Security: **tenant leakage count (hard gate 0)** across retrieval, aggregation, suggestions, filenames, RAG,
exports; authorization bypass tests; injection tests.

## 3. Initial targets (to confirm with the customer)

| Metric | Target |
|---|---|
| Exact recall@10 | ≥ 0.99 (at 1M chunks per tenant) |
| Top-1 accuracy | ≥ 0.90 exact/identifier, ≥ 0.75 semantic |
| Recall@5 | ≥ 0.90 |
| Semantic recall@10 | ≥ 0.80 |
| False-positive rate in top 5 (exact/identifier queries) | ≤ 0.05 |
| Aggregation accuracy | ≥ 0.95 |
| Citation precision | ≥ 0.98 |
| Abstention accuracy | ≥ 0.90 |
| No-result precision | ≥ 0.95 |
| Classification macro-F1 | ≥ 0.90 on the customer taxonomy |
| Key field extraction F1 | ≥ 0.90 (per field reported) |
| OCR CER (clean scans, English / Arabic) | ≤ 2% / ≤ 5% |
| Latency `documents` p95 | ≤ 1 s at 1M chunks; aggregation p95 ≤ 2 s |
| Tenant leakage | **0** |

## 4. Gates

| Gate | When | Blocks |
|---|---|---|
| Unit + integration + security tests | every commit | merge |
| Golden pass/fail map unchanged (or deliberately re-recorded with review) | every commit | merge |
| Tenant leakage = 0 | every commit and release | merge, release |
| Customer set metrics ≥ targets, no regression > 2 points on any query type | before cut-over (S13/S14) | cut-over |
| Scale probe at the agreed size within latency targets | before release | release |

## 5. Procedure

* **Baselines first.** Every phase records the current numbers before changing behaviour (as done for search:
  stand-in and MPNet baselines).
* **Shadow comparison** (S13): new and old paths run side by side on the same queries (golden, customer set,
  logged production queries where permitted); differences are reviewed per query type, not only in aggregate.
* **Judging:** relevance labels by customer subject-matter experts; LLM-as-judge only as a pre-filter with human
  confirmation on a sample, never as the sole judge for acceptance.
* **Reports:** per run, a machine-readable result (like `.pytest_cache/golden_<embedder>.json`) and a summary table
  by query type with deltas against the baseline; stored with the commit id and model/config versions.

## 6. Extending the golden suite

The current harness grades retrieval for `/search`. It is extended in this order:

1. Add the `[golden xdemo]` aggregation, classification and date/amount questions from
   `SEARCH_QUERY_EXAMPLES.md` as **planned** cases (expected answers recorded, marked not-yet-runnable) so the new
   engine is graded against them from its first day.
2. Planner tests (question → plan).
3. Extraction and classification fixtures with expected fields and provenance.
4. Scale tenant run in a nightly job (not per commit).
5. Customer set runner that reads the encrypted dataset from a configured location and never writes document text
   into reports.
