# Evaluation framework

Nothing in v2 is called better until this framework shows it. Every phase in MIGRATION_PLAN.md ends with an
evaluation report compared against the v1 baseline.

## Datasets

| Set | Content | Where | Used for |
|---|---|---|---|
| Fixtures | the synthetic v1 corpus (about 80 documents, 3 tenants) and per-pack synthetic documents | `tests/fixtures/` | CI on every change |
| Invariants | generated: every identifier, phrase and number in the fixtures, queried in several spellings | generated in CI | exact-retrieval guarantees |
| Domain sets | 100 to 300 real, representative documents per domain, with 30 to 50 labelled questions each (decision D4) | `evaluation/<domain>/`, access-controlled, never in production code or images | quality per domain |
| Unanswerable | questions whose answer is not in the corpus, per domain (at least 20%) | with each domain set | false positives and abstention |
| Scale | generated corpora of 10 thousand, 100 thousand and 1 million documents | generated on demand | latency, throughput, index size |
| Isolation | every question asked as every tenant of a multi-tenant corpus | fixtures and domain sets | leakage (must be zero) |

### Label format

```yaml
- id: legal-017
  question: Can either party cancel the agreement early?
  type: find_concept                   # matches planner intents
  relevant:                            # graded relevance; spans make evidence checkable
    - doc: msa_2024_acme.pdf
      grade: 3
      evidence: [{page: 12, text: "Either party may terminate this Agreement upon thirty (30) days"}]
    - doc: nda_globex.pdf
      grade: 1
  answer: null                         # for computed or fact questions: the expected value
  unanswerable: false
```

Labels are written by domain experts, reviewed by a second person, and versioned with the dataset.

## Metrics

| Area | Metric | Gate |
|---|---|---|
| Exact retrieval | recall of exact phrases and identifiers present in the corpus, in any supported spelling | **100%** (invariant) |
| Exact retrieval | does exact recall stay the same with the vector index disabled? | **yes** (invariant) |
| Ranking | Recall@10, MRR, nDCG@10 per query type and domain | no regression against the previous release; targets per domain set on the first baseline |
| Precision | Precision@5; share of top-5 results with relevance 0 | target 0.9 or better on exact and lookup types; tuned per type |
| Abstention | false-positive rate on unanswerable questions (any result in the top 3 above 0.5 confidence) | target 10% or less; reported with the recall trade-off |
| Evidence | share of results whose evidence span contains the matched text or supports the label | 100% for exact and lexical; 95% or better for semantic |
| Computed answers | exact match of counts, sums and lookups against labels | 100% on fixtures; 98% or better on domain sets (errors traced to extraction) |
| Extraction | precision and recall per field and entity type against labelled spans | per pack, set on the first baseline |
| Confidence | calibration error (expected vs observed precision by confidence bucket) | 0.05 or less |
| Isolation | foreign documents, counts or evidence in any response | **0** |
| Latency | p50 and p95 per query type, at 10 thousand, 100 thousand and 1 million documents | p95 under 500 ms interactive (TARGET_ARCHITECTURE.md) |
| Ingestion | documents and pages per minute per worker, and per stage | reported; no regression of more than 10% |

## Runner

```
docintel-eval run --dataset evaluation/legal --engine http://... --key ... --out reports/2026-11-01-legal.json
docintel-eval compare reports/baseline.json reports/candidate.json
```

* Runs every question with `explain=true`, stores plans, candidates per retriever, results, timings and
  evidence.
* Computes the metrics above, per query type, domain and retriever, plus an ablation view: the metric with each
  retriever disabled, which shows what each one contributes.
* Writes a JSON report and a readable summary; `compare` highlights regressions per question.
* CI runs fixtures, invariants and isolation on every change. Domain sets run nightly and before every release
  (they need access to the restricted data).

## Baseline (v1, measured)

| Measure | v1 |
|---|---|
| Fixture questions (retrieval, counts, filters, lookups, facts) | all pass (`tests/integration/test_search.py`) |
| Semantic paraphrases, real model, 10 test questions | 7 of 10 in the top 3; misses: press-contact question (not found), early termination (rank 4), repair visit (rank 10) |
| Unrelated questions | 0 results for all 5 |
| Tenant leakage | 0 across 3 tenants and all fixture questions |
| Latency, 2,000 documents, 4 cores, 8 concurrent users | p50 240 to 330 ms search, 60 to 90 ms structured |
| Latency, 50,000 documents, one user | 170 to 230 ms search; typo fallback about 0.8 s |
| Ingestion, 4 cores, 10% scanned | about 512 documents per minute |

The first task of phase 0 is to measure the same on real domain sets, because the synthetic numbers say little
about real documents.
