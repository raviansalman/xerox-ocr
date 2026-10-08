# Evaluation

Quality claims about the engine are made only from measurements. The harness in `docintel/evaluation` runs a
labelled dataset through the HTTP API and reports retrieval, answer, abstention and isolation metrics; CI runs it on
every change against the synthetic fixtures.

## What exists today

| Set | Where | Content | Status |
|---|---|---|---|
| Fixtures | `tests/fixtures/golden/fixtures.yaml` over `tests/fixtures/corpus.py` | 77 labelled questions over 83 synthetic documents in 3 tenants: identifiers, exact phrases, keywords, typos and OCR errors, paraphrases, concepts, entities, filters, computed answers, unanswerable questions | runs in CI, for v1 and v2 |
| Invariants | `tests/integration/test_retrieval_v2.py` | generated from the corpus: every extracted identifier in four spellings, quoted phrases from every document, evidence equal to the stored source text | runs in CI |
| Adversarial | `tests/integration/test_security_adversarial.py`, `test_answering.py` | injection, tenant and path manipulation, hostile files, prompt injection in documents | runs in CI |
| Real documents | not in the repository | none yet | **to do**: see below |

The fixture numbers detect regressions. They do not measure quality on real documents: the synthetic corpus was
written alongside the engine, so it cannot surprise it the way real documents will.

## Dataset format

```yaml
dataset: legal-2026
source: real                 # synthetic | real (reports say which)
tenant: tenant-a             # default tenant for questions
k: 10
gates:                       # minimum, or maximum for *_rate, leakage and latency
  exact_recall: 1.0
  computed_accuracy: 1.0
  abstention_fp_rate: 0.1
  leakage: 0
  mrr: 0.8
questions:
  - id: leg-001
    category: contextual     # free-form; metrics are reported per category
    question: Can either party cancel the agreement early?
    relevant: {msa_2024.pdf: 3, nda_globex.pdf: 1}       # file name -> grade 1 to 3
  - id: leg-002
    category: identifier
    exact: true              # every relevant document must be found (exact-recall gate)
    question: MSA-2024-117
    relevant: {msa_2024.pdf: 3}
  - id: leg-003
    category: computed
    question: How many contracts expire in 2027?
    answer: {value: 4}       # every key must equal the response's answer (rows compare as sets)
  - id: leg-004
    category: unanswerable
    question: What is the parking policy?
    unanswerable: true       # any result is a false positive
  - id: leg-005
    requires: [semantic]     # skipped when the target runs without semantic search
    question: rules about working from home
    relevant: {remote_work_policy.docx: 3}
```

## Metrics

| Metric | Definition |
|---|---|
| `precision_at_5` | share of the returned top-5 results that are relevant (questions with results) |
| `recall_at_k`, `mrr`, `ndcg_at_k` | standard definitions over file names with graded relevance |
| `exact_recall` | recall over questions marked `exact` (identifiers, quoted phrases, filters) |
| `computed_accuracy` | share of computed questions whose answer matches the label exactly |
| `abstention_fp_rate` | share of unanswerable questions that returned any result |
| `leakage` | foreign documents in any response; every question is also asked as every other tenant of the dataset |
| `p50_ms`, `p95_ms` | server-side query time |

Every metric is also reported per category; failed questions are listed with what came back.

## Running it

```bash
# keys.json: {"tenant-a": "<reader key>", "tenant-b": "<reader key>"}
docintel eval run --dataset evaluation/legal.yaml --url http://localhost:8000 --keys keys.json --out reports/legal.json
docintel eval compare reports/baseline.json reports/legal.json      # metric deltas and per-question regressions
```

`run` exits non-zero when a gate fails; `compare` exits non-zero when a metric regresses.

## Measured on the fixtures

80 questions (73 when semantic search is off: the 7 paraphrase questions are skipped), measured with
`tests/integration/test_evaluation.py` and `DOCINTEL_EVAL_REPORT_DIR` set to keep the reports.

| | v1, real model | **v2, real model** | v1, semantic off | **v2, semantic off** |
|---|---|---|---|---|
| Questions passed | 77 of 80 | **80 of 80** | 69 of 73 | **73 of 73** |
| MRR | 0.984 | **1.0** | 0.946 | **1.0** |
| nDCG@10 | 0.983 | **0.999** | 0.945 | **0.999** |
| Recall@10 | 0.976 | **0.992** | 0.938 | **0.991** |
| Precision@5 (of returned results) | 0.818 | 0.821 | 0.987 | 0.947 |
| Exact recall | 1.0 | 1.0 | 1.0 | 1.0 |
| Computed answers correct | 10 of 11 | **11 of 11** | 10 of 11 | **11 of 11** |
| False positives on 7 unanswerable questions | 1 | **0** | 0 | 0 |
| Cross-tenant results (160 isolation queries for v2) | 0 | 0 | 0 | 0 |

v1 misses: a cancellation question that shares no words with the clause, an entity question without semantic
search, a lookup that also returned another invoice's number, a figure stated in a relation, a two-word typo, and a
question about another company's merger answered with confidentiality clauses. v2's lower precision with semantic
search off is the price of returning related documents (tolerant and concept matches) below the relevant ones.

The cross-encoder reranker, measured on the same set with the real models, reordered five rankings below the
relevant documents and changed no metric, so it stays off by default.

## Next: real documents

Real documents are the only meaningful measure. For each domain, collect 100 to 300 representative documents,
write 30 to 50 questions with graded labels (at least 20% unanswerable), have a second person review the labels,
keep the set outside the repository with restricted access, and run it before every release. Expect the first run
to find problems the fixtures could not.
