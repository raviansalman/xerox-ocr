# Migration plan: v1 to v2

Every phase is shippable on its own. Each one keeps the v1 test suite green, adds its own tests, and ends with an
evaluation report against the previous phase. New behaviour sits behind configuration flags until its report
shows it is better, and v1 code paths are removed only after their replacement has been the default for one
release. Durations assume one to two engineers and depend on access to real data (D4).

## Phase 0: measurement first (1 to 2 weeks)

* Build `evaluation/` and the `docintel-eval` runner (EVALUATION_FRAMEWORK.md).
* Collect and label the first real domain sets (D4).
* Add the exact-retrieval and isolation invariants to CI.
* Record the v1 baseline on fixtures, real sets and scale corpora.

**Exit:** a baseline report per domain; known gaps ranked by their effect on the metrics.

## Phase 1: hygiene and foundations (1 to 2 weeks)

* Remove the v1 hardcoded-data findings (DOMAIN_PACKS.md). Add the CI scanner and settings without production
  defaults.
* Decide D1 and implement the chosen lexical access path, so lexical search uses its indexes at scale.
* Add metrics and tracing per stage (ingestion and query), and dashboards.
* Model registry as data (today's `models.yaml`), with thresholds per model.

**Exit:** scanner clean; latency at 100 thousand documents within target; no change in result quality.

## Phase 2: module boundaries (2 weeks, no behaviour change)

* Move the code into the target package layout (TARGET_ARCHITECTURE.md), with v1 import paths re-exported.
* Introduce the `Retriever` and `Candidate` contracts and wrap the existing lexical, vector and structured code
  as retrievers.
* Separate fusion from retrieval, and evidence verification from ranking.

**Exit:** identical results on every evaluation question (bit-for-bit result lists), the same latency.

## Phase 3: canonical model v2 (3 to 4 weeks)

* Add document versions, blocks with bounding boxes and reading order, tables and cells, images, sections, spans
  (additive tables).
* Parsers emit blocks and tables: PDF native layout, DOCX structure, spreadsheets as tables, slides and e-mails
  as blocks. OCR emits lines and blocks.
* Layout analysis for scanned pages (a layout model behind a flag, measured against the heuristic).
* Re-anchor fields and entities to spans. Units become passages, sections, table rows and documents.
* Background reprocessing of existing documents, version by version.

**Exit:** canonical invariants pass on every fixture; evidence spans on every result; no regression.

## Phase 4: retrieval quality (3 to 4 weeks)

* Lexical: field boosts, a tenant vocabulary for typo correction (replaces the slow trigram fallback), prefix and
  suffix matching.
* Semantic: multi-granularity vectors with context headers; evaluate stronger embedding models from the registry
  (general and domain-specific) on the domain sets.
* Reranker service (cross-encoder) with calibrated confidence.
* Evidence verification and abstention with tuned thresholds.

**Exit:** Recall@10 and MRR up on every domain set; unanswerable false positives within budget; latency within
target.

## Phase 5: entities, relations and contextual retrieval (3 to 4 weeks)

* Generic statistical or model-based named-entity recognition, alongside the rules. Entity resolution within
  documents, and links across documents.
* Relation extraction (patterns and models) into the relations table. Packs define predicates.
* Contextual retriever: concept expansion, relation retrieval, section-aware matching.
* Planner v2 intents: `find_concept`, `find_entity`, `fact` over relations; the plan schema v2 with validation.

**Exit:** concept and relation question types reach their targets; no regression elsewhere.

## Phase 6: domain packs (2 weeks per pack, after the platform phases)

* Convert v1 resources into the `business`, `legal` and `finance` packs.
* Build the first two new packs for the domains with real data (D5), each with fixtures, labels and an
  evaluation report.
* Per-tenant enablement and field access tags.

**Exit:** each pack meets its per-domain targets on its own evaluation set.

## Phase 7: answers and scale-out (3 to 4 weeks, optional parts per D2 and D3)

* Answer service: grounded answers from verified evidence, citation checking (every claim must map to a span),
  refusal when evidence is insufficient. Model-assisted planning for low-confidence questions.
* Document comparison (`compare` intent) over aligned sections.
* Split OCR, indexing, embedding and reranking into separately scaled workers and services; durable queue;
  outbox and reconciliation.
* OpenSearch lexical backend if D3's criteria are met.

**Exit:** answer faithfulness 95% or better on labelled questions (judged against evidence); scale targets met at
1 million documents.

## Risks and how they are handled

| Risk | Handling |
|---|---|
| No real labelled data | Phase 0 is a hard prerequisite. Without it, quality claims stay limited to synthetic tests, and the report says so. |
| Model quality varies by domain | Every model sits behind the registry and is chosen per domain by evaluation, not by reputation |
| Reprocessing large corpora | Versioned, background, resumable; the old version serves until the new one is complete |
| Latency growth from more retrievers | Parallel execution with budgets; the planner runs only the strategies a question needs; reranking limited to the top 50 |
| Security regressions | Isolation tests on every query type in CI; any change to how isolation is enforced needs explicit approval (D1) |
| Scope creep | Each phase has exit criteria; work outside them goes to the next phase's backlog |
