# Document Intelligence Engine v2: design

Status: **design for review. No code in this set is implemented yet.** The working engine described in
[../ARCHITECTURE.md](../ARCHITECTURE.md) (v1) stays the production baseline, and every change below is delivered
in increments behind it, each one measured against it.

## The requirement

Build a domain-agnostic engine that ingests every supported format in full. It preserves the original file and
its structure, and extracts text, layout, tables, entities, metadata, fields and relationships with provenance.
From these it builds independent lexical, semantic, structured and contextual representations.

The query engine decides how to search. It combines exact, lexical, semantic, contextual, entity, metadata and
structured retrieval, and returns evidence. Four rules apply to every answer:

* exact text is always retrievable without embeddings;
* paraphrased questions still find their information;
* structured questions are computed deterministically;
* every result is grounded in source evidence and the caller's tenant.

The engine must also be configurable per domain, observable, reproducible, and free of hardcoded customer,
domain or sample data.

## Documents

| # | Document | Answers |
|---|---|---|
| 1 | [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md) | Layers, module boundaries, how today's code maps onto them, how it is deployed and scaled |
| 2 | [CANONICAL_MODEL.md](CANONICAL_MODEL.md) | What "fully ingested" means: the document model, provenance, versioning, storage schema |
| 3 | [RETRIEVAL_CONTRACTS.md](RETRIEVAL_CONTRACTS.md) | The retriever interface, the eight retrievers, fusion and tiers, reranking, evidence verification, abstention |
| 4 | [INDEXING_STRATEGY.md](INDEXING_STRATEGY.md) | The six representations of every document, which store holds each, and how they scale and stay consistent |
| 5 | [QUERY_PLANNER.md](QUERY_PLANNER.md) | The query plan contract, how strategies are chosen, deterministic and model-assisted planning, validation |
| 6 | [DOMAIN_PACKS.md](DOMAIN_PACKS.md) | Generic core, pluggable domain configuration (legal, finance, medical, supply chain, HR), no hardcoded data |
| 7 | [EVALUATION_FRAMEWORK.md](EVALUATION_FRAMEWORK.md) | Datasets, metrics, quality gates and how every change is measured |
| 8 | [MIGRATION_PLAN.md](MIGRATION_PLAN.md) | Phases from v1 to v2, each shippable, with entry and exit criteria |

## Principles

1. **Representations, not one string.** Every piece of information gets a durable, addressable form with
   provenance (document, version, page, block, character span, coordinates, extractor). Retrieval mechanisms are
   built over those forms; none of them owns the information.
2. **Exact retrieval never depends on embeddings.** Lexical and structured retrieval are complete systems.
   Semantic retrieval adds candidates; it never gates or removes an exact match. This is tested as an invariant.
3. **Evidence or abstain.** A result is returned only with the span that supports it, re-checked against the
   source. When nothing clears the bar the engine says so ("no sufficiently reliable match") instead of showing
   loosely related documents.
4. **Deterministic where it can be.** Counts, sums, filters and lookups are computed from extracted data, never
   generated. Language models may help to understand a question or to phrase an answer, always under a schema
   and with citations that are verified.
5. **Generic core, domain packs on top.** The core understands documents in general: blocks, tables, dates,
   amounts, identifiers, people, organizations, places, relations. Contracts, invoices, patients or shipments are
   configuration and optional models, loaded per tenant.
6. **Measured, not asserted.** Nothing is called better until the evaluation framework shows it, per retrieval
   type and per domain, with no regression on the v1 suite and zero cross-tenant leakage.
7. **Modular monolith first.** Clear module contracts inside one codebase; only modules with a different scaling
   profile (OCR, embedding, indexing, answer generation) run as separate workers or services.
8. **Reproducible.** Every derived artifact records the parser, model and configuration versions that produced
   it, so the same input and versions give the same output, and a version change can be re-run selectively.

## What will not be promised

No search engine has zero false positives and zero misses on every question. The target is high precision, high
recall and correct abstention, measured per query type, with explicit thresholds (see EVALUATION_FRAMEWORK.md).

## Decisions needed before implementation

| # | Decision | Recommendation |
|---|---|---|
| D1 | Database roles for index-backed lexical search under row-level security (RETRIEVAL_CONTRACTS.md, INDEXING_STRATEGY.md) | Owner role plus non-owner application role, with tenant-enforcing `SECURITY DEFINER` lookup functions. This requires your approval because it changes a security control. |
| D2 | Use of a language model (query understanding fallback, answer composition, contextual enrichment) | Optional and configurable per tenant, self-hosted or API. The engine must work fully without it. |
| D3 | Lexical engine at large scale | PostgreSQL until roughly 5 to 10 million blocks per deployment, then OpenSearch behind the same contract (criteria in INDEXING_STRATEGY.md) |
| D4 | Real evaluation data | 100 to 300 representative documents per domain with 30 to 50 labelled questions each, stored outside production code |
| D5 | First domain packs | Business (exists), then the two domains with real data available first |

## Implementation status

Implemented in v2.0.0. Where the code departs from these documents (for example the B-tree postings table that
keeps lexical search indexed under row-level security, or the reranker left off by default after measurement),
[../ARCHITECTURE.md](../ARCHITECTURE.md) describes what was built and why; measured results are in
[../EVALUATION.md](../EVALUATION.md) and [../OPERATIONS.md](../OPERATIONS.md#performance).
