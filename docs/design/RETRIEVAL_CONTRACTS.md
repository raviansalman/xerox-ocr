# Retrieval contracts

## The retriever interface

Every retrieval mechanism implements the same contract, runs independently, and can be enabled, disabled,
measured and replaced on its own.

```python
class Retriever(Protocol):
    name: str                                   # "exact", "lexical", "fuzzy", "semantic", ...
    def retrieve(self, plan: QueryPlan, ctx: AuthContext, scope: Scope, budget: Budget) -> list[Candidate]: ...

@dataclass(frozen=True)
class Scope:            # what the retriever may search
    document_ids: frozenset[str] | None         # from structured filters; None = all of the tenant's current documents
    unit_types: frozenset[str]                  # block, passage, section, table_row, document

@dataclass(frozen=True)
class Budget:
    max_candidates: int
    timeout_ms: int

@dataclass(frozen=True)
class Candidate:
    document_id: str
    version_id: str
    unit_id: str
    unit_type: str
    retriever: str                              # which retriever produced it
    match_type: str                             # exact_phrase, exact_identifier, stemmed, fuzzy, semantic, entity, ...
    raw_score: float                            # retriever-specific, never compared across retrievers
    spans: tuple[Span, ...]                     # where in the unit the evidence is (empty for pure semantic hits)
    matched: tuple[str, ...]                    # query parts this candidate satisfies (terms, identifiers, concepts)
```

Rules:

* The tenant comes only from `ctx`. A retriever cannot take a tenant argument from the plan.
* A retriever that fails or times out returns nothing and reports the failure. The query continues with the
  others, and the response states which retrievers were degraded.
* Retrievers return candidates, not documents. Grouping by document happens in ranking.

## The retrievers

| Retriever | Finds | Over | Notes |
|---|---|---|---|
| **exact** | exact phrases, exact tokens, identifiers in any spelling (punctuation, spacing, case, Unicode), file names, quoted strings, section references ("Section 8.4") | lexical index (token positions, canonical identifiers, normalized file names) | No embeddings, ever. A span is mandatory. |
| **lexical** | all or most terms, stemmed forms, prefix and suffix matches, joined and split words, BM25-style ranking | lexical index | Field boosts: title, headings and key-value labels count more than body text |
| **fuzzy** | typos, OCR confusions (rn/m, l/1, O/0), spelling variants, partial identifiers | OCR-folded and trigram representations; vocabulary-based correction (query words corrected against the tenant's own vocabulary first, then searched exactly) | Used when exact and lexical are thin, or when the query is identifier-like |
| **semantic** | paraphrases and concepts | vector index at passage, section, table-row and document granularity | Similarity floor per model, from calibration. Never removes an exact candidate. |
| **contextual** | meaning that depends on how the document is organised | concept expansion from packs (terminate ≈ cancel ≈ end), section-aware vectors, relation index | See below |
| **entity** | documents and spans mentioning a person, organization, place or identifier, including aliases | mentions, entities, entity links | "Acme" finds "ACME Corp." and "Acme Corporation Ltd" |
| **metadata** | file name, type, source, collection, dates of upload, authorship, language | documents, versions | Filters and ranking signals |
| **structured** | typed values and fields matching conditions (dates, amounts, quantities, fields from packs, table cells) | values, fields, table cells | Exact computation. Also feeds aggregation. |

### Contextual retrieval

Contextual retrieval answers "Can either party cancel the contract early?" when the document says "The customer
may terminate this agreement upon thirty days written notice." Three mechanisms, each measurable:

1. **Concept expansion.** Packs define concepts with lexical variants and related terms
   (`termination`: terminate, cancel, end, rescind, early termination, notice of termination). The planner maps
   question words to concepts. The lexical retriever then searches the variants and labels matches `concept`.
   The table is data, not code.
2. **Contextual units.** Passages are embedded with their section path and the document's type ("Service
   agreement > 12 Termination > 12.4: ..."), so a short clause carries its context into the vector.
3. **Relations.** Extraction records structures like
   `(customer) —may_terminate→ (agreement) {notice: 30 days, condition: written notice}`. A question about
   termination rights retrieves the relation, and its qualifiers answer "how much notice" without parsing text
   at query time.

## Fusion and tiers

Candidates are grouped by document (and by unit for passages). Each group gets the best tier of its evidence:

| Tier | Evidence |
|---|---|
| 1 | exact identifier, exact phrase, quoted string, exact file name, structured match on all requested conditions |
| 2 | strong lexical (all terms, concept match), entity match, relation match |
| 3 | fuzzy and OCR-tolerant matches, partial identifiers |
| 4 | semantic or contextual match above the calibrated threshold |
| reject | semantic similarity below the threshold, or below the relative margin to the best semantic match |

Within a tier, groups are ordered by reciprocal rank fusion of the retrievers' ranks, with configurable weights
per query intent. A group with evidence from several retrievers ranks above one with a single source. Tiers can
be interleaved only by the reranker (below), and never past tier 1.

Hard rules:

* A query made only of identifiers or quoted strings returns only tier 1 to 3 results (v1 already does this).
* If tier 1 results exist for an exact-intent query, results from tier 4 are shown separately, as "related",
  never mixed in.

## Reranking

An optional cross-encoder rescores the top N groups (N about 50) using the question and each group's best
passages. Constraints:

* It reorders within tiers 2 to 4, and may promote a tier 4 result above tier 3 only when its score clears a
  calibrated threshold.
* It never demotes tier 1.
* Its score becomes the calibrated confidence shown to the user.
* It runs batched on a separate service. When it is unavailable, ranking falls back to fusion and the response
  says so.

## Evidence verification

Before a result is returned:

1. Every span is re-read from the canonical text of the current version, and its text must match what the
   retriever reported.
2. For exact and lexical matches, the matched query parts must be present in the span.
3. For semantic and contextual matches, the best supporting passage is selected (reranker or vector score) and
   returned as the evidence. A result with no passage above the threshold is dropped.
4. The tenant of every evidence row is checked against `ctx` (defence in depth over row-level security).

Every result carries:

```
document, version, title, type
match_types         ["exact_identifier", "semantic"]
retrievers          ["exact", "semantic"]
confidence          0.0 to 1.0, calibrated (EVALUATION_FRAMEWORK.md)
evidence            [{page, block, char_start, char_end, bbox, text, highlight ranges, retriever}]
explanation         short text: "Identifier INV-2025-00481 on page 2; related wording on page 5"
```

## Abstention

The engine returns "no sufficiently reliable match" when:

* no candidate reaches tier 1 to 3, and
* no semantic or contextual candidate clears its calibrated threshold.

For questions that need a computed answer, it also abstains when the plan's required fields are missing, or have
low confidence in the matching documents. In that case it reports how many documents were excluded and why.
Abstention thresholds are tuned on unanswerable questions in the evaluation sets, with a false-positive budget
per query type.

## Security note on lexical indexes (decision D1)

PostgreSQL cannot use its full-text, trigram and array indexes inside a row-level security policy, because those
operators are not leakproof. The tenant's rows are scanned instead. v1 mitigates this with one pass per query,
which measured 170 to 230 ms at 50,000 documents per tenant. For larger tenants there are two options:

* **(a) Owner role and lookup functions (recommended).** The schema is owned by a migration role. The
  application role is not the owner, so row-level security still applies to all of its queries. Lexical lookups
  go through `SECURITY DEFINER` functions that read the tenant from the same transaction setting the policy uses,
  take no tenant argument, and return nothing when it is unset. Isolation is unchanged and the indexes work.
* **(b) A separate lexical engine** (OpenSearch) holding the lexical representation, with the tenant as a
  mandatory routing key and filter, enforced in one query builder.

Both need explicit approval because they change how isolation is enforced. Either way, the cross-tenant tests
cover them unchanged.
