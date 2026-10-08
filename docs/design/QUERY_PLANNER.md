# Query planner contract

The user never chooses exact, semantic or hybrid. The planner turns a question into a plan, the plan is
validated, and the executor runs it. The plan is data: logged, explainable, replayable and testable.

## Pipeline

```
question
  → normalize (Unicode, whitespace, quotes)
  → understand: language, quoted strings, identifiers, entities (from the tenant's entity index), dates,
                amounts, quantities, concepts (from packs), document types, fields, operators, intent cues
  → plan: intent + retrieval strategy + filters + computation + output
  → validate: schema, the tenant's packs and fields, limits, permissions
  → execute → results / computed answer / grounded answer
```

## QueryPlan (v2)

```json
{
  "version": 2,
  "question": "Which contracts with Acme allow early termination, and with how much notice?",
  "language": "en",
  "intent": "find_with_facts",
  "targets": { "unit_types": ["passage", "section"], "document_types": ["agreement"], "types_hard": true },
  "must": {
    "phrases": [],
    "identifiers": [],
    "entities": [{ "type": "organization", "text": "Acme", "entity_ids": ["org:acme corp"] }]
  },
  "concepts": [{ "concept": "termination", "pack": "legal", "variants": ["terminate", "cancel", "end"] }],
  "filters": {
    "dates": [], "amounts": [], "fields": [], "metadata": {}, "relations": []
  },
  "retrieval": {
    "strategies": ["entity", "lexical", "contextual", "semantic"],
    "weights": { "entity": 1.0, "lexical": 1.0, "contextual": 0.9, "semantic": 0.8 },
    "semantic_text": "contracts allowing early termination and the notice period"
  },
  "computation": null,
  "facts": [{ "relation": "may_terminate", "qualifier": "notice" }],
  "output": { "kind": "results_with_facts", "limit": 20, "group_by_document": true },
  "abstain_if": { "min_tier": 4, "min_confidence": 0.5 },
  "explain": ["entity 'Acme' resolved to 1 organization", "concept 'termination' from pack legal"]
}
```

### Intents

| Intent | Example | Retrieval | Answer |
|---|---|---|---|
| `lookup_exact` | "INV-12345", "\"for immediate release\"" | exact, fuzzy (identifier-only) | documents with spans |
| `find` | "toner replacement procedure" | exact, lexical, semantic | ranked results |
| `find_concept` | "Which contracts allow early termination?" | lexical (concept), contextual, semantic | ranked results with the supporting clause |
| `find_entity` | "documents mentioning John Smith" | entity, exact, fuzzy | results grouped by document |
| `filter_list` | "invoices above USD 10,000 issued in 2025" | structured (+ metadata) | list, sortable |
| `field_lookup` | "When does the Acme agreement expire?" | resolve target (entity, lexical, semantic) → structured | value with source |
| `fact` | "How many days notice is required?" | relations → lexical/semantic → quoted figure | value with source |
| `aggregate` | "How much did Company X spend in 2025?" | structured | computed value per unit or currency, with contributing documents |
| `group` | "invoices per vendor in 2025" | structured | table |
| `compare` | "differences between the 2024 and 2025 Acme agreements" | resolve both → section alignment | differences with spans (phase 6) |
| `question_answer` | open questions that need synthesis | all retrievers → evidence → answering | grounded answer with verified citations (optional, D2) |

For your aggregation example, the plan is computed deterministically:

```json
{ "intent": "aggregate",
  "must": { "entities": [{ "type": "organization", "text": "Company X" }] },
  "computation": { "op": "sum", "measure": "money", "roles": ["total_amount", "amount_paid"],
                   "relation": "paid_by|billed_to", "period": { "field": "issue_date", "from": "2025-01-01", "to": "2025-12-31" },
                   "group_by": "currency" },
  "retrieval": { "strategies": ["structured", "entity", "exact"] } }
```

## How plans are produced

1. **Deterministic planner (always available).** v1's rule-based planner, extended with entity resolution
   against the tenant's index, concept detection from packs, and the new intents. Every rule is unit-tested, as
   in v1 (`tests/unit/test_planner.py`).
2. **Model-assisted planner (optional, D2).** For questions the deterministic planner marks low-confidence, a
   language model is asked to fill the same JSON schema. It is constrained to the tenant's known fields,
   document types, concepts and entities, and to structured output. Its plan goes through the same validation,
   and the deterministic plan is kept as a fallback. The engine is fully functional without this.
3. **Validation.** Unknown fields, types or operators are rejected, never guessed. The tenant always comes from
   the caller, never from the plan. Limits apply (candidates, time, result size). A failed plan degrades to
   `find` with the plain question, and the response says so.

## Execution rules

* Retrievers named in `strategies` run in parallel within a time budget. `exact` always runs when the question
  contains identifiers or quoted strings, whatever else the plan says.
* Structured filters are applied before retrieval when they are selective, after it otherwise (the executor
  decides by estimated counts).
* Computations run in SQL, return contributing documents and spans, and report excluded documents (missing or
  low-confidence values, still processing).

## Explainability

`explain=true` returns the plan, the retrievers run and their timings, candidate counts per retriever and tier,
and the reason for abstaining. The UI shows a short form ("Searched for identifier INV-12345 and related wording;
1 exact match").
