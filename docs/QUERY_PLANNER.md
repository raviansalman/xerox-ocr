# Query understanding and planning

Status: design for review. Today there is no planner: the legacy `enhance_query` (925 lines of corpus-specific
rules) raises on every call and is not part of the target (`SEARCH_FORENSICS.md`).

Principle: **the user asks a question; the engine decides how to answer it**, by turning the question into a
typed, validated plan that deterministic components execute.

## 1. Pipeline

```
question ─► normalize (same normalize() as ingest) ─► detect language
        ─► rule analyzers (quotes, identifiers, dates, amounts, counting cues, doc-type terms, field names)
        ─► [optional] LLM planner, only for composite/unparsed questions, schema-constrained
        ─► plan validation (schema, allow-listed fields and operators, limits)
        ─► executor (retrievers + aggregation + answer) with AuthContext injected
        ─► answer + evidence + explain
```

## 2. Plan schema

```
QueryPlan {
  version: "1"
  intent: RETRIEVE_DOCUMENTS | RETRIEVE_PASSAGES | COUNT | SUM | AVG | MIN | MAX | PERCENTAGE |
          GROUP_BY | COMPARE | ANSWER_QUESTION | LOOKUP_FIELD
  text: {
    exact_phrases: [str], identifiers: [str], terms: [str],
    semantic: str | null,               # what to embed, if anything
    language: "en" | "ar" | ...
  }
  scope: { doc_types: [label], collections: [id] }          # never tenant: injected by the executor
  filters: [ { field: <allow-listed field>, op: = | != | < | <= | > | >= | between | in | exists | not_exists,
               value: typed, origin: explicit | inferred, confidence } ]
  visual: [ { region_type: signature | stamp | handwriting | checkbox, min_confidence } ]
  aggregate: { op, field?, group_by?: [field], distinct_on: "document_id", currency_policy?: per_currency | convert }?
  output: { kind: documents | passages | number | table | quoted_figure | generated_answer, limit, explain: bool }
  planner: { source: rules | llm, model?: name@version, confidence }
}
```

Allow-listed fields come from the deployment's field schema (`fields.name` values configured per document type)
plus built-ins (`doc_type`, `language`, `created_at`, `page_count`, `filename`). Anything else fails validation.

## 3. Rule planner (always on, deterministic)

| Signal | Recognizer | Plan effect |
|---|---|---|
| Quoted text | `"..."` | `exact_phrases`, exact tier |
| Identifier shapes | letters+digits with `- / . #`, `No.`/`Ref`/`رقم` prefixes, configurable tenant patterns | `identifiers` |
| Dates and ranges | multi-locale parser (`in 2027`, `Q1 2026`, `between …`, `next year` relative to a supplied clock) | date filters on the right field (`issue_date`, `expiry_date`, …) when the field is named or implied by a configured cue (`expire` → `expiry_date`) |
| Amounts | number + currency, comparison words (`greater than`, `over`, `أكثر من`) | numeric filters |
| Counting cues | `how many`, `number of`, `count`, `كم عدد` | `COUNT` |
| Ratio cues | `percentage`, `what share`, `نسبة` | `PERCENTAGE` |
| Sum/avg cues | `total`, `sum`, `average` + a numeric field | `SUM` / `AVG` |
| Document types | taxonomy labels and synonyms (data, per deployment) | `scope.doc_types` |
| Field names | configured field synonyms (`governed by` → `jurisdiction`) | filters |
| Signature words | `signed`, `signature`, `unsigned` | `visual` / `signature_*` filters |
| Short query without structure | 1 to 6 tokens, no cues | `RETRIEVE_DOCUMENTS` with exact + lexical + semantic |

All vocabulary (doc types, field synonyms, identifier patterns, gazetteers) is **data loaded per deployment**,
versioned, and tested. None of it is a Python literal (`HARDCODED_DATA_AUDIT.md`).

## 4. LLM planner (optional, on-prem)

* Used only when the rule planner reports `unparsed_constraints` or a composite question it cannot type.
* Input: the question, the allow-listed schema (field names, types, document type labels, operators), today's
  date. **Never** document text, never other tenants' data, never the tenant id.
* Output: JSON constrained to the plan schema (constrained decoding or validation + one repair attempt).
* Validation rejects: unknown fields/operators, tenant or ACL fields, limits above caps, filters whose values are
  not grounded in the question (for example a date range that does not appear in it).
* On failure or low confidence: fall back to the rule plan + hybrid retrieval and say which constraints were not
  applied ("I searched for … but could not apply 'signed by John Smith'").
* The model, prompt and schema versions are logged with every plan.

The LLM may **not**: choose or change identity, tenant or ACL; bypass validation; produce data values or numbers
that are presented as results; decide authorization.

## 5. Execution

* The orchestrator maps each plan node to retrievers (`SEARCH_ARCHITECTURE.md`): exact/lexical, vector,
  structured (SQL), visual (regions), all receiving `AuthContext`.
* Filters are pushed down into each retriever (pre-filtering), not applied after truncation.
* Aggregations run as SQL over the structured store with `DISTINCT document_id` by default, return the number,
  the evidence ids and the confidence buckets (`high`, `review`, `pending`).
* Per-node budgets and timeouts; partial results are labelled.

## 6. Answers

| Output kind | Produced by | Contains |
|---|---|---|
| `documents` / `passages` | retrieval + fusion + rerank | ranked items with `match_type`, confidence, evidence |
| `number` / `table` | aggregation executor | value(s), evidence document ids, buckets, applied filters |
| `quoted_figure` | lookup of a stated value | the figure as written, its source span, a note that it is quoted, not computed |
| `generated_answer` | RAG over retrieved passages | prose where every sentence cites evidence; statements labelled FACT / INFERENCE / UNCERTAIN |
| `none` | any path below its floor | "No sufficiently reliable evidence was found", plus what was searched |

Generation rules: the generator sees only authorized evidence passages, treats them as data (prompt-injection
boundary), must cite for every claim, and may not introduce numbers that are not in the evidence or computed by
the executor. A post-check verifies that each cited span exists and supports the quoted text.

## 7. Abstention and uncertainty

* Each retriever has a calibrated floor per match type; below all floors → `none`.
* Aggregations over low-confidence classifications report them separately: "2 NDAs (1 more possible, 0 pending
  classification)".
* Signer identity: "11 documents with signature-like regions; 7 associated with John Smith with high confidence;
  4 need review" rather than a single number.
* Ambiguous questions (two plausible plans with different answers) return the interpretation used and offer the
  alternative.

## 8. Worked examples

`How many contracts governed by California law are NDAs?`

```json
{"intent":"COUNT","scope":{"doc_types":["nda"]},
 "filters":[{"field":"jurisdiction","op":"=","value":"US-CA","origin":"explicit"}],
 "aggregate":{"op":"count","distinct_on":"document_id"},
 "output":{"kind":"number","explain":true},"planner":{"source":"rules","confidence":0.92}}
```

`Show contracts signed by John Smith governed by California law`

```json
{"intent":"RETRIEVE_DOCUMENTS","scope":{"doc_types":["@contract_family"]},
 "filters":[{"field":"jurisdiction","op":"=","value":"US-CA"},{"field":"signer","op":"=","value":"John Smith"}],
 "visual":[{"region_type":"signature","min_confidence":0.6}],
 "output":{"kind":"documents","explain":true},"planner":{"source":"llm","model":"<on-prem>@<v>","confidence":0.81}}
```

`What is the total value of Acme contracts expiring in 2027?`

```json
{"intent":"SUM","scope":{"doc_types":["@contract_family"]},
 "filters":[{"field":"party","op":"=","value":"Acme"},{"field":"expiry_date","op":"between","value":["2027-01-01","2027-12-31"]}],
 "aggregate":{"op":"sum","field":"contract_value","distinct_on":"document_id","currency_policy":"per_currency"},
 "output":{"kind":"number","explain":true}}
```

`FOR IMMEDIATE RELEASE`

```json
{"intent":"RETRIEVE_DOCUMENTS","text":{"exact_phrases":["for immediate release"],"semantic":"for immediate release"},
 "output":{"kind":"documents"},"planner":{"source":"rules","confidence":0.99}}
```

More: `SEARCH_QUERY_EXAMPLES.md` (39 cases with expected plans and answers).

## 9. Testing the planner

* Plan-level golden tests: question → expected plan (fields, operators, values), independent of retrieval.
* Adversarial: questions that try to name another tenant, inject instructions, request unlimited results, or
  smuggle SQL/expression syntax into values; all must be rejected or neutralized.
* LLM planner variance: the same question 5 times must produce an equivalent plan; disagreement is logged.
