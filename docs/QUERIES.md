# Questions the engine answers

Every example below runs in the test suite against the synthetic corpus (`tests/integration/test_search.py`,
`test_retrieval_v2.py`, `test_computations.py`, the golden set in `tests/fixtures/golden/fixtures.yaml`) or, for the
remaining date and amount forms, at the planner level (`tests/unit/test_planner.py`).

```
POST /api/v1/query {"q": "...", "limit": 20, "explain": false, "answer": false}
```

With `explain`, the response includes the validated plan (intent, filters, concepts, entities, strategies) and the
time spent per retriever. With `answer`, it includes a grounded answer (see below).

## Finding documents

| Kind | Examples | Behaviour |
|---|---|---|
| Identifier | `INV-2026-00481`, `INV 2026 00481`, `inv202600481`, `Contract No. 17/2024`, `Article 12.4` | Any spelling matches. A query made only of identifiers returns only documents that contain them. |
| Partial identifier | `00481`, `48213` | Digit groups inside longer identifiers (prefix and suffix). |
| Exact phrase | `"sixty (60) days written notice"` | Quoted text must appear as a phrase (or OCR-tolerantly). |
| Names and terms | `Nadia Hartwell`, `Northgate Utilities`, `VersaLink C405` | Exact words first, then stems. |
| Joined or split words | `Versa Link C405` | Matches `VersaLink C405`. |
| OCR errors | `Califomia`, `FOR IMMEDlATE RELEASE`, `rnaintenance visits` | OCR confusions are folded on both sides. |
| Typos | `Nadia Hartwel`, `Northbrige Data` | Corrected from the tenant's own vocabulary, only when nothing matches exactly. |
| File names | `Toner_Replacement_Procedure`, `Payroll_Summary_2024.xlsx` | |
| Meaning | `rules for working from home`, `how long does the equipment lease last` | Semantic search (when enabled), above a calibrated similarity floor. |
| Concepts | `Can either party cancel the contract early?` | The pack's concept (termination) reaches clauses of that type, relations and every way it is written ("terminate", "end the employment"), with or without semantic search. |
| People and organizations | `agreements involving Bluefin Analytics`, `Riyadh Logistics` | Matched against extracted entities, with the mention as evidence. |
| Inside attachments | `delivery schedule Jeddah` | E-mail attachments are pages of the e-mail. |

Each result has the document type, the retrievers and match types that found it, a tier (1 exact, 2 strong, 3
tolerant, 4 meaning only), a confidence, key fields, an explanation, and up to three evidence items with page and
character span, re-read from the stored text. When nothing reliable is found the answer is
`{"kind": "none", "text": "No sufficiently reliable evidence was found for this question."}`; when every result
is related only by meaning, the answer says so.

## Arabic

Arabic text is normalized the same way when it is indexed and when it is searched: diacritics, tatweel and invisible
marks are ignored, alef forms (أ إ آ ٱ), ى/ي, ة/ه and the hamza seats ؤ ئ are folded, and Arabic-Indic digits equal
ASCII digits. Words are reduced to light stems (articles, attached prepositions and common suffixes removed), so
`طابعة` finds `الطابعات` and `للعقد` finds `العقد`. Broken plurals (`عقود` for `عقد`) are not stemmed; meaning search
covers them.

| Kind | Examples |
|---|---|
| Phrase, any spelling | `محطة تحلية المياه`, `تحليه المياه`, `تَحْلِيَة المِيَاه` |
| Identifiers | `WDP-2024/017`, `CON-٢٠٢٤/٠٧٧` (same as `CON-2024/077`) |
| Names | `احمد الزهراني` finds `أحمد الزهراني` |
| Meaning (multilingual model) | `ما هي مدة عقد الصيانة؟`; English questions find Arabic documents and the reverse |

Meaning search in Arabic needs `DOCINTEL_EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2` (the embedding
service must be built with the same model; documents are reprocessed after a model change). With the default English
model, Arabic phrases, words, names and identifiers are still found. The question planner (counts, totals, date
filters) and document types, fields and clauses understand English questions and documents only.

## Filters

Filters combine with each other and with search text.

| Filter | Examples |
|---|---|
| Document type | `press release`, `invoices from 2025` (types come from the tenant's packs) |
| Governing law | `contracts governed by California law` |
| Dates | `Which agreements expire in 2027?`, `invoices issued in 2026`, `in Q1 2026`, `in March 2027`, `between 2025 and 2027`, `next year`, `in the next 90 days`, `since 2025`, `before 2025` |
| Amounts | `contracts where the payment amount is greater than $100,000`, `invoices above SAR 400,000`, `between 1,000 and 5,000 USD` |
| Parties | `contracts with Northwind`; a known organization named without a cue is resolved from extracted entities |
| Signatures | `documents signed by John Smith`, `Which contracts are unsigned?`, `documents that contain a signature` |
| Clauses | `Which contracts have termination clauses?` |
| Scans and formats | `scanned documents`, `pdf invoices` |

Date words choose the date: *issued* the issue date, *expire* the expiry date, *due* the due date, *signed* the
signature date; each type's default date comes from its pack.

## Computed answers

Computed from extracted fields with SQL, never estimated and never written by a language model. Each answer has a
`calculation` (operation, field, filters, periods, group, documents considered, documents with and without values)
and `records` (document, value, page, character span).

| Kind | Example |
|---|---|
| Count | `How many NDAs do we have?`, `How many documents mention Jeddah?` |
| Percentage | `What percentage of our contracts are NDAs?` (33.3%, 2 of 6) |
| Sum, average, min, max | `What is the total value of all invoices?`, `How much did Northwind spend?` (one row per currency; currencies are never added) |
| Group | `How many documents per type?`, `total value of invoices by customer`, `total invoice value by currency` |
| Period comparison | `Compare the total value of invoices in 2025 and 2026`, `How many invoices were issued in 2025 and 2026?` |
| Field lookup | `What is the invoice number of the Northgate Utilities invoice?`, `When does the service contract expire?` (value, document, page, span) |
| Figure stated in a document | `How many days notice is required to terminate the service contract?` (quoted from the relation that states it) |

Documents without a value are counted and reported, not silently dropped.

## Grounded answers (optional)

`"answer": true` adds `generated_answer`:

* `extractive` (default): the evidence sentences that best answer the question, quoted with document, page and span.
* `anthropic` (`DOCINTEL_ANSWER_PROVIDER=anthropic`): a Claude-written answer. Each sentence must cite evidence ids;
  sentences whose citations do not exist, whose numbers or quotes are not in the cited evidence, or whose words are
  mostly unsupported are dropped (and listed in `dropped`). With nothing left, the answer abstains. A provider
  failure or refusal falls back to the extractive answer with a `degraded` note.
* Computed questions return the computed answer (`status: computed`); questions without evidence abstain without
  calling a model.

## Isolation

Every question runs inside the caller's tenant. The evaluation harness asks every labelled question as every other
tenant, and the adversarial suite probes each index (postings, vocabulary corrections, entities, structured counts,
vectors, settings) for foreign documents; both require zero.
