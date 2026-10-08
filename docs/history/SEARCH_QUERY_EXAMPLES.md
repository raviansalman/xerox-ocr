# Search query examples (target behaviour specification)

These examples define what the target document query engine (`docs/SEARCH_ARCHITECTURE.md`) must do. **None of
the planned capabilities below are claimed to exist today.** The "Today" line records current behaviour, measured
on the golden corpus where it applies.

Where an example says **[golden xdemo]**, the expected answer is computed over the synthetic `xdemo` tenant in
`ultimate/tests/golden/cases.py`, so it can become an executable evaluation case. Tenant `xother` deliberately
contains a California NDA and the same invoice number, so any answer that counts or cites it for `xdemo` is a
tenant leak. Other examples use a hypothetical Xerox corpus and need real documents before they can be graded.

Plan notation: `R.lex` lexical retrieval, `R.vec` vector retrieval, `R.struct` structured/metadata query,
`R.vis` visual/page-region query, `F` fusion, `RR` rerank, `AGG` deterministic aggregation, `GEN` answer
generation from evidence. `T` is the tenant/ACL context injected by the executor, never by the planner.

---

## A. Exact and lexical

### 1. `FOR IMMEDIATE RELEASE`
* **Query type:** EXACT (phrase)
* **Capabilities:** phrase match, case folding, whitespace normalization
* **Plan:** `R.lex.phrase("for immediate release", T)` → exact tier; `R.vec(T)` for the remaining slots → `F` (exact first) → results
* **Evidence:** document, page, chunk, highlighted span; `match_type=exact_phrase`
* **Answer type:** ranked document list
* **Expected:** press release first; the firmware note (all three words, no phrase) must not outrank it
* **Today:** passes, but only through the sort tiebreak among vector candidates (vector score 0.49 vs 0.56)

### 2. `INV-2026-00481` [golden xdemo]
* **Query type:** EXACT (identifier)
* **Capabilities:** identifier tokenization that keeps `INV-2026-00481` as one token plus its parts; n-gram/substring fallback
* **Plan:** `R.lex.identifier(T)`; if no hit, `R.lex.substring("00481", T)`
* **Evidence:** `Tax_Invoice_INV-2026-00481.pdf`, page 1, span
* **Answer type:** document list
* **Expected:** `x_invoice` only. `xo_nda` (tenant `xother`) contains the same id and must never appear
* **Today:** passes (vector + overlap boost on a 13-document tenant)

### 3. `INV 2026 00481` [golden xdemo]
* **Query type:** EXACT (identifier, separators differ)
* **Capabilities:** identifier normalization (strip `- / . #` and spaces into a canonical form at ingest and query time)
* **Plan:** `R.lex.identifier_canonical("inv202600481", T)`
* **Expected:** `x_invoice`, `match_type=normalized_identifier`

### 4. `Contract No. 17/2024` [golden xdemo]
* **Query type:** EXACT (identifier inside a phrase)
* **Capabilities:** keeps `17/2024` as a token (the Postgres spike showed `simple` parsing does; ICU splits it unless configured)
* **Plan:** `R.lex.phrase(T)` + `R.lex.identifier("17/2024", T)`
* **Expected:** `x_contract` first, `x_arabic_contract` (same number, Arabic) second with `match_type=identifier`

### 5. `Article 12.4` [golden xdemo]
* **Query type:** EXACT (clause reference)
* **Capabilities:** numeric dotted tokens, clause-reference recognizer
* **Plan:** `R.lex.phrase(T)`; clause index lookup `R.struct(clause_ref="12.4", T)` once clauses are extracted
* **Evidence:** the article heading and its paragraph
* **Expected:** `x_contract`, then `x_arabic_contract` (`المادة 12.4`) if cross-language clause references are linked

### 6. `PO-2025-0193` [golden xdemo]
* **Query type:** EXACT (identifier)
* **Expected:** `x_po`. Filename and content both contain it; the evidence must say which matched

### 7. `blue heron protocol 7781` (haystack tenant)
* **Query type:** EXACT (phrase in a semantically unrelated document)
* **Capabilities:** lexical retrieval independent of vector similarity
* **Plan:** `R.lex.phrase(T)` → exact tier
* **Expected:** the needle first
* **Today:** **fails in every mode**: the needle is 61st of 61 by cosine and vector search keeps 50. Spike: Milvus 2.6 BM25, Milvus `PHRASE_MATCH` and Postgres `phraseto_tsquery` all return it first; plain RRF of dense+BM25 put it second, which is why exact matches need a precedence tier

### 8. `Remote_Work_Policy` / `Toner_Replacement_Procedure.txt`
* **Query type:** FILENAME
* **Plan:** `R.struct(filename ~ query, T)` exact, then token match; content retrieval in parallel
* **Expected:** the named file first, `match_type=filename_exact`

## B. Semantic and contextual

### 9. `agreements that let either side end the contract early` [golden xdemo]
* **Query type:** SEMANTIC
* **Capabilities:** dense retrieval, paraphrase understanding, cross-encoder rerank
* **Plan:** `R.vec(T)` + `R.lex(T)` (low weight) → `F` → `RR`
* **Expected:** `x_contract` (Article 12.4 termination for convenience) and `x_employment_ca` (either party may end employment) in the top 2; `x_policy` ("termination of the policy") ranked below them
* **Today:** passes in `both`/`vector`

### 10. `who handles questions from journalists`
* **Query type:** SEMANTIC (weak lexical overlap)
* **Expected:** press release (media contact)
* **Today:** fails in all modes (remote-work policy first; semantic mode returns nothing because of key-term gates)

### 11. `Find documents discussing customer payment obligations`
* **Query type:** SEMANTIC / CONTEXTUAL
* **Plan:** `R.vec(T)` + `R.lex` on expanded terms (payment, due, net, invoice) → `F` → `RR`
* **Expected:** invoices, the lease, the supply agreement (net 45) ahead of policies that only mention invoices

### 12. `Which documents discuss intellectual property ownership?`
* **Query type:** SEMANTIC (question form, retrieval intent)
* **Plan:** planner strips question scaffolding → `R.vec` + `R.lex` → `F` → `RR` → document list with evidence passages
* **Answer type:** document list (no generated summary unless asked)

## C. Entity and metadata

### 13. `Northgate Utilities` [golden xdemo]
* **Query type:** ENTITY (organization)
* **Capabilities:** organization extraction at ingest; entity index; lexical fallback
* **Plan:** `R.struct(party/org = "Northgate Utilities", T)` ∪ `R.lex.phrase(T)`
* **Expected:** `x_invoice`, `match_type=entity_org` with the "Bill to" span as evidence

### 14. `John Smith` [golden xdemo]
* **Query type:** ENTITY (person)
* **Expected:** `x_nda_ca` (typed signature line) and `x_scanned_letter` (OCR text "Approved by John Smith"); `xo_nda` excluded (other tenant)
* **Note:** the scanned letter matches through OCR text, not through a detected signature (see 21)

### 15. `contracts governed by California law` [golden xdemo]
* **Query type:** METADATA (jurisdiction) + SEMANTIC fallback
* **Capabilities:** governing-law clause extraction to a normalized `jurisdiction` field; OCR-tolerant jurisdiction values
* **Plan:** `R.struct(doc_type ∈ contract family, jurisdiction = "US-CA", T)`; fall back to `R.lex.phrase("laws of the State of California")` + OCR-folded match when extraction is missing
* **Expected:** `x_nda_ca`, `x_employment_ca` (California Labor Code), `x_ocr_califomia` (OCR "Califomia", `match_type=ocr_tolerant`); not `x_nda_tx`, not `x_contract` (Saudi law)
* **Today:** `both`/`vector` rank the three first; **semantic mode returns the Saudi contract first**

### 16. `documents mentioning Jeddah`
* **Query type:** ENTITY (location)
* **Expected:** `x_po`; location extraction normalizes `Jeddah` / `Jiddah` / `جدة`

## D. Classification and counting

### 17. `How many NDAs do we have?` [golden xdemo]
* **Query type:** AGGREGATION (count) over CLASSIFICATION
* **Plan:** `R.struct(doc_type = "nda", confidence ≥ θ, T)` → dedupe by `document_id` → `AGG.count`
* **Evidence:** the list of counted documents and any "uncertain" documents (θ_low ≤ confidence < θ)
* **Answer type:** number + evidence list
* **Expected:** **2** (`x_nda_ca`, `x_nda_tx`). Tenant `xother` has 1 more; returning 3 is a tenant leak
* **Must not:** count chunks, or ask an LLM to estimate from retrieved text

### 18. `What percentage of our contracts are NDAs?` [golden xdemo]
* **Query type:** AGGREGATION (ratio)
* **Plan:** `AGG.count(doc_type="nda", T) / AGG.count(doc_type ∈ contract family, T)`
* **Expected:** contract family = NDA ×2, service contract, employment agreement, supply agreement, equipment lease, Arabic maintenance contract = 7 → 2/7 ≈ 28.6%, with the denominator listed. The definition of "contract family" must be explicit config, not LLM judgement

### 19. `How many contracts are governed by California law?` [golden xdemo]
* **Query type:** AGGREGATION + METADATA
* **Expected:** 3 (NDA, employment agreement, supply agreement via OCR-tolerant jurisdiction); the answer flags 1 as OCR-tolerant evidence

### 20. `How many invoices were issued in 2026?`
* **Query type:** AGGREGATION + DATE
* **Plan:** `R.struct(doc_type="invoice", issue_date ∈ [2026-01-01, 2026-12-31], T)` → `AGG.count`
* **Expected [golden xdemo]:** 1 (`x_invoice`, 14 January 2026)

## E. Signature and visual

### 21. `How many documents contain a signature?`
* **Query type:** VISUAL + AGGREGATION
* **Capabilities:** page rendering, signature region detection with confidence, per-page storage of regions
* **Plan:** `R.vis(region_type="signature", confidence ≥ θ, T)` → dedupe by document → `AGG.count`
* **Evidence:** document, page, bounding box, crop thumbnail, confidence
* **Note:** typed names ("Signed: John Smith") are **signature text**, not proof of a signature. The answer must distinguish `signature_detected` (visual) from `signature_line_text` (textual)
* **Today:** not implemented

### 22. `Which contracts are unsigned?`
* **Query type:** VISUAL + CLASSIFICATION + negation
* **Plan:** contracts (`doc_type` family) MINUS documents with a detected signature region on any page
* **Risk:** negation over a detector produces false "unsigned" results for low-quality scans; return "no signature detected (confidence)" rather than asserting "unsigned"

### 23. `documents signed by John Smith`
* **Query type:** VISUAL + ENTITY
* **Plan:** signature regions ∩ person entity "John Smith" near the region (same page, within N lines below/above)
* **Expected [golden xdemo]:** `x_nda_ca` by text association; `x_scanned_letter` is "approved by", not a signature, unless a signature region is detected

## F. Dates and numbers

### 24. `Which agreements expire in 2027?` [golden xdemo]
* **Query type:** STRUCTURED (date)
* **Plan:** `R.struct(expiry_date ∈ 2027, T)`; extraction stores the date with its source span
* **Expected:** `x_contract` (31 December 2027)

### 25. `Show contracts where the payment amount is greater than $100,000` [golden xdemo]
* **Query type:** STRUCTURED (numeric) + currency
* **Capabilities:** amount extraction with currency; currency conversion only if rates are configured (otherwise filter per currency and say so)
* **Expected:** `x_lease` (total USD 432,000). `x_invoice` is SAR 418,750: included only if the user asked in any currency or a SAR→USD rate is configured; the answer must state which

### 26. `How many employees were paid in 2024?` [golden xdemo]
* **Query type:** STRUCTURED extraction + AGGREGATION
* **Plan:** if payroll records are row-level data: `AGG.count(distinct employee_id, pay_date ∈ 2024, T)`. If only a summary document exists: return the stated figure **as a quoted fact with its source**, not as a computed count
* **Expected:** "42, as stated in Payroll_Summary_2024.txt" with `answer_kind=quoted_figure`

### 27. `What is the total value of contracts with Acme?`
* **Query type:** ENTITY + STRUCTURED + AGGREGATION (sum)
* **Plan:** `R.struct(party="Acme", doc_type ∈ contract family, T)` → amounts → group by currency → `AGG.sum`
* **Answer type:** per-currency totals + contributing documents; refuse to add currencies without rates

## G. Legal clauses

### 28. `Which contracts have termination clauses?`
* **Query type:** CLAUSE (classification at clause level)
* **Plan:** clause index `R.struct(clause_type="termination", T)`; fallback `R.vec` over clause-segmented chunks + `RR`
* **Expected [golden xdemo]:** `x_contract` (Article 12.4), `x_arabic_contract` (المادة 12.4), `x_employment_ca` (at-will end of employment, lower confidence). `x_policy` mentions "termination of the policy" and must be excluded

### 29. `Find the termination clause in the Riyadh Logistics contract`
* **Query type:** ENTITY + CLAUSE + exact evidence
* **Plan:** resolve party → document(s) → clause index → return the clause text with page and span
* **Answer type:** passage (extractive), optional short summary marked as generated

### 30. `What notice period do our termination clauses require?`
* **Query type:** RAG over clauses + extraction
* **Plan:** clause retrieval → extract `notice_period_days` → group → `GEN` summary citing each clause
* **Expected [golden xdemo]:** 60 days (Article 12.4, both language versions); employment agreement: none ("at any time")

## H. Multi-condition

### 31. `How many signed NDAs governed by California law do we have?`
* **Query type:** COMPOSITE: CLASSIFICATION + METADATA + VISUAL + AGGREGATION
* **Plan:** `doc_type="nda"` ∩ `jurisdiction="US-CA"` ∩ (`signature_detected` ∪ `signature_line_text`) → dedupe → `AGG.count`, reporting each condition's evidence and confidence separately
* **Expected [golden xdemo]:** 1 by signature text (`x_nda_ca`); visual signature status unknown (PDF has no image)

### 32. `invoices from Northgate Utilities above SAR 400,000 in Q1 2026`
* **Query type:** COMPOSITE: ENTITY + NUMERIC + DATE + CLASSIFICATION
* **Expected [golden xdemo]:** `x_invoice` (SAR 418,750.00, 14 January 2026)

### 33. `contracts mentioning California that contain a signature`
* **Query type:** COMPOSITE: LEXICAL/ENTITY + VISUAL
* **Note:** "mentioning California" is lexical/entity (not jurisdiction); the planner must not silently upgrade it to `jurisdiction = US-CA`

## I. OCR errors and language

### 34. `Califomia` [golden xdemo]
* **Query type:** EXACT (user typed the OCR form)
* **Expected:** `x_ocr_califomia` exact; also offer `California` results as "did you mean" (not merged silently)

### 35. `maintenance contract` (document text is `MAINTENANCE C0NTRACT ... rnaintenance`)
* **Query type:** LEXICAL with OCR folding
* **Capabilities:** OCR-confusion folding at ingest and query (`0↔o`, `1↔l↔i`, `rn↔m`, `vv↔w`), applied to a secondary normalized field only
* **Expected:** the OCR-noisy contract, `match_type=ocr_tolerant`
* **Today:** passes through embeddings on a tiny corpus; no fuzzy layer exists

### 36. `Lisa Riordon` (misspelled person)
* **Query type:** ENTITY with fuzzy name matching
* **Plan:** exact entity lookup fails → name-similarity (edit distance ≤ 1 for names ≥ 6 chars) → result flagged `fuzzy_name`

### 37. `إنهاء العقد` (Arabic: termination of the contract) [golden xdemo]
* **Query type:** EXACT/LEXICAL (Arabic)
* **Capabilities:** Arabic normalization (alef forms, ta marbuta, diacritics, tatweel), light stemming (prefix `ال`, `ب`, `و`)
* **Expected:** `x_arabic_contract`. Postgres `arabic` config produced `انهاء`, `عقد`, `اشعار` (prefixes stripped); Milvus ICU tokenized without stemming

### 38. `termination clause` → also Arabic documents?
* **Query type:** CROSS-LANGUAGE SEMANTIC
* **Requirement:** a multilingual embedding model (all-mpnet-base-v2 is English-centric) or translation at query time. **Open decision**; measure on real Xerox Arabic documents

### 39. `zzqxunknownzzq`
* **Query type:** no match
* **Expected:** an empty result with "no documents matched"; never "the closest 10 documents"
* **Today:** `both`/`vector` return 10 unrelated documents (score floor 0.0)
