# Canonical document model

"Fully ingested" means: the original file is stored unchanged, and every piece of its content that can be
extracted exists as an addressable unit with provenance, with the annotations derived from it attached to exact
spans. Indexes are projections of this model and can always be rebuilt from it.

## Structure

```
Document                     identity, tenant, source, file metadata, type(s), language(s), status
 └─ DocumentVersion          one processing run: parser/model/config versions, timings, warnings
     ├─ Page                 number, kind (native|ocr|sheet|slide|section|attachment), size, rotation, OCR confidence,
     │   │                   image reference
     │   ├─ Block            ordinal, type (heading|paragraph|list_item|table|caption|header|footer|footnote|
     │   │                   key_value|signature|figure|other), reading order, bbox, text, section path, confidence
     │   │   └─ Line / Word  text, bbox, confidence (OCR and native PDF), stored compactly per page
     │   ├─ Table            bbox, caption, header rows, cells (row, col, spans, text, bbox), header→cell links
     │   └─ Image            bbox, kind (photo|chart|diagram|stamp|signature|logo), OCR text, description (optional)
     ├─ Section              hierarchy from headings: path, title, block range
     ├─ Unit                 retrievable text unit (see "Units"), with its block range and character offsets
     ├─ Annotation           everything extracted, always anchored to a Span
     │   ├─ Mention          entity mentions: type, surface text, normalized value, span
     │   ├─ Entity           resolved entity within the document (mentions grouped), and its tenant-level link
     │   ├─ Value            dates, amounts, quantities, percentages, durations, identifiers: typed and normalized
     │   ├─ Field            named field from a domain pack (invoice_number, expiry_date, diagnosis ...) → Value
     │   ├─ Classification   document type, domain, block or section labels, with confidence and method
     │   └─ Relation         subject → predicate → object (+ qualifiers), each end a Mention, Value or Document
     └─ Provenance           on every row: document, version, page, block, char_start, char_end, bbox, extractor,
                             extractor_version, confidence
```

### Span

The anchor that makes every fact provable:

```
Span = (document_id, version, page, block_id, char_start, char_end, bbox?)
```

Character offsets refer to the block's canonical text. For tables a span may instead address a cell
(`table_id, row, col`). Offsets are stable for a given version and regenerated with each new version.

## Worked example

Page 4 of a PDF contains "Agreement signed by John Smith on March 14, 2025 for $250,000."

| Representation | Stored as |
|---|---|
| Original | the PDF, content-addressed, never modified |
| Block | `b-4-12`, paragraph, page 4, bbox (72, 410, 523, 438), text as written, section "7 Execution" |
| Lexical | exact tokens, phrase positions, English stems, OCR-folded form, identifiers (none here), trigram text |
| Vector | embedding of the unit that contains `b-4-12`, with its section title as context |
| Mention | PERSON "John Smith" chars 20–30, normalized `john smith`, linked to tenant entity `person:john smith` |
| Value | DATE 2025-03-14 chars 34–48, role `signature_date`; MONEY 250000 USD chars 53–61, role `contract_value` |
| Classification | document type `agreement` (pack: legal), confidence 0.93, method rules+embedding |
| Relations | (John Smith) —signed→ (this document) [date 2025-03-14]; (this document) —has_value→ (250000 USD) |
| Evidence | every row above carries page 4, block `b-4-12`, its character span and the extractor that produced it |

Seven independent ways to reach the same sentence: the phrase, the name, the date, the amount, the document
type, the relation, or the meaning.

## Units (what retrieval returns)

| Unit | Built from | Used by |
|---|---|---|
| Block | one block | exact and lexical evidence, highlighting, citations |
| Passage | consecutive blocks within one section, about 150 to 400 tokens, with the section path as context | lexical ranking, semantic retrieval, reranking, RAG |
| Section | a heading and its blocks | contextual and semantic retrieval of long topics |
| Table row | one row with its header cells as context ("Item: Toner; Qty: 20; Price: USD 40") | exact, semantic and structured retrieval of tabular data |
| Document | title, type, key fields, optional summary | document-level semantic retrieval, listing |

v1 has one unit, the page-bounded chunk. v2 keeps chunks as passages, and the other units are added beside them.

## Versioning and reprocessing

* A `DocumentVersion` is created by each processing run and records parser, OCR engine, layout model, extractors,
  packs, embedding model and configuration hashes.
* The version becomes current atomically, once all of its indexes are written. Readers never see half a version.
* Changing a model or pack marks the affected documents stale. Re-runs go through the normal queue in the
  background, and the previous version keeps serving until the new one is complete.
* Old versions are kept for a configurable period (for audit and comparison), then purged with their index rows.

## Storage schema (additive to v1)

All tables carry `tenant_id` and the same row-level security policy as v1.

| Table | Key columns |
|---|---|
| `document_versions` | id, document_id, number, status, versions (jsonb), created_at, is_current |
| `pages` (v1, extended) | + version_id, rotation, image_key, words (compact binary or jsonb) |
| `blocks` | id, version_id, page, ordinal, type, section_path, bbox, text, norm_text, confidence |
| `tables` / `table_cells` | table: id, version_id, page, bbox, caption, n_rows, n_cols; cell: table_id, row, col, row_span, col_span, is_header, text, bbox |
| `images` | id, version_id, page, bbox, kind, ocr_text, description |
| `units` (v1 `chunks`, extended) | + version_id, unit_type, block_from, block_to, section_path, context_text |
| `mentions` | id, version_id, span (block_id, char_start, char_end), type, surface, value_norm, entity_id, confidence, extractor |
| `entities` (v1, extended) | + canonical key, aliases, tenant-level `entity_links` table for cross-document identity |
| `values` | id, version_id, span, kind (date, money, quantity, percent, duration, identifier, ...), normalized columns (value_date, value_num, unit, value_text), role |
| `fields` (v1, extended) | + pack, value_id, span |
| `relations` | id, version_id, subject (kind, ref), predicate, object (kind, ref), qualifiers (jsonb), span, confidence, extractor |
| `classifications` | id, version_id, target (document, section, block), label, pack, confidence, method |

Large per-page data (word boxes) is stored compactly and is never queried row by row. Everything that retrieval
filters on is a typed, indexed column.

## Invariants (tested)

1. Every annotation has a span, and the span's text equals the annotation's surface text.
2. The concatenated block text of a page equals the page's canonical text (no content lost between layers).
3. Every unit's text is the concatenation of its blocks (plus declared context), and no block is missing from
   every unit.
4. Every row belongs to exactly one tenant, which is the document's tenant.
5. Reprocessing the same file with the same versions produces the same canonical content.
