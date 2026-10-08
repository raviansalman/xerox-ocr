# Ingestion and document processing architecture

Status: design for review. Current behaviour and defects are referenced by KD id (`ASSESSMENT.md`).

## 1. Today (short)

`POST /process` (URL) or `/process-file` (upload) → Celery task on a queue chosen by file type and size →
`DocumentProcessor.process_document` (format-specific extraction, Tesseract OCR for images and scanned PDFs) →
`UltimateVectorIntegration.upsert_document` (character chunking with a `[FILE: name]` header, embedding via the
embedder service, Milvus insert) → temporary file deleted. Defects that shape this design: originals deleted,
no content hash (duplicates), page numbers lost, page order random, mixed PDFs skip OCR, English-only OCR in the
pipeline, fake OCR confidence, text cleaning that corrupts words, failures indexed as success, extracted metadata
discarded (KD-DATA-01/02, KD-OCR-01 to 09, KD-OPS-07).

## 2. Pipeline

```
1 Intake        authz (uploader role, tenant from identity) · size/type limits · SSRF-guarded URL fetch
2 Preserve      stream to object storage under tenant prefix · sha256 · registry: document/version (received)
3 Dedupe        same (tenant, document, sha256) → no-op, return existing version (explicit "duplicate" status)
4 Detect        true type by magic bytes, not extension · encryption/password · corrupted file → failed(reason)
5 Parse         per-format parser → pages with native text, text layer quality, embedded images
6 Page triage   per page: native | scanned | mixed | image-only  (text-layer coverage + rendering check)
7 OCR           only where needed, per page, language-aware, engine via OcrEngine, words + boxes + confidence
8 Layout        blocks, reading order, headings, tables, header/footer removal → canonical document
9 Understand    plugins: language, classification, entities, fields, clauses, visual regions (all with provenance)
10 Chunk        structure-aware, page-anchored, token-sized for the active embedding model
11 Index        outbox events → lexical (Postgres), vector (Milvus), structured (Postgres) → status indexed
12 Report       per-stage status, timings, versions, warnings (e.g. "page 7 OCR confidence 0.42")
```

Stages 5 to 11 run as jobs; each stage is idempotent for a given `run_id`, so retries do not duplicate rows.

## 3. Formats

| Format | Parser approach | Current status (CAPABILITY_MATRIX) |
|---|---|---|
| PDF (native) | Text with positions per page (PyMuPDF), images extracted | WORKING without positions |
| PDF (scanned / mixed) | Per-page triage; OCR pages with poor or missing text layer | Scanned WORKING, mixed BROKEN |
| Images (PNG, JPEG, TIFF incl. multi-page, BMP, WEBP, GIF) | Render each frame as a page; OCR | PNG/JPEG WORKING, others UNKNOWN |
| DOCX / DOC | DOCX: paragraphs, styles (headings), tables as tables; DOC: convert (LibreOffice headless) then DOCX path | DOCX WORKING (tables flattened), DOC UNKNOWN |
| XLSX / XLS / ODS / CSV | Sheets as tables; row/column provenance; size caps | UNKNOWN (code exists, untested) |
| PPTX / PPT / ODP | Slides as pages, shapes as blocks, notes optional | UNKNOWN |
| TXT / MD / HTML / RTF | Text; HTML sanitized, headings from tags | TXT/HTML WORKING |
| Email (EML, MSG) | Headers as fields, body as text, attachments as child documents | MISSING |
| Archives (ZIP) | Optional: children as documents, with limits against zip bombs | MISSING (decide) |

Every parser returns the same canonical structures (`DOCUMENT_MODEL.md`); format detection never trusts the
extension alone.

## 4. Preservation and storage

* Originals are written once to object storage at `tenant/<tenant_id>/originals/<sha256>`; processing reads from
  there, never from a temporary path that gets deleted.
* Page renders and OCR artefacts are stored per run (`tenant/<tenant_id>/runs/<run_id>/...`) so evidence can show
  the page image with boxes.
* Deletion is an explicit, authorized, audited operation that removes registry rows, index entries and objects
  (and is verified by a test that searches afterwards).
* Retention policies are configuration per tenant.

## 5. OCR abstraction

```python
class OcrEngine(Protocol):
    name: str; version: str
    def capabilities(self) -> OcrCapabilities: ...        # languages, handwriting, boxes, confidence
    def recognize(self, page: PageImage, hints: OcrHints) -> OcrPage: ...   # words, lines, boxes, conf, lang
```

* First engine: Tesseract 5 with `eng+ara` (and per-tenant language lists), `image_to_data` for words, boxes and
  real confidences (replaces `image_to_string` and the synthetic confidence, KD-OCR-08).
* Preprocessing (deskew, denoise, binarize, DPI normalization) is a separate, versioned step; its parameters are
  stored with the run.
* Language detection per page/block on native text, and on a first OCR pass for scans to choose the final
  language set; mixed-language pages keep per-block language.
* Handwriting: separate capability flag; an engine that does not support it reports so. Handwriting is not
  claimed until an engine is integrated and evaluated on labelled pages.
* Other engines (PaddleOCR, docTR, cloud OCR where allowed, vision-language models on-prem) plug in behind the
  same interface; results are compared with `EVALUATION_STRATEGY.md` OCR metrics (CER/WER on labelled pages)
  before promotion.
* Raw OCR text is stored unmodified. Normalization and OCR-confusion folding are derived fields only (fixes
  KD-OCR-01/07 by design).

## 6. Text normalization (single implementation)

One `normalize()` module used by ingestion **and** queries (today there are three inconsistent versions, F7/F8):
NFKC, case folding, whitespace collapse, punctuation normalization (quotes, dashes), Arabic letter normalization
(alef variants, ta marbuta, alef maqsura, diacritics, tatweel), digit normalization (Arabic-Indic to ASCII in the
derived field), identifier canonical forms (`INV-2026-00481` → `inv202600481`), and the OCR fold used only for
`text_ocrfold` (`rn→m`, `vv→w`, digit/letter confusions inside alphabetic tokens). Versioned; changing it triggers
reindexing of derived fields, not re-OCR.

## 7. Understanding plugins

```python
class Extractor(Protocol):
    name: str; version: str
    applies_to: set[str]                                 # doc types or "*"
    def extract(self, doc: CanonicalDocument, ctx: ExtractionContext) -> list[Finding]: ...
```

`Finding` = entity mention, field, clause, classification or region, always with page, span/bbox, confidence,
method. Plugins are configured per deployment (taxonomy, field schemas per document type) from data, not code.
First set:

| Plugin | Approach | Notes |
|---|---|---|
| Classifier | Rules on title/first page → embedding prototypes → optional LLM | Configured taxonomy, confidence bands |
| Dates and amounts | Deterministic parsers (multi-locale, Hijri dates optional) | High precision |
| Identifiers | Configurable patterns per tenant (invoice numbers, contract numbers) plus generic letter-digit shapes | |
| Entities | NER model (multilingual) + gazetteers as data | No hardcoded names |
| Fields per type | Rules + model/LLM with schema validation; value must be found in the page text or region | Hallucination guard: extracted text must have a source span |
| Clauses | Heading/numbering segmentation + clause type classifier | S12 |
| Visual regions | Signature, stamp, checkbox, logo detectors on page images | S11, pluggable |

## 8. Job orchestration

* Celery queues by stage and resource profile (parse, ocr, understand, index), not by customer shard; size-based
  routing kept for very large files.
* Job state machine per run: `received → parsing → ocr → layout → understanding → chunking → indexing → indexed`,
  with `failed(stage, reason)`, `partial(warnings)`, `duplicate`, `skipped(reason)` as explicit terminal states
  (fixes KD-OCR-04 and KD-OPS-07).
* Outbox table written in the same transaction as canonical rows; index consumers are idempotent by
  `(chunk_id, index, version)`. Postgres and Milvus therefore converge after crashes; a reconciliation job
  compares counts and tenants.
* Back-pressure and per-tenant fairness: per-tenant concurrency limits instead of hash-sharded queues that no
  worker may consume (KD-OPS-05).
* Every job logs `tenant_id`, `document_id`, `version_id`, `run_id`, stage timings and model versions.

## 9. Reprocessing

* Triggered by a new pipeline or model version, a tenant configuration change, or an operator.
* Creates a new run; indexes keep pointing to the promoted run until the new one finishes and passes checks
  (page count, text coverage, OCR confidence distribution, extraction deltas within thresholds); then the pointer
  flips atomically and old index entries are removed by the outbox consumer.

## 10. Limits and safety

Max file size, page count, pixels per page, archive expansion ratio, processing time per stage; password-protected
files fail with a clear reason; HTML/Office macros never executed; parsers run without network access in workers.
