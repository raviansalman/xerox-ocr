# Canonical document model

Status: design for review. The current system has no document model: Milvus stores text chunks with a handful of
fields (below) and everything else is discarded (KD-DATA-01, KD-OCR-02).

Goals: one representation that preserves structure and provenance, from which every index (lexical, vector,
structured, visual) is derived and can be rebuilt; nothing extracted without page/region, method, version and
confidence; reprocessing never destroys previous provenance.

## 1. Identity and versioning

| Concept | Identity | Notes |
|---|---|---|
| Tenant | `tenant_id` | From the authenticated identity only |
| Collection | `collection_id` (optional) | Generalizes today's `bucket_id` / `connection_id` / `path` scoping; carries ACL groups |
| Document | `document_id` (UUIDv7) | Stable across re-uploads of the same logical document (client-supplied external id maps to it) |
| Document version | `version_id` | One per distinct content hash (`sha256`); re-uploading identical bytes is a no-op |
| Processing run | `run_id` | One per (version, pipeline version); OCR/extraction re-runs create new runs |
| Page | (`version_id`, `page_number`), 1-based | |
| Block / table / region / chunk | stable ids derived from run + ordinal | |

A document has one **active** version and, per version, one **promoted** processing run that indexes point to.
Re-running OCR with a better engine creates a new run; it is promoted only after its quality checks pass, and the
old run stays queryable for audit until retention removes it.

## 2. Structure

```
Document (tenant, collection, external_id, title, doc_type, language, acl_groups)
 └── DocumentVersion (sha256, mime, size, page_count, source: upload|url|connector, original_ref)
      └── ProcessingRun (pipeline_version, started/finished, status, per-stage versions and timings)
           ├── Page (number, width, height, unit, rotation, kind: native|scanned|mixed|image,
           │        ocr_engine, ocr_confidence, image_ref, language)
           │    ├── Block (type: heading|paragraph|list_item|table|figure|header|footer|caption|form_field,
           │    │          bbox, reading_order, text, confidence)
           │    │    └── Line / Word (text, bbox, confidence)            [stored compactly per page]
           │    ├── Table (bbox, rows, cols) └── Cell (row, col, span, text, bbox)
           │    └── Region (type: signature|handwriting|stamp|logo|checkbox|image, bbox, confidence, detector)
           ├── Chunk (text, page_start/page_end, block_ids, char_start/char_end, heading_path, token_count)
           ├── Classification (label, confidence, method, model_version, is_active)
           ├── EntityMention (type: person|organization|location|date|amount|identifier|email|phone,
           │                  surface, normalized, page, span|bbox, confidence, method)
           ├── Field (name, value_text|value_num|value_date|value_bool, unit/currency, page, span|bbox,
           │          confidence, method, extractor_version)
           └── Clause (clause_type, clause_ref, heading, page_start/page_end, block_ids, confidence)
```

Coordinates: per page, origin top-left, unit = PDF points for native pages and pixels at a recorded DPI for
images, stored with the page so boxes can be drawn on the original. Text spans are character offsets into the
page text produced by a defined reading order.

Text forms, stored per block and chunk (`INGESTION_ARCHITECTURE.md` section 6):

* `text`: as extracted (never rewritten);
* `text_norm`: NFKC, case-folded, whitespace collapsed, punctuation-normalized, Arabic letter normalization;
* `text_ocrfold`: `text_norm` with OCR confusions folded (only used for OCR-tolerant matching).

## 3. Confidence semantics

* Values in `[0, 1]`, produced by the component that made the claim, with the method recorded
  (`rule`, `model:<name>@<version>`, `llm:<name>@<version>`, `human`).
* Thresholds are configuration per field/class (`high`, `review`); consumers read the band, not raw numbers.
* Human corrections are stored as new values with `method=human` and win over machine values.
* A missing value is absent, never a guessed default.

## 4. Relational schema sketch (PostgreSQL)

Illustrative; final DDL is part of S2/S4 with migrations. Every table has `tenant_id` and row-level security.

```sql
create table documents (
  document_id uuid primary key, tenant_id text not null, collection_id uuid, external_id text,
  title text, doc_type text, doc_type_confidence real, language text, acl_groups text[] default '{}',
  active_version_id uuid, status text not null, created_at timestamptz, updated_at timestamptz,
  unique (tenant_id, external_id)
);
create table document_versions (
  version_id uuid primary key, document_id uuid references documents, tenant_id text not null,
  sha256 char(64) not null, mime text, size_bytes bigint, page_count int, source text, original_ref text,
  promoted_run_id uuid, created_at timestamptz, unique (tenant_id, document_id, sha256)
);
create table processing_runs (
  run_id uuid primary key, version_id uuid references document_versions, tenant_id text not null,
  pipeline_version text, stage_versions jsonb, status text, error jsonb, started_at timestamptz, finished_at timestamptz
);
create table pages (
  run_id uuid, page_number int, tenant_id text not null, width real, height real, unit text, dpi int,
  kind text, ocr_engine text, ocr_confidence real, language text, image_ref text, words jsonb,
  primary key (run_id, page_number)
);
create table blocks (
  block_id text primary key, run_id uuid, tenant_id text not null, page_number int, type text,
  reading_order int, bbox real[4], text text, text_norm text, text_ocrfold text, confidence real,
  tsv tsvector, -- lexical index (generated from text_norm with the language-specific configuration)
  heading_path text[]
);
create table chunks (
  chunk_id text primary key, run_id uuid, document_id uuid, tenant_id text not null, page_start int, page_end int,
  block_ids text[], text text, text_norm text, text_ocrfold text, tsv tsvector, vector_ref text, embed_model text
);
create table fields (
  id bigserial primary key, run_id uuid, document_id uuid, tenant_id text not null, name text,
  value_text text, value_num numeric, value_date date, value_bool boolean, unit text,
  page_number int, span int4range, bbox real[4], confidence real, method text, extractor_version text,
  is_active boolean default true
);
create table entities (... same provenance columns ..., type text, surface text, normalized text);
create table clauses  (... same provenance columns ..., clause_type text, clause_ref text, block_ids text[]);
create table regions  (... same provenance columns ..., region_type text, page_number int, bbox real[4]);
create table classifications (... provenance ..., label text, confidence real, is_active boolean);
create table audit_events (id bigserial, tenant_id text, actor text, action text, target jsonb, at timestamptz);
create table outbox (id bigserial, tenant_id text, topic text, payload jsonb, created_at timestamptz, done_at timestamptz);
```

Indexes: GIN on `tsv` (per-language configurations), `gin_trgm_ops` on `text_ocrfold` and identifier fields,
B-tree on `(tenant_id, name, value_*)` for fields, `(tenant_id, doc_type)`, `(tenant_id, document_id)`.

## 5. Vector index mapping (Milvus)

Milvus holds one row per chunk with only what retrieval needs: `chunk_id` (primary key), `tenant_id`
(partition key in a future collection), `document_id`, `collection_id`, `acl_groups` (array, for filtering),
`page_start`, `embedding`, `embed_model`. Text and metadata live in Postgres; the vector hit is joined back by
`chunk_id`. The current collection keeps working unchanged until the S6 cut-over.

Current fields and their destination:

| Current Milvus field (`ultimate_document_chunks`) | Destination |
|---|---|
| `id` (truncated to 95 chars, KD in ASSESSMENT L) | `chunk_id` (no truncation) |
| `embedding` | same |
| `chunk_id`, `text` | `chunks` |
| `page_number` (always 0 today) | `chunks.page_start/page_end` (real values) |
| `element_type` | `blocks.type` |
| `source_file`, `file_id`, `filename`, `original_filename` (dynamic) | `documents.external_id`, `title`, version `original_ref` |
| `object_id` | dropped after mapping review |
| `user_id` | `tenant_id` |
| `bucket_id`, `path`, `connection_id` | `collection_id`, `source_path`, connector id (compatibility mapping) |
| `created_at` (string) | timestamptz |
| `[FILE: name]` header injected into chunk text | not in stored text; filename is a separate indexed field |

Image collection (`ultimate_image_vectors`, 512 dims): re-evaluated in S11 (visual capabilities).

## 6. Chunking rules

* Built from blocks in reading order, never across page boundaries unless a block spans pages; tables become their
  own chunks (header row repeated); headings become `heading_path` context, not repeated text.
* Size in **tokens of the active embedding model**, with a hard maximum below the model window (fixes KD-EMB-02).
* Each chunk records its blocks and character spans, so any hit can be highlighted on the page.

## 7. Example (abridged)

```json
{
  "document_id": "0190f0c2-...", "tenant_id": "t-acme", "doc_type": "invoice", "doc_type_confidence": 0.97,
  "version": {"sha256": "9b1f...", "mime": "application/pdf", "page_count": 1, "original_ref": "s3://die-originals/t-acme/9b1f..."},
  "run": {"pipeline_version": "2027.1", "stage_versions": {"ocr": "tesseract-5.3.4+ara,eng", "layout": "rules-1", "fields": "invoice-rules-3"}},
  "pages": [{"page_number": 1, "kind": "scanned", "ocr_confidence": 0.91, "language": "en"}],
  "fields": [
    {"name": "invoice_number", "value_text": "INV-2026-00481", "page_number": 1, "bbox": [72, 96, 210, 110], "confidence": 0.99, "method": "rule"},
    {"name": "total_amount", "value_num": 418750.00, "unit": "SAR", "page_number": 1, "confidence": 0.94, "method": "model:fields@3"},
    {"name": "bill_to", "value_text": "Saudi Aramco", "page_number": 1, "confidence": 0.88, "method": "model:ner@2"}
  ],
  "regions": []
}
```

The example values come from the synthetic golden tenant (`tests/golden`), not from production code.
