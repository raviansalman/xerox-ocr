# Ingestion

Every supported format has a defined path into the canonical model, and every file that cannot be processed ends
`failed` with a reason, never `indexed` with missing content.

## Supported formats

The type is detected from the file's bytes (magic numbers and container contents); the extension is only a tie
breaker for plain-text formats.

| Format | Parser | Pages | Structure kept |
|---|---|---|---|
| PDF, native | PyMuPDF | PDF pages | blocks with bounding boxes, headings by font size, repeated header and footer bands, tables (`find_tables`) with cells |
| PDF, scanned or mixed | PyMuPDF render + Tesseract | PDF pages; only pages without a text layer are OCR'd | OCR paragraphs as blocks with boxes and confidence, word boxes |
| Images: PNG, JPEG, TIFF (multi-frame), GIF, BMP, WebP | Tesseract | one per frame | as scanned PDF |
| DOCX | python-docx | sections between page breaks | headings from styles, paragraphs, tables with cells |
| DOC, ODT, RTF | LibreOffice to DOCX, or RTF to text | as DOCX | as DOCX |
| XLSX, XLS, ODS, CSV | openpyxl, xlrd, LibreOffice, csv | one per sheet | a heading per sheet and a table of its rows |
| PPTX, PPT, ODP | python-pptx (LibreOffice for PPT/ODP) | one per slide | title, text frames with positions, tables, speaker notes |
| EML | email | the message, then one per attachment | headers as key-value blocks; attachments parsed by their own type (depth-limited) |
| HTML | html.parser | one | headings, paragraphs, tables; scripts and styles dropped |
| TXT | encoding detection | one | paragraphs |

Not supported, rejected at upload with a reason: Outlook `.msg` (save as `.eml`), archives (`.zip`, `.7z`, `.tar`),
other binary formats (audio, video, CAD). Password-protected PDFs and encrypted or corrupted Office files are
accepted at upload and end `failed` with the reason. Not provided:
handwriting recognition, OCR for scripts other than the configured Tesseract languages (`DOCINTEL_OCR_LANGUAGES`,
default `eng`), chart or image understanding beyond OCR of their text.

## What processing produces

For each document version: pages; typed blocks (`heading`, `paragraph`, `table`, `key_value`, `list_item`, `notes`)
whose concatenation is the page text; tables with cells, header rows and numeric values; retrieval units; fields
(dates by role, amounts with currency, identifiers, jurisdiction, parties, signer, signature mark), entities,
clauses and relations, each with page, block and character span; language; document type with confidence and
method; and a signature-mark flag for scans. Embeddings are computed for passages (and, for documents with three or
more passages, the document as a whole) when semantic search is enabled.

## States

`queued` → `processing` → `indexed` or `failed`. A document is `indexed` only after the database rows, postings
and vectors are all written. Each run is a row in `document_versions` (`processing`, `current`, `superseded`,
`failed`) with the components that produced it. Re-uploading identical bytes returns the existing document
(`duplicate: true`). Reprocessing replaces the content in one transaction, serialized per document.

## Limits and protections

| Protection | Setting | Default |
|---|---|---|
| Upload size | `DOCINTEL_MAX_UPLOAD_BYTES` | 200 MB |
| Pages per document | `DOCINTEL_MAX_PAGES` | 2,000; a longer PDF fails, a multi-frame image keeps the first frames and says so |
| Rows per sheet | `DOCINTEL_MAX_SHEET_ROWS` | 20,000; longer sheets are cut and the document says so |
| Image and rendered page pixels (decompression bombs) | `DOCINTEL_MAX_IMAGE_PIXELS` | 120 million, checked per frame before decoding; scans and page previews are rendered at a lower resolution to stay under it |
| OCR per page | `DOCINTEL_OCR_PAGE_TIMEOUT_SEC` | 180 s |
| Request body | `DOCINTEL_MAX_REQUEST_BYTES` (uploads), 1 MB (other calls) | 1 GB; refused with 413 while streaming, before it is spooled |
| Office container (zip) uncompressed size, ratio and entries; member paths escaping the container | `DOCINTEL_MAX_ARCHIVE_UNCOMPRESSED_BYTES`, `DOCINTEL_MAX_ARCHIVE_RATIO` | 1 GB, 200:1, 20,000 entries |
| Attachment nesting in e-mail | `DOCINTEL_MAX_ATTACHMENT_DEPTH` | 2 |
| LibreOffice conversion | `DOCINTEL_CONVERT_TIMEOUT_SEC`; private temporary profile, no shell, own process group (a timeout stops every helper process) | 180 s |
| Whole job | `DOCINTEL_TASK_TIME_LIMIT_SEC` (Celery hard limit) | 30 min |
| Empty files | rejected at upload | |
| URL ingestion (SSRF) | `DOCINTEL_URL_FETCH_ENABLED`, `DOCINTEL_URL_ALLOWED_HOSTS` | off; see [SECURITY.md](SECURITY.md) |
| File names | path components, control characters and quotes removed | |

Nothing is dropped silently: when a limit cuts a document short (image frames, sheet rows, skipped e-mail
attachments) the document is still indexed and its `error` field lists what was left out.

A vector index or embedding service outage does not fail a document permanently: the job stays queued, the
document shows the reason, and the job is retried (Celery retries, the in-process retry, then the reaper).
Vectors are written by upsert on the unit id and stale units of an earlier version are removed afterwards, so
concurrent or repeated processing of one document converges on one copy.

Temporary files live in a per-job directory that is removed afterwards. Parsers run in the worker process; a hung
conversion is killed by its timeout and the job by the task limit.

## OCR

Tesseract is behind an interface (`docintel.processing.ocr.get_ocr`) that returns words with boxes and
confidence and paragraphs as blocks; its name, version and languages are recorded in each document version, so changing the
OCR engine or its languages marks OCRed documents stale (as does a new PDF parser version for PDFs). Common OCR confusions (`rn`/`m`, `l`/`I`, `0`/`O` inside words) are folded in a
separate index form, so `Califomia` finds `California` and the reverse.

## Bulk ingestion and reprocessing

```bash
docintel ingest /path/to/folder --tenant T --wait     # register every file and process it; re-running skips duplicates
docintel reprocess --tenant T --stale --wait          # rebuild documents produced by another pipeline, pack set or model
```
