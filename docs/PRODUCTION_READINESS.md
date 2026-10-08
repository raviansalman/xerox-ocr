# Production readiness report

Validated commit: `a9dee42` on `main` (2026-10-08). Everything below was measured or tested on that commit unless a
line says otherwise. Raw benchmark output is in [benchmarks/](benchmarks/).

## Verdict

**Not "enterprise production ready".** The engineering is sound and verified: every claimed capability works
in tests, on a clean machine and at 50,000 documents. Two things still stand in the way:

* Accuracy on real documents has never been measured.
* One machine handles about 15 to 21 questions a second, with p95 above one second from 8 concurrent users.

It is ready for a **controlled pilot**: one host, a known group of users, and real documents labelled and run
through the evaluation harness before anyone relies on the answers.

| | Status |
|---|---|
| **Engineering capability** (does the machinery do what it says) | Verified: 463 tests pass with the real model; the clean deployment passes 23 of 23 steps; the 50k benchmark completes with 0 failures; tenant leakage is 0 |
| **Real-world accuracy** (are the answers right on your documents) | **Unknown.** The only labelled set is synthetic: 80 questions written alongside the engine. It catches regressions but cannot predict quality on real contracts, invoices or scans. |

## The previous claims, re-verified

| Claim | Found | Now |
|---|---|---|
| Canonical model | VERIFIED; stale detection ignored the OCR engine and the PDF parser | fixed |
| Ingestion of all listed formats | PARTIALLY: vector-index errors failed documents permanently; image size checks applied only at twice the limit; frames and sheet rows were dropped silently; OCR had no time limit; a reprocess during processing was lost | fixed, each with a test |
| Eight retrievers | VERIFIED | |
| Exact search without vectors | PARTIALLY: v2 was correct, but the v1 and shadow engines still called the embedder | fixed; CI runs the whole suite with semantic search off |
| Deterministic calculations | PARTIALLY: amounts without a currency could enter totals; grouped counts ignored the question's text; reversed date ranges matched nothing | fixed |
| Evidence spans | PARTIALLY: results whose evidence failed verification were still shown; filtered sets above the id limit were searched only partly | fixed |
| Abstention | VERIFIED on the synthetic set (no answers to unanswerable questions) | |
| Grounded answers | VERIFIED for the extractive provider under v2; broken with the v1 engine | fixed. The Claude provider is tested only against a fake API server (no key here) |
| Prompt-injection protection | VERIFIED by tests with a fake model server | real-model behaviour not measured |
| 54 adversarial tests, zero leakage | VERIFIED | now 55 tests, plus 160 cross-tenant evaluation queries: 0 leakage |
| Hardcoded data removed | PARTIALLY: the scanner covered only `docintel/`; git history still holds old host details | scanner covers every tracked file; history needs the owner's decision (below) |
| 420 tests | PARTIALLY: the count depended on the mode (409 / 407 / 419) | 453 / 449 / 463 |
| 23-step Docker workflow | PARTIALLY: run only locally, without the embedder image. On a clean machine it **failed**: the API and worker raced to initialize the shared volume, and after a PostgreSQL restart the worker's first jobs failed and their documents stayed queued | fixed; **23 of 23 on a fresh CI runner** |
| Embedder image builds reproducibly | NOT VERIFIED | VERIFIED: built from the repository on a clean runner in every CI run |
| 50,000 documents | PARTIALLY: semantic off only | VERIFIED with semantic search on |
| About 590 documents per minute | VERIFIED for its mix (2,000 documents, 10% scanned) | 650 per minute at 50,000 documents with 2% scanned |
| 80 to 280 ms | PARTIALLY | true at one user for identifiers, typos, filters, counts, phrases, keywords and meaning; concept questions (498 ms), lookups (335 ms) and totals (359 ms) are above it |
| Concept questions 0.6 to 1.2 s at 8 users | NOT VERIFIED under realistic load | at 8 users with all question kinds mixed, concept p95 is 1.9 s (semantic on) and 1.6 s (off). The old figure measured one kind at a time. |

## Defects found and fixed in this validation

Each fix has a test that fails without it.

**Retrieval and answers**

* The v1 and shadow engines used vectors when semantic search was off.
* Large filtered sets were searched only partly.
* Unverified results were shown.
* Grouped counts ignored the question's text.
* Totals could mix in amounts with no currency.
* Reversed date ranges matched nothing.
* v1 grounded answers failed.
* Fusion was quadratic: 670 ms at 20,000 candidates, now 51 ms.
* A document with no text could not be found by its file name.

**Ingestion and recovery**

* Vector-index outages and database errors during processing failed documents permanently; both are now retried.
* After a database restart the connection pool took 32 s or longer to recover: it reconnected each dead connection with a doubling back-off. Recovery now takes 10 ms.
* In thread mode a reprocess was dropped, and the rerun after it lost its job row.
* With several processes in thread mode, each process reprocessed the other's backlog: 11% of the 50k corpus ran twice. Thread mode is now held to one process by a database lock.
* Worker child processes lost all their log output.

**Safety limits**

* Images were refused only at twice the pixel limit.
* Truncation of frames and sheet rows was silent.
* Previews were unbounded.
* OCR had no time limit.
* A LibreOffice timeout left helper processes behind and could overrun.
* Request bodies were spooled before the size check.
* URL downloads could be held open by a server trickling bytes, and followed environment proxies.

**Deployment**

* The volume initialization race on a new host.
* The stand-in test models were accepted in production.
* Containers kept Linux capabilities.
* The CI token could write, and actions were not pinned.

## Capabilities

| Verified | Partially verified | Not supported |
|---|---|---|
| PDF (native, scanned, mixed), images incl. multi-page TIFF, DOCX/DOC/ODT, XLSX/XLS/ODS/CSV, PPTX/PPT/ODP, EML with attachments, HTML, RTF, text | Claude-written answers (fake server only) | Handwriting |
| Exact phrase, identifier in any spelling, typo and OCR-tolerant, keyword, semantic, concept, entity, metadata filters | Accuracy on real documents | OCR beyond the configured Tesseract languages (English by default) |
| Counts, sums and averages per currency, group-bys, period comparisons, field lookups, each with its records | Prompt-injection resistance of a real model | Outlook `.msg` and archives (`.zip`, `.7z`, `.tar`): rejected with a reason |
| Evidence with page and character span; abstention | Latency targets under load (see below) | High availability: compose runs one PostgreSQL and a standalone Milvus |
| Tenant isolation (row-level security, vector partitioning, adversarial tests) | | |
| Recovery from restarts of PostgreSQL, Milvus, Redis and the embedder; reprocessing; deletion from every index | | |

## Results

| Area | Result |
|---|---|
| Test suite, stand-in vectors | 453 passed, 11 skipped (mode-specific), 0 failed |
| Test suite, semantic search off | 449 passed, 15 skipped, 0 failed |
| Test suite, real `all-mpnet-base-v2` | 463 passed, 1 skipped, 0 failed |
| Unit tests (no services) | 211 passed |
| Security: adversarial tests | 55 passed |
| Security: bandit | no findings |
| Security: pip-audit | no known vulnerabilities |
| Security: hardcoded-data scan | clean, whole repository |
| Tenant leakage | 0, in 55 adversarial tests and 160 cross-tenant evaluation queries per mode |
| Evaluation, v2, real model | 80 of 80; MRR 1.0; exact recall 100%; recall@10 99.2%; computed answers 100%; 0 false answers to unanswerable questions |
| Evaluation, v2, semantic off | 73 of 73 (7 meaning-only questions skipped); same exact, computed and abstention results |
| Evaluation, v1 (reference) | 77 of 80 (real model); 69 of 73 (semantic off) |
| Migrations | clean install; upgrade from the v1 schema with backfill; rollback; four processes migrating an empty database at once (`tests/integration/test_migrations.py`) |
| Docker build | API and embedder images built from the repository on a fresh runner |
| Clean deployment | 23 of 23 end-to-end steps |
| Dependencies | pinned by range, pip-audit clean; base images Ubuntu 24.04 by tag |

**Performance** (one 4-core, 16 GB VM running everything, 50,000 documents; details in
[OPERATIONS.md](OPERATIONS.md#performance)):

* **Ingestion:** 650 documents per minute with real embeddings, 2% scanned, 0 failures.
* **One user, p50 / p95, semantic on:**

  | Question kind | p50 / p95 (ms) |
  |---|---|
  | Identifiers | 84 / 219 |
  | Phrases | 172 / 215 |
  | Keywords | 197 / 296 |
  | Typos | 90 / 106 |
  | Meaning | 224 / 315 |
  | Concept | 498 / 588 |
  | Counts | 121 / 192 |
  | Totals | 359 / 396 |
  | Filters | 91 / 164 |
  | Lookups | 335 / 417 |

* **Concurrency, mixed questions, semantic on:**

  | Users | Queries/s | p50 / p95 / p99 (ms) |
  |---|---|---|
  | 1 | 5.5 | 169 / 443 / 545 |
  | 8 | 15.5 | 380 / 1,358 / 1,700 |
  | 16 | 17.4 | 721 / 2,130 / 3,049 |
  | 32 | 18.6 | 1,464 / 3,770 / 4,809 |
  | 64 | 18.4 | 2,967 / 7,222 / 9,015 |
  | 100 | 17.4 | 5,052 / 10,287 / 12,305 |

  There were 0 errors at every level. With semantic off, throughput peaks at 21 queries per second.
* **Bottleneck:** CPU, mostly PostgreSQL. From 8 users the host is at 87 to 98% CPU. PostgreSQL uses about 2 cores, the API 1 to 1.3, the query embedder 0.4 and Milvus under 0.1. Database connections are not the limit.

## Remaining limitations

* **Real-world accuracy is unmeasured.** Label a few hundred real questions over real documents. The harness takes them as YAML with no code changes; see [EVALUATION.md](EVALUATION.md).
* **Capacity:** about 15 to 21 questions per second per 4-core host; p95 stays under 500 ms only for a single user. Scale PostgreSQL first.
* **Large filtered sets** are passed to queries as id arrays. This is correct but grows with the filter's size; a semi-join would scale better.
* **Generated-answer verification is lexical:** a sentence that negates its source can pass. The extractive provider quotes the source and is not affected.
* **Thread mode** is single-process by design. Use Celery mode for several API processes.
* **A process killed mid-document** leaves a `building` version row behind. The document itself recovers; the row is cosmetic.
* **Operations still to do before production:**
  * High availability for PostgreSQL and Milvus.
  * Scheduled backups.
  * TLS termination.
  * Base images pinned by digest and a dependency lock file.
  * A separate migration role.
  * API keys in a session rather than browser storage.

## Decisions for the repository owner

1. **Git history** still contains public IP addresses and SSH targets of the hosts the earlier service ran on, from commits that predate v2. Rotating those hosts' keys and addresses is needed whatever you decide. Removing the details from history needs a rewrite and a force-push, which is your call. Keep the repository private meanwhile.
2. **Credentials** that were ever in the old service's files should be rotated.
3. **The old branch** `claude/inspiring-bohr-3kphnv`: a force-push to it was blocked earlier; delete it or leave it as it is.

## Reproduce

```bash
git checkout a9dee42
pip install -e ".[dev,answer]"
ruff check docintel scripts tests && python scripts/check_hardcoded.py
bandit -q -c pyproject.toml -r docintel && pip-audit
pytest -q tests/unit

# integration (services as in docs/OPERATIONS.md#testing), three ways
pytest -q
DOCINTEL_TEST_VECTORS=disabled pytest -q
DOCINTEL_TEST_EMBEDDER_URL=http://localhost:8080 DOCINTEL_TEST_EMBEDDING_MODEL=all-mpnet-base-v2 pytest -q
DOCINTEL_EVAL_REPORT_DIR=reports pytest -q tests/integration/test_evaluation.py

# clean deployment (what CI's deployment job runs)
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
python scripts/verify_deployment.py http://localhost:8000 keys.json -f deploy/docker-compose.yml --env-file deploy/.env
#   keys.json: admin keys for two empty tenants, {"alpha": "...", "beta": "..."}

# benchmarks
python scripts/load_test.py --url http://localhost:8000 --key $KEY --count 50000 --scanned 0.02 --query-concurrency 1
python scripts/concurrency_test.py --url http://localhost:8000 --key $KEY --levels 1,8,16,32,64,100 --duration 60
```
