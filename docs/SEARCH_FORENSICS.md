# Search forensics: the factual baseline before any search change

Milestone scope: establish what search actually does today, with evidence. **No search code was changed.**
Nothing was restored. The only code added is test and tooling code:

* `ultimate/tests/golden/cases.py`: golden corpus (73 documents, 4 tenants) and 45 graded cases
* `ultimate/tests/integration/test_golden_search.py`: runs every case in every mode through `POST /search`
* `ultimate/tests/golden/baseline_standin.json`, `baseline_mpnet.json`: recorded pass/fail maps
* `ultimate/scripts/forensics/compare_deployed.py`: deployed tree vs repository classifier (read-only)

Evidence labels used below: **VERIFIED** (executed and observed), **TRACED** (read in source, call path
followed), **INFERRED** (strong indirect evidence, not proven), **UNKNOWN** (needs data we do not have).

Line numbers refer to branch `claude/inspiring-bohr-3kphnv` at the commit that adds this file.

---

## Summary

1. **Production code is UNKNOWN, and there is strong evidence that it differs from the repository.** The
   repository's own `scripts/compare_staging_prod_search.py` compares staging and production responses for
   `query_meta.persons/locations/dates` and relies on a MetadataIndex disk cache. Neither can work with the code in
   this repository: `query_meta` is never populated and the MetadataIndex cannot be constructed. (INFERRED)
2. **Default search (`searchMethod=both`) is vector search only.** The semantic path is gated on
   `query_meta["needs_semantic"]`, which is never set, so it never runs. Every inspected `both`-mode result (the
   top 5 of all 45 cases) came from the vector path. (VERIFIED, both embedders)
3. **There is no exact-text retrieval path.** Exact phrases only affect ordering among the at most 50 chunks that
   vector search returns. An exact phrase in a document that is semantically far from the query cannot be found.
   With real MPNet the needle document ranked 61st of 61 by cosine and was never returned. (VERIFIED)
4. **The four regression queries pass because of a sort tiebreak, not because of exact search.** For
   "FOR IMMEDIATE RELEASE" the press release's vector score is 0.49 and the firmware note's is 0.56. The press
   release ranks first only because the sort key puts "query is a substring of the returned chunk" before the
   score. (VERIFIED)
5. **The dead code is the historical exact search, and it is unsafe to switch back on as is.** Injecting only
   `LOCATION_PEERS` in memory revived the supplements and fixed the needle case in all three modes. It also
   revived a scope bypass: bucket- and path-scoped searches returned files from other buckets. Adding `threading`
   as well produced **cross-tenant leaks in 19 case/mode runs**. A tenant with no documents (`carol`) received two
   other tenants' documents. (VERIFIED, in-memory experiment, not committed)
6. **A third missing import was found:** `query_enhancement.py` uses `os` without importing it, so
   `known_organizations.json` never loads, even if `LOCATION_PEERS` came back. (VERIFIED)
7. **Tenant isolation holds on every live path today.** 45 cases × 3 modes × 2 embedders: zero foreign-tenant
   results. Four `bucketId`/`connectionId` injection payloads returned nothing foreign. (VERIFIED)

---

## 1. Current production search architecture

**UNKNOWN.** There is no access to the deployed containers from this session. The production hosts belong to the
previous team (StorageChain). Calling their public endpoints was deliberately not attempted: they need a real
tenant id and would touch customer data.

What the repository says about production:

| Evidence | What it shows | Label |
|---|---|---|
| `scripts/compare_staging_prod_search.py` (docstring and `_summarize_meta`) | Production and staging return `query_meta` with `persons`, `locations`, `organizations`, `date`, `date_range`, `vector_query`. The parity checklist says the search container loads MetadataIndex from `data/metadata_cache/` | INFERRED: production ran `enhance_query` successfully (so `LOCATION_PEERS` existed) and a working MetadataIndex (so `threading` was imported) |
| Same script, "main-byoc-v1 style" | A branch or deployment named `main-byoc-v1` existed | INFERRED |
| `universal_deploy.sh:209-210`, `sync_ultimate.sh:77-78, 155` | Production configuration `docker_byoc.env` and `docker_byoc_staging.env` exist outside git | TRACED |
| `docs/FORENSICS.md` F3 | Deployments rsync the developer's working tree and `docker cp src/` into running containers | TRACED |
| Zip comment `e97b41f…` | The snapshot we have is the committed `main` | VERIFIED (earlier milestone) |

Conclusion: production very probably ran a version with the full query-understanding stack and the MetadataIndex.
If it did, then per section 8 production also had the shared-index cross-tenant exposure. **That is the first
question to put to the previous team.**

## 2. Repository search architecture (what runs today)

`POST /search` → `_search_blocking` (`ultimate_ui.py:3043`). Live stages, in order:

| # | Stage | Where | Live? |
|---|---|---|---|
| 1 | Temporal phrasing normalization | `:3047` | LIVE |
| 2 | `vector` → `both` auto-upgrade on detected intent | `:3058-3077` | DEAD (needs `enhance_query`) |
| 3 | `enhance_query` → `query_meta` | `:3094-3098` | DEAD: ImportError, replaced by `{original_query, normalized_query}` |
| 4 | `vi._last_query_meta = query_meta` (shared singleton) | `:3117-3122` | LIVE (harmless today, KD-SRCH-03) |
| 5a | `both`: vector and semantic in a 2-thread pool; semantic only if `needs_semantic` | `:3299-3335` | **Vector only.** `needs_semantic` is never set |
| 5b | `vector` / `semantic`: the one path, 60 s timeout | `:3336-3367` | LIVE |
| 6 | Merge, dedupe by normalized file id, +0.15 for semantic hits | `:3369-3436` | LIVE |
| 7 | Constraint prune (required keywords, regex, city anchors) | `:3440-3524` | DEAD (no keywords) |
| 8 | Score floor `SEARCH_MIN_SCORE` (0.0 code default; 0.05 in `.env.example`) | `:3526-3557` | LIVE; nothing is ever filtered at 0.0 |
| 9a | `vector` mode: `apply_constraint_boost`, then sort | `:3580-3593` | Boost DEAD (ImportError), sort LIVE |
| 9b | `both`/`semantic`: validators, NDA supplement, filename supplement, content-scan supplement, entity supplement, constraint boost, dedupe, trims | `:3595-4368` | **All DEAD.** The block starts with `enhance_query` (`:3600`) and falls to the `except` at `:4369` |
| 10 | Sort by `(phrase_match, filename_token_hits, score)` | `:3561-3577`, `:4374` | LIVE: this is the only "exact" signal |
| 11 | Zero-result fallbacks: vector threshold 0, year scan, filename lexical scan | `:4380-4464` | LIVE, only when nothing was returned |
| 12 | Texas peer prune | `:4466` | LIVE but returns early (no anchors) |
| 13 | Response, including internal `query_meta` | `:4469-4511` | LIVE |

**Vector path** (`UltimateVectorIntegration.search_documents`, `src/ultimate_vector_integration.py:879-1028`), the
engine behind `both` and `vector`:

1. Embed the raw query via `EMBEDDER_SEARCH_URL` (fallback `EMBEDDER_URL`). No query cache.
2. Milvus HNSW search on `embedding`, COSINE, `expr = user_id == "<escaped>"` plus scope filters
   (`vector_db_milvus_server.py:617-645`). Candidate count `min(limit*5, 500)`, then **hard cap 50 chunks**
   (`:601-605`). `ef = MILVUS_SEARCH_EF` or `max(64, min(512, 2k))`, never below k (`:609-611`).
3. Best chunk per file (`ultimate_vector_integration.py:932-939`).
4. Term-overlap boost `+0.25 × matched/total` terms. Terms are whitespace tokens of length ≥ 2 minus 13
   stopwords, matched as **substrings** of chunk text + filename (`:942-954`). Cap at 1.0.
5. Temporal metadata enrichment of the response only (`:994-1010`).

**Semantic path** (`SemanticPipeline.search_documents`, `src/semantic/semantic_pipeline.py:1126-4196`), reached only
with `searchMethod=semantic`: one Milvus vector search (same 50 cap), aggregate per document, cross-encoder
rerank (BAAI/bge-reranker-base; 1.0 for every document if it fails to load), blend (general `0.3v + 0.7rr`, person
`0.1v + 0.9rr`), lexical floors (whole query substring ≥ 0.98, full term overlap ≥ 0.95), filename/person/signature
boosts, then many **key-term gates** (`\b` word-boundary checks on persons, orgs, years, clause terms, token
coverage with a 0.65 final floor). The metadata-first router (`:1256-1703`) and every date/location/clause
branch are dead. One more defect found here: `self.ner = None` at `:495-502` overwrites the `en_core_web_sm`
fallback, so spaCy NER is only active if `en_core_web_trf` loads.

**Lexical scan** (`search_lexical`, `src/ultimate_vector_integration.py:1281-1333`), reached only as a zero-result
fallback today: full Milvus scan (`limit=16384`, **without text**), so it matches **filenames and file ids only**.
Tiers: exact filename 0.98, phrase in filename 0.95, every token in filename 0.91. Per-process cache 90 s, keyed
by `(user_id, query, scope)` but not `limit`.

## 3. Production vs repository diff

**UNKNOWN until the deployed tree is provided.** Tooling is ready:

```bash
# on each API and worker host (the previous team)
docker exec <container> tar c -C /app src ultimate_ui.py search_api.py > deployed_<host>.tar
mkdir deployed && tar xf deployed_<host>.tar -C deployed
# here
python3 -I ultimate/scripts/forensics/compare_deployed.py deployed ultimate > deployed_vs_repo.md
```

The script reports first whether `LOCATION_PEERS` and `threading` exist in the deployed tree. Then it lists files
only on one side, and for each differing file its area (search, semantic_utils, query understanding, ranking,
metadata indexing, Milvus, embeddings, chunking, OCR), the line counts, and the symbols present on only one side.
It was checked against a synthetic deployed tree. Also ask for `docker_byoc.env` / `docker_byoc_staging.env`
**with secrets removed**: search tuning (`MILVUS_SEARCH_EF`, `SEARCH_MIN_SCORE`, `SEMANTIC_STRICTNESS`,
`CHUNK_SIZE`) changes results.

## 4. Real embedding and model configuration

| Item | Value | Source | Label |
|---|---|---|---|
| Model | `sentence-transformers/all-mpnet-base-v2`, HF revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` | `embedder_service.py:21` (hardcoded; ignores `EMBED_MODEL_TEXT`) | VERIFIED |
| Weights used for this baseline | `model.safetensors` sha256 `78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658` | See note below | VERIFIED locally; cross-check on huggingface.co |
| Runtime | `src/embedder_service.py` **unchanged**, sentence-transformers 2.7.0, transformers 4.41.1, tokenizers 0.19.1, torch 2.3.1 CPU path, offline | Versions pinned in `Dockerfile.ultimate:35-41` (torch unpinned there) | VERIFIED |
| Dimensions | 768 | `/embed` output | VERIFIED |
| Normalization | L2, server side (`embedder_service.py:26-28`); client does none | measured norms 1.0000 | VERIFIED |
| Distance | COSINE | `ultimate_vector_integration.py:484`, schema tests | VERIFIED |
| Index | HNSW M=32, efConstruction=200 | `vector_db_milvus_server.py:337-343`, pinned by tests | VERIFIED |
| Search ef | `MILVUS_SEARCH_EF` (48 in `.env.example`) else `max(64, min(512, 2k))`, ≥ k | `:609-611` | TRACED |
| Max chunks per vector search | 50 | `:601-605` | VERIFIED |
| Chunking | characters, `CHUNK_SIZE` 1200 / overlap 50 (code), `CHUNK_SIZE=500` in `.env.example` | `ultimate_vector_integration.py:63-64` | TRACED. The golden run used the code default (1200); results under `CHUNK_SIZE=500` were not measured |
| Reranker | `BAAI/bge-reranker-base` | `semantic_components.py:56-63` | **NOT reproduced**: no reachable copy; the semantic mode below ran with the no-op reranker (1.0 for all). Production semantic-mode ranking is UNKNOWN |
| Sanity | cos("how do I change the toner", toner procedure) = 0.594; vs remote-work policy = −0.016 | | VERIFIED |

Note on the weights: huggingface.co is blocked by this environment's network policy. The weights came from
Weaviate's published image `semitechnologies/transformers-inference:sentence-transformers-all-mpnet-base-v2`
(digest `sha256:d13a0973b20465f3454ecebad24182325e8a9f6d7a91da69c57111a82bae6570`), which contains a standard
Hugging Face cache snapshot of the revision above. Please compare the sha256 with the file listing on
huggingface.co. If it matches, these are byte-identical weights.

## 5. Golden search baseline

Corpus (`tests/golden/cases.py`): 13 purposeful documents plus 60 distractors. They cover text PDFs, a DOCX table,
a scanned (image-only) PDF that goes through OCR, OCR-corrupted text, Arabic, near-word traps ("ink" vs
"link/blinking/inkjet"), a firmware note that contains the words of "for immediate release" but not the phrase,
and a haystack tenant with 60 heron notes plus one needle with the exact phrase "blue heron protocol 7781".
Tenants: `acme`, `globex` (repeats acme's key terms on purpose), `haystack`, `carol` (empty). Every case is also
graded for foreign-tenant documents.

Runs: 45 cases × 3 modes, ingest through the real Celery task, query through `POST /search`, limit 10. Both runs
were repeated and reproduced exactly. Mean latency per query: 0.08 to 0.09 s (local, tiny corpus).

| Category | Cases | both (stand-in / MPNet) | vector | semantic |
|---|---|---|---|---|
| Exact phrase | 5 | 4 / 4 | 4 / 4 | 4 / 4 |
| Exact keyword / identifier | 5 | 5 / 5 | 5 / 5 | 5 / 5 |
| Joined / split form | 3 | 2 / 2 | 2 / 2 | 3 / 2 |
| Word boundary | 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| Person / entity (incl. name only in OCR text) | 5 | 5 / 5 | 5 / 5 | 5 / 5 |
| Semantic paraphrase | 6 | 3 / 3 | 3 / 3 | 3 / 4 |
| OCR-corrupted query or document | 5 | 5 / 5 | 5 / 5 | 4 / 5 |
| Filename | 4 | 4 / 4 | 4 / 4 | 4 / 4 |
| Arabic | 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| No result expected | 1 | 0 / 0 | 0 / 0 | 1 / 1 |
| Tenant, scope, injection | 8 | 8 / 8 | 8 / 8 | 8 / 8 |
| **Total** | **45** | **39 / 39** | **39 / 39** | **40 / 41** |

Failures with real MPNet, and why:

| Case | Mode(s) | Got | Root cause |
|---|---|---|---|
| `phrase_needle` "blue heron protocol 7781" | all | heron notes | Needle is 61st of 61 by cosine (0.361 vs 0.635 for the 50th); 50-chunk cap; no exact retrieval path |
| `join_versa_link` "Versa Link C405" | all | AltaLink agreement first | No joined-form matching; "link" overlap boost favours the agreement |
| `sem_repair_visit` "record of a printer repair visit" | all | toner procedure | MPNet prefers the procedure over the OCR'd ticket; no metadata/document-type signal |
| `sem_media_contact` "who handles questions from journalists" | both/vector: remote-work policy; semantic: **nothing** | | Weak paraphrase; semantic mode's key-term gates drop every candidate |
| `sem_contract_length` "how long does the printer lease last" | both, vector | press release | Paraphrase; semantic mode gets it right |
| `unknown_term` "zzqxunknownzzq" | both, vector | 10 unrelated documents | Score floor 0.0 (KD-SRCH-06) |

Read these numbers with care. **The corpus is tiny (11 documents in the main tenant).** At this size vector search
plus the overlap boost finds most things, so passing cases do not prove quality at production scale. The needle
case shows what happens once a tenant has more than 50 chunks that are closer to the query than the target.

Stand-in vs MPNet: identical in `both`/`vector` (the hashing stand-in is lexical by construction). Semantic mode
differs on 3 cases. The stand-in baseline is the CI regression gate; the MPNet baseline is the quality reference.

## 6. Search capability matrix (code as in the repository)

| Capability | Status | Evidence |
|---|---|---|
| Exact literal matching (content) | **BROKEN** as retrieval, LIVE as re-ordering | Only `phrase_match` in the sort key (`ultimate_ui.py:3571`) over the ≤ 50 vector candidates. Needle case fails. The content-scan supplement that did real exact retrieval (`:3930-4100`) is dead |
| Exact literal matching (filename) | PARTIALLY WORKING | Lexical tiers 0.98/0.95/0.91 exist, but only run as a zero-result fallback (`:4439`). Filename queries pass in the golden set through vector similarity on the `[FILE: name]` chunk header |
| Normalized exact matching | PARTIALLY WORKING | Lowercase and whitespace collapse in the sort key; URL decoding of filenames. No Unicode or punctuation normalization of content; CamelCase split is dead (KD-SRCH-05) |
| Joined-form matching | MISSING | Not implemented for content. "Storage Chain" passes through embeddings only; "Versa Link" fails |
| Phrase matching | PARTIALLY WORKING | Same as exact literal: re-ordering only; requires ≥ 4 characters and the phrase inside the best chunk |
| Ordered-word matching | MISSING | No general implementation anywhere (only fixed regexes such as signature lines in the semantic path) |
| Single-word boundary matching | PARTIALLY WORKING | Semantic mode key-term gates use `\b` (LIVE). Default path uses substring overlap (`"ink"` matches `"link"`); passes the golden case only because the true document repeats the word |
| Vector similarity | WORKING | Verified with real MPNet; capped at 50 chunks |
| OCR / fuzzy similarity | MISSING | No live fuzzy code. `rapidfuzz` exists only in the dead `DocumentProcessor.search_documents` (`ultimate_search_processor.py:2787-3660`; ingestion passes `target_words=[]`). OCR-noisy golden cases pass through subword embeddings on a tiny corpus, not through fuzzy matching |
| Partial-text boosting | WORKING | `+0.25 × overlap` in the vector path (substring based); lexical floors in semantic mode |
| Metadata / entity matching | BROKEN | Query side dead (`enhance_query`), MetadataIndex dead, extracted entities not persisted (KD-DATA-01). Semantic mode has regex person/org extraction (LIVE) |
| Query understanding | BROKEN | `enhance_query` raises on every call; `known_organizations.json` would not load anyway (missing `os`) |
| Hybrid ranking | BROKEN (default) / PARTIALLY WORKING (semantic mode) | `both` never runs semantic; no score fusion with a lexical retriever exists. Semantic mode blends vector + reranker + boosts, but the production reranker could not be reproduced |
| Tenant filtering | WORKING | Milvus-side `user_id` filter on every live path; escaped; 0 leaks in 270 graded runs; 4 injection payloads harmless |
| User-scoped retrieval | WORKING | Tenant comes from the API key (or `userId` for service keys); `carol` gets nothing |
| Bucket / path / connection scoping | WORKING on the live path, BROKEN on the dead path | Live: Milvus expr (vector) or post-filter (semantic). Dead supplements ignore scope (KD-SRCH-09), verified by experiment E1 |
| "No results" | BROKEN in `both`/`vector` | Floor 0.0 returns everything (KD-SRCH-06); semantic mode returns empty correctly |

## 7. Missing and broken components

| ID | Component | Effect | Label |
|---|---|---|---|
| F1 / KD-SRCH-11 | `LOCATION_PEERS` missing from `semantic_utils.py` | `enhance_query`, `apply_constraint_boost`, the `/search` validation and supplement block, and `both`-mode semantic routing are all dead | VERIFIED |
| F2 / KD-SRCH-04 | `threading` not imported in `semantic_components.py` | MetadataIndex is never built; the metadata-first router is dead | VERIFIED |
| **F11 (new)** | `os` not imported in `query_enhancement.py` (used at `:109-110`) | `known_organizations.json` never loads; org detection would be degraded even with F1 fixed | VERIFIED (warning observed in experiment E1) |
| **KD-SRCH-12 (new)** | No exact-text retrieval path | Exact phrases outside the vector top 50 are unreachable | VERIFIED |
| **KD-SRCH-13 (new)** | `both` mode never runs the semantic path | Default search = vector search + phrase tiebreak | VERIFIED |
| **KD-SRCH-14 (new)** | spaCy fallback NER overwritten with `None` (`semantic_pipeline.py:495-502`) | Semantic mode relies on regex entities unless `en_core_web_trf` loads | TRACED |
| KD-SRCH-05 | CamelCase filename split dead | | VERIFIED earlier |
| KD-SRCH-06 | Score floor 0.0 | Unknown queries return everything | VERIFIED |
| KD-SRCH-08 | Reranker fallback = 1.0 for all | | VERIFIED |
| KD-SRCH-09 | Supplements ignore scope filters | Becomes live if F1 is fixed | VERIFIED (E1) |
| KD-SEC-09 | Shared MetadataIndex across tenants | Becomes live if F2 is fixed | VERIFIED (E2, end to end through `/search`) |
| Dead | `DocumentProcessor.search_documents` fuzzy layer, `_TERM_TYPO_VARIANTS`, `is_entity_only_query` etc. | Never called | TRACED |

### Experiments (in memory only, nothing committed)

A pytest plugin outside the repository injected the missing names into the running process, and the same golden
suite was run with real MPNet:

| | both | vector | semantic | Foreign-tenant results |
|---|---|---|---|---|
| Baseline (repository as is) | 39 | 39 | 41 | 0 |
| E1: `LOCATION_PEERS = {austin: [dallas], dallas: [austin]}` (minimal guess) | 39 | 41 | 37 | 0 |
| E2: E1 + `threading` | 34 | 38 | 31 | **19 case/mode runs** |

E1 in detail: the dead supplements are the historical exact search. `phrase_needle` passes in all three modes,
`join_versa_link` passes in `both`/`vector`, and results become tighter (often exactly one document). They also
bring **regressions**: `scope_bucket` and `scope_path` fail (files from other buckets and folders return,
KD-SRCH-09), all four injection probes now return documents because the bucket filter is bypassed (same tenant
only), and semantic mode loses `phrase_governing_law`, `kw_needle_code` and `boundary_ink`.

E2 in detail: the tenant invariant failed. Examples: `tenant_carol` (a tenant with no documents) received
`globex_release` and `press_release`; acme queries received haystack documents; haystack queries received acme
documents. This turns KD-SEC-09 from a traced risk into a demonstrated leak through the public API, in all three
modes.

## 8. Tenant-isolation analysis

Where filtering happens today, for every live path:

| Stage | Mechanism | Where |
|---|---|---|
| Before Milvus search | Tenant from the API key, `userId` only for service keys; validated against `^[A-Za-z0-9_.:@-]{1,128}$` | `ultimate_ui.py:3037-3039`, `src/security.py` |
| In Milvus (vector) | `expr = user_id == "<escaped>" and <scope>` | `vector_db_milvus_server.py:617-645` |
| In Milvus (full scans: lexical fallback, year fallback) | `user_id == "<escaped>"` plus bucket/connection | `:820-840` |
| Metadata filtering | None live (MetadataIndex dead) | |
| After retrieval | Scope post-filters on semantic results (`ultimate_ui.py:3179-3193`); app-side scope filter when the Milvus expression fails (`vector_db_milvus_server.py:679-698, 771-780`) | |
| During ranking | No tenant logic; operates on the already-filtered candidates | |
| Before the response | **No re-check of `user_id`** on returned items | |

Paths by which unauthorized data could enter the candidate set:

| # | Path | Live today? | Risk |
|---|---|---|---|
| T1 | Shared MetadataIndex object, file-id-keyed `_doc_embedding_cache` | No (F2) | **CRITICAL if F2 is fixed** (demonstrated, E2) |
| T2 | `search_similar` / fallback with a falsy `user_id` adds no filter (`:619-623, 689`) | No: the API always passes a validated tenant. Internal callers could | Medium (defence in depth) |
| T3 | `query_all_chunks(expr=...)` replaces the user filter (`:824-825`) | No caller passes `expr` | Low |
| T4 | Non-string filter values interpolated raw (`key == {value}`, `:644, 838`) | Reachable with JSON `bucketId` lists/numbers/dicts; tested: the expression fails or matches nothing, then the user-only fallback and app-side filter apply. 0 foreign results | Low today; fix by type-checking scope fields |
| T5 | Rows without `user_id` stored as `"default_user"` (`:404`) | Only via internal callers; the API never ingests without a tenant | Low; reject such rows |
| T6 | Process-wide caches: content-scan (keyed by user), lexical (keyed by user+scope), circuit breaker (shared) | Content scan dead; others keyed correctly. Breaker is availability, not data | Low |
| T7 | Supplements ignoring scope | No (F1) | Same-tenant scope bypass if F1 is fixed (E1) |

Recommended invariant for any new architecture: every candidate carries `user_id`, and the response layer drops
any item whose `user_id` differs from the request's tenant, logging it as a security event. The golden test
already enforces this externally.

## 9. Historical implementation reconstruction

What the code itself records (no git history is available):

1. **Earliest design: per-document keyword matcher.** `DocumentProcessor` (`ultimate_search_processor.py`):
   direct substring (100), `rapidfuzz.partial_ratio` (threshold 60), word variations (90), synonyms (85), regex
   patterns (80), with an OCR character-fix normalizer. It searched in-memory `UltimateSearchResult` objects, not
   an index. Ingestion still calls `process_document(target_words=[])`, so only its extraction survives. (TRACED)
2. **"Standard pipeline" (`src/api.py`, `src/search.py`, `src/chunking.py`, `src/normalize.py`).** Named in the
   README, `docker-compose.dev.yml` and `.cursor/rules/analyst.mdc` (token chunking 500/100, Presidio PII
   redaction, SHA-256 idempotency). None of these files exist. (TRACED that they are referenced; content UNKNOWN)
3. **"Ultimate" vector + semantic stack, versioned v4.0 to v7.0 in comments.** Hybrid keyword overlap boost
   (v6.0), "hyper-aggressive" lexical overlap (v7.0), temporal engine v5.x, metadata index v5.0, then
   the `/search` supplements: NDA filename supplement, filename lexical supplement ("A prior cache-based scan
   diverged from this logic", `ultimate_ui.py:3871`), content keyword scan with whole-word person matching,
   entity+extension supplement, constraint boost. This is the "exact search" layer. (TRACED)
4. **Consolidation.** Comments mark at least 13 modules merged into 4 files: `reranker.py`, `metadata_index.py`,
   `clip_utils.py`, `entity_extractor.py`, `normalize.py`, `chunking.py`,
   `validators/universal_constraint_validator.py` into `semantic_components.py`; `aggregation.py` and
   `semantic_pipeline_semantic_mode.py` into `semantic_pipeline.py`; `filename_utils.py` and
   `location_normalizer.py` into `semantic_utils.py`; `ingestion_temporal_normalizer.py` into `temporal_engine.py`.
   (TRACED)

**What removed the historical search components, and why (D):** all three missing names sit in merged code.
`threading` was needed by the section "merged from metadata_index.py". `LOCATION_PEERS` is a location concept
and `semantic_utils.py`'s section "merged from location_normalizer.py" contains only a two-city helper
(`tx_metro_snippet_has_wrong_peer_only`) that hardcodes the same Austin/Dallas peer idea. `os` is missing from
`query_enhancement.py`, which imports `normalize_location` from that same merged section. **The most probable
cause is the consolidation: the merged files lost module-level imports and constants, and the importers were not
updated.** Every failure is caught and logged as a warning, so search kept returning results, and the regression
queries kept passing thanks to the vector path plus the phrase tiebreak. Nobody would have noticed. (INFERRED:
consistent with every observation; the commit cannot be identified without history)

If the deployed tree (section 3) defines these names, production was never affected and the bug exists only in
`main`. If it does not, production has run with the exact-search layer disabled since that deployment.

## 10. Recommended next architecture (not implemented)

Decision input from the baseline:

* **Restoring the legacy names is not a fix.** It revives exact retrieval but also a same-tenant scope bypass
  (E1) and a cross-tenant leak (E2), changes results for most queries, and keeps 2,400 lines of rules written for
  another company's corpus (NDA, Austin/Dallas, named people).
* **The current live path is safe but shallow.** Vector top 50 plus a phrase tiebreak.

Recommended direction, a modular hybrid retriever inside the existing monolith (no new services):

1. **Lexical retriever over chunk text, per tenant.** Phrase, keyword, boundary, ordered words and joined forms
   are retrieval candidates, not only tiebreaks. Options, best first: (a) upgrade Milvus to 2.5 and use built-in
   BM25 sparse vectors on the same collection, filtered by `user_id`; (b) keep 2.3 and store a sparse vector
   (2.4+), or (c) a per-tenant inverted index in Redis/Postgres. Decide with the golden set and a scale test.
2. **Vector retriever** as today (MPNet, COSINE, HNSW), with the 50-chunk cap raised or paged and measured.
3. **Fusion** with reciprocal rank fusion (no score mixing across incomparable scales), then an exact-match
   precedence rule that is explicit and tested (exact phrase hits first when the query is quoted or short).
4. **Optional cross-encoder rerank** of the top N, with health reporting when it is unavailable (no 1.0 fallback).
5. **Query understanding as a small, generic, tested module** (dates, quoted phrases, file types, person names)
   behind a `DomainProfile`; the legacy StorageChain rules only in a legacy profile, if wanted at all.
6. **Tenant guard at three layers**: Milvus filter, per-tenant caches only, and a final response check.
7. **Normalization done identically at ingest and query** (one `normalize_text`, Unicode NFKC, case fold,
   de-hyphenation, joined-form tokens, an OCR confusion map for 0/O, 1/l/I, rn/m).
8. **Persist extracted metadata** (dates, entities, page numbers) so filters use fields, not regex over chunks.

Suggested next step: run `compare_deployed.py` on the deployed tree first (answers section 1 and 3). Then decide
"legacy restore under a legacy profile" vs "new hybrid retriever" using this golden set, extended with 20 to 50
real Xerox queries and documents and a 10k-chunk scale tenant. The golden test already gates every change.
