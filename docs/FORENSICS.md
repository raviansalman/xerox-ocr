# Forensic report: why ~2,500 lines of search logic are unreachable

Scope: the two missing names (`LOCATION_PEERS`, `threading`), what depends on them, and whether
deployed code can differ from the repository. **Nothing was restored.** No search code was changed.

## What evidence exists

| Source | Available | Notes |
|---|---|---|
| Git history of the original StorageChain repo | **No** | `StorageChain-LLC/vector-storage-processing-service-python` is not reachable with this session's GitHub credential |
| Other branches | **No** | Only the `main` snapshot was provided |
| The snapshot zip | Yes | Zip comment `e97b41f1d9c109db8ea7d93d26d3efd4285dd8d7`; every entry stamped `2026-08-24 10:44` |
| Deployment scripts and compose files | Yes | Read in full |

The zip comment plus uniform timestamps are exactly what GitHub's "Download ZIP" produces. **The snapshot is
the committed state of `main` at `e97b41f`**, not a working copy saved mid-edit. So the broken imports are
committed on `main`.

Questions 3 and 4 below ("which commit removed the names") cannot be answered without the original
history. Everything else is answered from the snapshot.

## Findings

| # | Commit | File | Change | Dependency | Intended behaviour | Security implications | Recommendation |
|---|---|---|---|---|---|---|---|
| F1 | `e97b41f` (introducing commit unknown) | `src/semantic/semantic_utils.py` | No `LOCATION_PEERS` defined. The module only has `tx_metro_snippet_has_wrong_peer_only`, which hardcodes one Austin↔Dallas peer pair | Imported by `query_enhancement.py:1097` (inside `enhance_query`, unconditional), `constraint_ranking.py:338` (`apply_constraint_boost`), `ultimate_ui.py:54` (`_prune_tx_peer_from_search_results`) | A `Dict[str, List[str]]` of mutually exclusive "peer" cities, used to set `location_anchor_cities`, demote chunks that only mention a peer city, and boost anchor-city matches. The consumers call `.keys()` and `.get(city, [])`, so it generalizes the two-city helper | Indirect. While it is missing, `enhance_query` fails on every call, which disables query understanding, the `/search` validators, all supplements and constraint ranking. The code it re-enables queries Milvus with the request's `user_id` (now escaped), so restoring it should not by itself open a tenant leak (TRACED, not run). It does change results for most queries | Do **not** restore yet. Decide in Phase 5 whether the legacy domain profile gets it back (golden set required) and what the generic profile uses instead |
| F2 | `e97b41f` (introducing commit unknown) | `src/semantic/semantic_components.py` | `MetadataIndex.__init__` (line 196) and `__setstate__` (line 966) call `threading.RLock()`; `threading` is never imported | `SemanticPipeline.__init__` catches the `NameError` and sets `metadata_index = None`. Every metadata-first path (`_ensure_metadata_index_for_user`, `_lexical_candidate_file_ids`, `_rerank_metadata_hits`, the Redis sync watcher, startup pre-warm, merged-constraint full-text checks) is then skipped | Thread-safe access to a per-user inverted index (dates, persons, orgs, locations, clauses) loaded from disk/Redis and hot-reloaded on the search server | **Critical if restored as is.** The index is one shared object per process; loading user B overwrites or appends to user A's data while A stays marked as "built". Reproduced: with `threading` patched in, Alice's search returned Bob's invoice text (KD-SEC-09). The Redis sync also deserialized pickle (fixed in this branch, see SECURITY.md) | Do **not** restore until the index is per-tenant and every hit is tenant-checked. Pinned by `test_metadata_index_can_be_constructed` (strict xfail) |
| F3 | n/a | `universal_deploy.sh`, `sync_ultimate.sh`, `copy_code_to_containers.sh`, every compose file | Deployments `rsync --delete` the developer's local `ultimate/` tree to servers, `docker cp src/` into running containers, and bind-mount `./src` over the image | Production and staging run whatever was on the deploying machine, not necessarily `main` | Fast iteration without rebuilding images | High. Code that was never committed or reviewed can run in production, and the repository is not a reliable record of what serves customers | Ask the previous team for the deployed tree (`docker exec <api> tar c /app/src`) and diff it against `e97b41f`. Stop source bind mounts in the target deployment |
| F4 | n/a | `.gitignore` (original) | Ignored `test_*.py`, `*_test.py`, `tools/`, `local_documents/`, `docker.env`, `.dockerignore` | Local-only files existed that the deploy scripts rsync (`.dockerignore` is rsynced explicitly) | Keep scratch files out of git | Supports F3: the deployed tree contained files git never saw | Already changed for tests; review the rest when F3 is answered |
| F5 | n/a | `docker-compose.dev.yml`, `readme.md` | Mount `src/chunking.py`, `src/normalize.py`, `src/search.py`, `ultimate/semantic/`; README documents `src/api.py`, `env.example`, a root `docker-compose.yml` | None of these files exist | An older "standard pipeline" with its own API, normalizer and chunker | None directly | Treat as evidence of an earlier architecture; see F6 |
| F6 | n/a | `src/semantic/semantic_components.py:1588-1960` | Contains code "merged from chunking.py" and "merged from normalize.py": a token-aware `TextChunker` (500/100 tokens, tiktoken `gpt-3.5-turbo`) and `DocumentElement.page_number` / `text_by_page` | Not used anywhere. Live chunking is `_chunk_text` (characters) in `ultimate_vector_integration.py` | The original pipeline chunked by tokens and kept page numbers | None | Explains the env file's `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`, `CHUNK_MODEL=gpt-3.5-turbo`: they were written for the token chunker but the live code reads them as characters. Page-aware extraction existed before and was lost. Reuse the design in Phase 4 |
| F7 | n/a | `src/semantic/semantic_utils.py:110`, `src/ultimate_search_processor.py:2758` | Two different `_normalize_text` functions (one correct, one corrupting, KD-OCR-07) | Each used in its own module | Text normalization | None today | Consolidate in Phase 3 |
| F8 | n/a | `semantic_components.py:511, 742`, `query_enhancement.py:167` | Three `normalize_org_name` implementations; two `extract_persons` (`semantic_components.py:1455`, `query_enhancement.py:456`) | Ingest-side vs query-side entity handling | Entity normalization | None today | One implementation per concept in Phase 3; ingest and query must normalize identically |
| F9 | n/a | `ultimate_vector_integration.py`, `semantic_pipeline.py`, `ultimate_search_processor.py` | Three `search_documents` implementations; `DocumentProcessor.search_documents` and the fuzzy layer are never called | Only the vector-integration and semantic-pipeline versions are live | Historical fuzzy keyword matching | None | Delete the dead one in Phase 3 |
| F10 | n/a | `ultimate_vector_integration.py:84-172` | `is_entity_only_query`, `contains_boilerplate_contract`, `filename_entity_overlap`, `debug_candidate` are never referenced | None | Earlier entity-aware ranking | None | Delete in Phase 3 |

## Answers to the seven questions

1. **Does another branch contain the missing implementations?** Unknown: no branch or history is available. The
   strongest lead is F3: the deployed tree on the staging/production servers.
2. **Intentional refactor or incomplete migration?** Incomplete. Both names are still used by code committed in the
   same snapshot, there is no replacement, and the errors are swallowed rather than handled. F5/F6 show at least one
   earlier consolidation (`chunking.py`, `normalize.py` merged into `semantic_components.py`). Moving code between
   modules without updating importers is consistent with the same kind of consolidation.
3. **Which commits removed the names?** Cannot be determined without history.
4. **What depended on them?** See F1 and F2: query enhancement (925 lines), constraint ranking (655), the `/search`
   validation and supplement block (about 770 lines), and the metadata-first router and index (several hundred lines across `semantic_pipeline.py` and `MetadataIndex`).
5. **Does deployment reference a different entry point?** No. All paths use `ultimate_ui:app` (or `search_api:app`,
   which calls the same factory), `src.embedder_service:app` and `src.ultimate_celery_app`. The divergence risk is the
   code behind those entry points (F3), not the entry points.
6. **Duplicate implementations?** Yes: F6 to F10 (chunking, text normalization, org/person normalization,
   `search_documents`, dead ranking helpers). No duplicate OCR or retrieval stack beyond these.
7. **Does deployment config point to code other than the default branch?** Yes, structurally (F3). Whether it
   actually does today needs the deployed tree.

## Recommended next step (needs the previous team)

```bash
# on the processing and search servers
docker exec <api-container> tar c -C /app src ultimate_ui.py | tar x -C ./deployed-snapshot
diff -ru ./deployed-snapshot/src ./xerox-ocr/ultimate/src
```

If the deployed tree defines `LOCATION_PEERS` and imports `threading`, production has been running the full
search stack, and the four regression queries need a golden baseline taken from production, not from `main`.
