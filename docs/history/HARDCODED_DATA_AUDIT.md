# Hardcoded-data audit

Goal: production code must work against arbitrary documents and tenants. It must not know that any sample person,
organization, place, customer, host or document exists.

Method: `ultimate/scripts/audit/hardcoded_scan.py` (read-only, re-runnable) scans everything under `ultimate/`
except `tests/` and tooling, plus manual reading of every hit class below. Counts separate **executable code and
string literals** from **comments and docstrings** (the separation is heuristic for Python; comment-only mentions
are documentation debt, not behaviour).

```bash
python3 -I ultimate/scripts/audit/hardcoded_scan.py ultimate > hardcoded_scan.md
```

Baseline at commit `582aabb` (to be driven down phase by phase; see `MIGRATION_PLAN.md` S1):

| Category | In code / literals | Files | In comments only |
|---|---|---|---|
| Person names | 23 | 3 | 56 |
| Organization names | 19 | 5 | 46 |
| Locations (cities, states) | 122 | 6 | 98 |
| Document-type rules (NDA, governing law, "signed by", bank statement) | 206 | 7 | 135 |
| Company domains (`*.storagechain.io`) | 3 | 2 | |
| Private IP address | 1 | 1 (`.env.example`, placeholder) | |
| URLs | 49 | 15 | |
| Absolute filesystem paths | 7 | 5 | |
| Model identifiers | 23 | 8 | |
| Tenant/user literals (`user000`, `default_user`, `file_example`) | 14 | 4 | |
| Queue / collection / index names | 75 | 10 | |
| Secret-like literals | 0 | 0 | |
| Environment variables with literal defaults in code | 95 distinct | 17 | |

Secrets: none found in tracked files (the real `.env` files were never committed in this repository; the original
history is a separate matter, see `SECURITY.md`).

## Classification and disposition

Legend: **CONFIG** → typed settings from environment/config files; **SECRET** → secret store / mounted file;
**DOMAIN** → database or tenant/policy configuration; **TEST** → `tests/fixtures`, `tests/golden`;
**DEMO** → `demo/` or seed data, never imported by production code; **RULE** → explicit, configurable
query-planning or extraction rule; **LEGACY** → belongs to code scheduled for deletion (legacy search, S15).

### A. Customer, corpus and person coupling in executable code

| # | Where | What | Class | Disposition |
|---|---|---|---|---|
| A1 | `src/semantic/semantic_pipeline.py:2794-2795` | `if not query_persons and "<name>" in query_lower: query_persons = ["[person D]"]` | LEGACY | Delete with the semantic pipeline (S15). Behaviour replaced by generic person extraction |
| A2 | `src/semantic/query_enhancement.py:1386-1392` | Hardcoded `("chris", "california", 2015)` compound intent | LEGACY | Delete (S15). Generic: person + location + year are plan filters |
| A3 | `src/semantic/constraint_ranking.py:480-487, 598-623` | Chris/Christopher filename rule; California, Dallas, Austin office penalties | LEGACY | Delete (S15) |
| A4 | `src/semantic/constraint_ranking.py:71-77` `_TX_PEER_CITIES` (28), `query_enhancement.py:346-381` `_GEO_PAIR_SPECS` (51 US cities), `:409-431` `LOCATION_HINTS`, `_US_CITIES`, `_INTL_CITIES`, Austin/Dallas regexes `:288-316`, `semantic_utils.py:97-119` two-city helper | City/state lists and Texas-specific peer logic | LEGACY (+ DOMAIN if any survives) | Delete with legacy search. Target: a gazetteer loaded as data (GeoNames-class or customer list), never Python literals |
| A5 | `src/semantic/known_organizations.json` (7 orgs: RH Associates, [company B], Storage Chain, eightM Corp, Tech Holding, Misfits Gaming, eSports Now); `query_enhancement.py:52-58` `COMMON_ENGLISH` containing `storage`, `chain`, `misfits`, `gaming` | Previous customer's counterparties | DOMAIN (data) / LEGACY | Remove from the repository. Organization lists become per-tenant data (optional, curated by the customer) |
| A6 | `src/semantic/location_map.json` (30 keys) | Location aliases | DOMAIN | Replace with gazetteer data |
| A7 | NDA rules: `ultimate_ui.py:3660-3800` (`_NDA_TERMS*`, NDA filename supplement), semantic pipeline and components (≈ 190 literals) | One document type special-cased throughout search | RULE / LEGACY | Delete. Document types come from the configurable taxonomy and the classifier (`DOCUMENT_INTELLIGENCE_ARCHITECTURE.md`) |
| A8 | "bank statement" canonicalization `query_enhancement.py:724-725, 848-858` | Single-corpus synonym | LEGACY | Delete; synonyms become tenant-configurable data if needed |
| A9 | `src/ultimate_vector_integration.py:1232-1261` `file_example` tier (score 0.97) | Test-fixture special case inside production ranking | TEST | Remove from production code; the lexical tests already pin the real tiers |
| A10 | `src/semantic/semantic_pipeline.py:604-627`, `semantic_components.py:535-686`, `ultimate_vector_integration.py:84-89, 549-562` | Comments citing [person A], [person D], [person C], Storage Chain, Curation as examples | Documentation | Replace with neutral examples when those files are touched; most are in LEGACY code |

### B. Hosts, domains, users and UI defaults

| # | Where | What | Class | Disposition |
|---|---|---|---|---|
| B1 | `ultimate_ui.py:703` placeholder `https://file-view-stage.storagechain.io/...` | Previous company's staging host in the UI | CONFIG / DEMO | Neutral placeholder (`https://files.example.com/...`) |
| B2 | `ultimate_ui.py:704, 746, 788, 1179, 1546` default tenant `user000` in the built-in UI | A fixed tenant id | DEMO | Remove the default; the UI takes the tenant from the API key (already enforced server side) |
| B3 | `scripts/compare_staging_prod_search.py:35-38` production and staging URLs; query suite with real names (`[person B]`) | Previous operator's environment | DEMO / forensic | Move to `tools/legacy/` (kept as forensic evidence for S0), URLs only from env vars, no defaults |
| B4 | `src/ultimate_tasks.py:51`, `src/workflow_manager.py:52` `http://localhost:3333` workflow API default | External StorageChain workflow service | CONFIG / LEGACY | No default; feature disabled unless configured. Decide in S1 whether the workflow callback concept survives (generic webhooks) |
| B5 | `test_multiuser_load_curl.sh:3` remote sample file URL, `:7` local URL | Load-test script in the product tree | TEST | Move to `tests/load/` |
| B6 | `.env.example:22` `10.0.1.23` | Commented example of a private address | CONFIG (placeholder) | Keep as a documented placeholder or replace with `<private-ip>` |

### C. Tenant and identity literals

| # | Where | What | Class | Disposition |
|---|---|---|---|---|
| C1 | `src/vector_db_milvus_server.py:404, 559, 1583`; `src/ultimate_search_processor.py:3438` `"default_user"` | Rows without a tenant are stored under a shared fallback tenant | Security defect | Reject rows without a tenant (fail closed). Listed in `SECURITY_ARCHITECTURE.md` |
| C2 | `src/ultimate_vector_integration.py:803` comment about `user000` | Historical default | Documentation | Remove comment |
| C3 | API field names `bucketId`, `connectionId`, `bucket_id`, `connection_id`, `path` (53 references in `ultimate_ui.py`, Milvus schema fields) | StorageChain's storage concepts (bucket, S3 connection) baked into the API and schema | LEGACY API CONTRACT | Keep for backward compatibility during migration; the canonical model generalizes them to `collection_id` / `source_id` / `source_path` with ACL groups (`DOCUMENT_MODEL.md`) |

### D. Infrastructure names, paths and models (configuration)

| # | Where | What | Class | Disposition |
|---|---|---|---|---|
| D1 | Collection names `ultimate_document_chunks`, `ultimate_image_vectors` in 4 modules with separate defaults (`semantic_pipeline.py:393-394`, `ultimate_vector_integration.py:55-56`, `ultimate_search_processor.py:3430`, `health.py:71`) | Same setting defined 4 times | CONFIG | One settings module (S1) |
| D2 | Queue names in `ultimate_ui.py:97-98, 2090-2094`, `ultimate_celery_app.py:45-48`, `autoscale_manager.py:26-49` | Routing table spread over 3 files | CONFIG | One routing config |
| D3 | `/app/cache` (`embedder_service.py:21`, `semantic_components.py:53`, Dockerfile), `/app/temp_uploads` (`ultimate_ui.py:2499`, `ultimate_tasks.py:1020`) | Container paths in code | CONFIG | Settings with documented defaults |
| D4 | Model ids hardcoded: `embedder_service.py:21` (ignores `EMBED_MODEL_TEXT`), `embeddings.py:57, 320, 345`, `semantic_pipeline.py:413` (search side hardcodes MPNet), `semantic_components.py:58-63` reranker, `:1296-1332` BLIP, spaCy model names | Model choice and its dimension fixed in several places (KD-MLV-02) | CONFIG | Model registry config: name, revision, sha256, dimension, normalization; one source of truth checked at startup against the collection schema |
| D5 | `MILVUS_HOST` defaults `localhost` vs `milvus` (5 places); `WORKFLOW_ENABLED` `false` vs `true` (4 places) | **Conflicting defaults** (scanner flags) | CONFIG | One typed settings object; no per-module defaults |
| D6 | 95 environment variables read with literal defaults across 17 files | Configuration scattered through modules | CONFIG | Typed settings (pydantic-settings), validated at startup, printed (secrets redacted) in a startup report |
| D7 | `Dockerfile.ultimate:40-41` `download.pytorch.org` index URL; `requirements_ultimate.txt:70` spaCy model wheel URL | Build-time network dependencies | CONFIG / DEPLOYMENT | Pin by version and checksum; mirror for air-gapped builds (`DEPLOYMENT_ARCHITECTURE.md`) |
| D8 | Search constants: 50-chunk cap (`vector_db_milvus_server.py:601-605`), lexical tiers 0.98/0.95/0.91, `+0.25` overlap boost, `+0.15` semantic merge boost, 0.65 semantic floor, strictness profiles | Ranking behaviour as literals | RULE | Current values stay pinned by tests until the new retrieval path replaces them; new path takes all weights from versioned config |

### E. Test and demo data

| # | Where | Class | Disposition |
|---|---|---|---|
| E1 | `tests/golden/cases.py` (synthetic StorageChain-era and Xerox-style tenants, including [person A], Northgate Utilities, John Smith) | TEST | Correct location. Never imported by production code (verify with an import-linter rule in S1) |
| E2 | Legacy sample queries in `scripts/compare_staging_prod_search.py` | DEMO / forensic | `tools/legacy/` |
| E3 | Built-in UI sample values (B1, B2) | DEMO | Neutral placeholders |

## Order of work

1. **S1 (configuration):** D1 to D6 (typed settings, single source per value, conflicting defaults removed), B1,
   B2, B4, B6, C1 (fail closed), C2, A9, E2/B3/B5 moves. Add a CI check: the scanner must report zero person,
   organization and company-domain literals outside `tests/` and `tools/legacy/`, ignoring the legacy search
   modules until S15.
2. **S7/S8 (new planner and retrieval):** generic replacements for what A1 to A8 tried to do (persons, places,
   document types, synonyms) as data and rules.
3. **S15 (remove legacy implementation):** delete A1 to A8 with the legacy search modules; the scanner baseline for
   person, organization, location and document-rule literals in production code drops to zero.

Nothing in A1 to A8 is refactored in place: those rules are tied to one previous corpus, and their surrounding
code is the dead or unsafe legacy stack (`SEARCH_FORENSICS.md`).
