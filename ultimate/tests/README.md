# xerox-ocr regression suite

Characterization tests that pin how the system behaves today, so refactoring can
proceed without silently changing search, indexing or OCR results.

## Tiers

| Tier | Needs | Command (from `ultimate/`) |
| --- | --- | --- |
| Unit | Python deps only (Tesseract optional) | `pytest` |
| Integration | Milvus 2.3.1 on :19530, Redis on :6379 | `RUN_INTEGRATION=1 pytest` |

Setup:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-test.txt
# optional, enables the OCR tests:
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara fonts-dejavu-core libfribidi0
```

For the integration tier:

```bash
docker run -d --name xocr-milvus -p 19530:19530 -p 9091:9091 \
  -e ETCD_USE_EMBED=true -e ETCD_DATA_DIR=/var/lib/milvus/etcd -e COMMON_STORAGETYPE=local \
  milvusdb/milvus:v2.3.1 milvus run standalone
docker run -d --name xocr-redis -p 6379:6379 redis:7-alpine
RUN_INTEGRATION=1 pytest
```

The integration tier starts a deterministic stand-in embedder
(`tests/support/fake_embedder.py`, same `/embed` contract as `src/embedder_service.py`)
and uses throwaway collections (`xocr_test_docs_*`) that are dropped afterwards.
It validates plumbing, tenant scoping and exact/lexical behaviour, **not** the semantic
quality of all-mpnet-base-v2. Semantic quality has to be checked against the Docker stack.

## API keys in tests

`conftest.py` configures test keys (hashes only) from `tests/support/auth.py`: a `service` key, an `admin`
key and tenant-bound reader/uploader keys for `alice` and `bob`. Use `headers("service")` etc. in tests.

## Known defects

Tests marked `known_defect` assert the *correct* behaviour and are `xfail(strict=True)`.
Each one documents a verified bug (IDs match `docs/ASSESSMENT.md`). When a fix lands the
test starts passing, strict mode turns that into a failure, and the marker must be
removed in the same change. That keeps every behaviour change deliberate and reviewed.

To see why each known defect fails right now: `pytest --runxfail --tb=line`.

## What is frozen

* `test_milvus_contract.py`: collection schemas, 768/512 dims, HNSW(M=32, efConstruction=200)
  + COSINE, search `ef`/limit cap, tenant filter expression format, inserted row shape.
  Changing any of these breaks compatibility with populated collections.
* `test_chunking_and_upsert.py`: chunk spans, `[FILE: ...]` header on every chunk, scoping metadata.
* `test_lexical_matching.py`: filename/file_id lexical scan tiers (0.98 / 0.95 / 0.91).
* `test_api_contract.py`: queue routing table, request validation.
* `integration/test_search_regression.py`: "FOR IMMEDIATE RELEASE", "press release",
  "Lisa Riordan", "StorageChain" rank the press release first in vector, semantic and both modes;
  tenants with well-formed ids never see each other's documents.
