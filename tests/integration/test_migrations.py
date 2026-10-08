"""Database migrations: a clean install, rollback and re-apply, and an upgrade from the v1 schema with data
followed by the stale-document backfill. Each test uses a scratch database (needs DOCINTEL_TEST_ADMIN_DATABASE_URL,
a role allowed to create databases)."""
import argparse
import os
import uuid

import psycopg
import pytest

pytestmark = pytest.mark.integration
ADMIN = os.environ.get("DOCINTEL_TEST_ADMIN_DATABASE_URL")
TENANT = "tenantmig"
TENANT_TABLES = ("documents", "pages", "chunks", "fields", "entities", "clauses", "audit_events", "blocks", "doc_tables",
                 "table_cells", "relations", "document_versions", "tenant_settings", "unit_terms", "vocabulary")
SYSTEM_TABLES = ("ingest_jobs",)        # the job queue: read only through system connections, never by tenant code


@pytest.fixture
def scratch(engine_env):
    if not ADMIN:
        pytest.skip("set DOCINTEL_TEST_ADMIN_DATABASE_URL to run migration tests")
    from docintel.storage.db import Database
    name = f"docintel_mig_{uuid.uuid4().hex[:8]}"
    app_url = psycopg.conninfo.make_conninfo(engine_env.url, dbname=name)
    owner = psycopg.conninfo.conninfo_to_dict(engine_env.url)["user"]
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}" OWNER "{owner}" TEMPLATE template0 ENCODING UTF8 LC_COLLATE "C.UTF-8" LC_CTYPE "C.UTF-8"')
    db = Database(app_url, 1, 4)
    yield db
    db.close()
    with psycopg.connect(ADMIN, autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _tables(db):
    with db.system() as conn:
        return {r["relname"]: r["relforcerowsecurity"] for r in conn.execute(
            "SELECT relname, relforcerowsecurity FROM pg_class WHERE relkind = 'r' AND relnamespace = 'public'::regnamespace")}


def test_clean_install_creates_every_table_with_forced_row_level_security(scratch):
    from docintel.storage.db import up_migrations
    applied = scratch.migrate()
    assert applied == [p.name for p in up_migrations()]
    tables = _tables(scratch)
    missing = [t for t in TENANT_TABLES if t not in tables]
    assert not missing and all(tables[t] for t in TENANT_TABLES), {t: tables.get(t) for t in TENANT_TABLES}
    with scratch.system() as conn:
        with_tenant = {r["table_name"] for r in conn.execute(
            "SELECT table_name FROM information_schema.columns WHERE table_schema = 'public' AND column_name = 'tenant_id'")}
    assert with_tenant - set(SYSTEM_TABLES) == set(TENANT_TABLES)        # a new tenant table must be added here
    assert scratch.migrate() == [] and scratch.collation_check() is None
    assert not scratch.security_check()["role_bypasses_rls"]


def test_rollback_and_reapply(scratch):
    from docintel.storage.db import up_migrations
    names = [p.name for p in up_migrations()]
    scratch.migrate()
    assert scratch.rollback("002") == list(reversed(names[2:]))
    assert "unit_terms" not in _tables(scratch) and "ingest_jobs" in _tables(scratch)
    assert scratch.migrate("004") == names[2:4] and "unit_terms" not in _tables(scratch)
    assert scratch.migrate("004") == []                       # already there: nothing beyond it is applied
    assert scratch.migrate() == names[4:]
    with pytest.raises(ValueError):
        scratch.rollback("09")                                # a typo never reverts everything
    assert scratch.rollback("none") == list(reversed(names)) and set(_tables(scratch)) == {"schema_migrations"}


def test_upgrade_from_v1_with_data_then_backfill(scratch, monkeypatch):
    """A v1 database (migrations 001-002) with an indexed document: the upgrade keeps the data, the document is
    reported stale, and `docintel reprocess --stale` rebuilds it into the v2 model and indexes."""
    from docintel import cli
    from docintel.query import get_engine, reset_engines
    from docintel.storage import db as dbmod
    from docintel.storage.objects import get_object_store

    scratch.migrate("002")
    body = b"Legacy archive record LR-90210. The cold storage vault was inspected on 4 May 2023.\n"
    key, sha, size = get_object_store().put_bytes(TENANT, body, 10_000_000)
    doc_id = str(uuid.uuid4())
    with scratch.tenant(TENANT) as conn:                       # what the v1 pipeline left behind
        conn.execute("INSERT INTO documents (id, tenant_id, filename, filename_search, size_bytes, sha256, storage_key, "
                     "status, kind, page_count, pipeline_version, indexed_at) VALUES (%s, %s, 'legacy_record.txt', "
                     "'legacy record txt', %s, %s, %s, 'indexed', 'text', 1, '1.0.0', now())",
                     (doc_id, TENANT, size, sha, key))
        conn.execute("INSERT INTO pages (document_id, tenant_id, page_number, kind, text) VALUES (%s, %s, 1, 'text', %s)",
                     (doc_id, TENANT, body.decode()))
        conn.execute("INSERT INTO chunks (id, document_id, tenant_id, ordinal, page_start, page_end, text, search_text, "
                     "fold_text) VALUES (%s, %s, %s, 0, 1, 1, %s, %s, %s)",
                     (f"{doc_id}:0", doc_id, TENANT, body.decode(), body.decode().lower(), body.decode().lower()))

    assert scratch.migrate() and "unit_terms" in _tables(scratch)
    with scratch.tenant(TENANT) as conn:
        row = conn.execute("SELECT status, components FROM documents WHERE id = %s", (doc_id,)).fetchone()
        assert row["status"] == "indexed" and row["components"] is None          # data kept; marked as pre-v2
        assert conn.execute("SELECT count(*) AS n FROM chunks WHERE document_id = %s", (doc_id,)).fetchone()["n"] == 1

    monkeypatch.setattr(dbmod, "_db", scratch)
    reset_engines()
    try:
        cli.cmd_reprocess(argparse.Namespace(tenant=TENANT, stale=True, wait=True, timeout=120))
        with scratch.tenant(TENANT) as conn:
            row = conn.execute("SELECT status, components, current_version FROM documents WHERE id = %s", (doc_id,)).fetchone()
            assert row["status"] == "indexed" and row["components"]["pipeline"] and row["current_version"]
            assert conn.execute("SELECT count(*) AS n FROM unit_terms WHERE document_id = %s", (doc_id,)).fetchone()["n"] > 0
        out = get_engine().run(cli._ctx(TENANT), "LR-90210")
        assert [r["document_id"] for r in out["results"]] == [doc_id] and out["results"][0]["tier"] == 1
        cli.cmd_reprocess(argparse.Namespace(tenant=TENANT, stale=True, wait=True, timeout=120))   # nothing stale now
    finally:
        monkeypatch.setattr(dbmod, "_db", None)
        reset_engines()


def test_replicas_migrating_at_the_same_time_do_not_collide(scratch):
    """Several API processes start together on an empty database; exactly one applies each migration."""
    from concurrent.futures import ThreadPoolExecutor

    from docintel.storage.db import Database, up_migrations
    dbs = [Database(scratch.url, 1, 1) for _ in range(4)]
    try:
        with ThreadPoolExecutor(4) as pool:
            applied = list(pool.map(lambda d: d.migrate(), dbs))
    finally:
        for d in dbs:
            d.close()
    assert sorted(n for a in applied for n in a) == sorted(p.name for p in up_migrations())
