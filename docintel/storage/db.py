"""PostgreSQL access: connection pool, migrations and tenant-scoped transactions.

All tenant data access goes through ``Database.tenant(tenant_id)``, which opens a transaction and sets
``docintel.tenant``; row-level security (``migrations/001_initial.sql``) then restricts every statement to that
tenant. The application role must not be a superuser or have BYPASSRLS (checked by ``security_check``).
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from docintel.config import get_settings
from docintel.security import is_valid_tenant_id

logger = logging.getLogger(__name__)
MIGRATIONS = Path(__file__).resolve().parent / "migrations"


class Database:
    def __init__(self, url: str, min_size: int = 1, max_size: int = 10):
        self.url = url
        self.pool = ConnectionPool(url, min_size=min_size, max_size=max_size, open=True,
                                   kwargs={"row_factory": dict_row, "autocommit": False})

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        """A live pooled connection. Connections that died (database restart, failover) are replaced instead of
        failing the caller. On the first dead one, every idle connection is checked in one pass: the pool's own
        per-connection check reconnects each with a doubling back-off (measured: 6 dead connections, 32 s)."""
        conn = self.pool.getconn()
        try:
            ConnectionPool.check_connection(conn)
        except psycopg.OperationalError:
            self.pool.putconn(conn)                    # broken: the pool discards it
            self.pool.check()                          # the others are almost certainly dead too
            conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def tenant(self, tenant_id: str) -> Iterator[psycopg.Connection]:
        """A transaction in which only ``tenant_id``'s rows are visible or writable."""
        if not is_valid_tenant_id(tenant_id):
            raise ValueError("invalid tenant id")
        with self._connection() as conn:
            with conn.transaction():
                conn.execute("SELECT set_config('docintel.tenant', %s, true)", (tenant_id,))
                yield conn

    @contextmanager
    def system(self) -> Iterator[psycopg.Connection]:
        """A transaction without tenant context (migrations, health). Tenant tables return no rows here."""
        with self._connection() as conn:
            with conn.transaction():
                yield conn

    def migrate(self, upto: str | None = None) -> list[str]:
        """Apply pending migrations in order (all, or up to and including the one whose name starts with ``upto``)."""
        applied = []
        with self.system() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(4242)")       # before any DDL: replicas start together
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz DEFAULT now())")
            done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()}
            if upto:
                _one_migration(upto)
            for path in up_migrations():
                if path.name not in done:
                    conn.execute(path.read_text(encoding="utf-8"))
                    conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
                    applied.append(path.name)
                if upto and path.name.startswith(upto):
                    break
        if applied:
            logger.info("applied migrations", extra={"migrations": applied})
        return applied

    def rollback(self, to: str) -> list[str]:
        """Revert applied migrations newer than ``to`` (a name prefix of exactly one migration, or "none" for all),
        newest first, using their .down.sql files."""
        if to != "none":
            _one_migration(to)
        reverted = []
        with self.system() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(4242)")
            done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()}
            for path in reversed(up_migrations()):
                if path.name.startswith(to):
                    break
                if path.name not in done:
                    continue
                down = path.with_name(path.name.replace(".sql", ".down.sql"))
                if not down.exists():
                    raise RuntimeError(f"migration {path.name} cannot be reverted (no {down.name})")
                conn.execute(down.read_text(encoding="utf-8"))
                conn.execute("DELETE FROM schema_migrations WHERE name = %s", (path.name,))
                reverted.append(path.name)
        return reverted

    def collation_check(self) -> str | None:
        """A warning when the database does not sort by bytes (prefix lookups on names would be unreliable)."""
        with self.system() as conn:
            row = conn.execute("SELECT datcollate FROM pg_database WHERE datname = current_database()").fetchone()
        coll = (row or {}).get("datcollate") or ""
        if coll.split(".")[0] not in ("C", "POSIX"):
            return f"database collation is {coll}; create the database with LC_COLLATE 'C.UTF-8' (deploy/postgres-init.sql)"
        return None

    def security_check(self) -> dict:
        with self.system() as conn:
            row = conn.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user").fetchone()
        unsafe = bool(row and (row["rolsuper"] or row["rolbypassrls"]))
        return {"role_bypasses_rls": unsafe}

    def ping(self) -> None:
        with self.system() as conn:
            conn.execute("SELECT 1")


def _one_migration(prefix: str) -> Path:
    """The single migration a prefix names; anything else is an error (a typo must never revert everything)."""
    matches = [p for p in up_migrations() if p.name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"{prefix!r} names {len(matches)} migrations; give a prefix of exactly one "
                         f"({', '.join(p.name for p in up_migrations())})")
    return matches[0]


def up_migrations() -> list[Path]:
    return sorted(p for p in MIGRATIONS.glob("*.sql") if not p.name.endswith(".down.sql"))


_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        s = get_settings()
        _db = Database(s.require("database_url"), s.db_pool_min, s.db_pool_max)
        check = _db.security_check()
        if check["role_bypasses_rls"]:
            msg = ("database role is a superuser or has BYPASSRLS, so row-level tenant isolation is not enforced; "
                   "connect with a dedicated application role (see deploy/postgres-init.sql)")
            if s.is_production:
                raise RuntimeError(msg)
            logger.warning(msg)
        warning = _db.collation_check()
        if warning:
            logger.warning(warning)
    return _db


def set_db(db: Database | None) -> None:
    """Install a database (tests) or reset to lazy creation."""
    global _db
    _db = db
