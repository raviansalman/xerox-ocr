"""Per-tenant settings stored in the database (enabled domain packs, example questions for the UI)."""
from __future__ import annotations

import json

import psycopg
from pydantic import BaseModel, Field, field_validator

from docintel.config import get_settings


class Example(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    q: str = Field(min_length=1, max_length=300)


class TenantSettings(BaseModel):
    packs: list[str] | None = None            # None: the deployment default
    examples: list[Example] = Field(default_factory=list, max_length=30)

    @field_validator("packs")
    @classmethod
    def _known_packs(cls, v):
        if v is None:
            return v
        from docintel.packs import available_packs
        unknown = sorted(set(v) - set(available_packs()))
        if unknown:
            raise ValueError(f"unknown packs: {', '.join(unknown)}")
        return list(dict.fromkeys(v))


def load(conn: psycopg.Connection) -> TenantSettings:
    row = conn.execute("SELECT packs, examples FROM tenant_settings").fetchone()
    if not row:
        return TenantSettings()
    return TenantSettings(packs=row["packs"], examples=row["examples"] or [])


def save(conn: psycopg.Connection, tenant_id: str, settings: TenantSettings, actor: str) -> None:
    conn.execute(
        "INSERT INTO tenant_settings (tenant_id, packs, examples, updated_at, updated_by) VALUES (%s, %s, %s, now(), %s) "
        "ON CONFLICT (tenant_id) DO UPDATE SET packs = EXCLUDED.packs, examples = EXCLUDED.examples, "
        "updated_at = now(), updated_by = EXCLUDED.updated_by",
        (tenant_id, settings.packs, json.dumps([e.model_dump() for e in settings.examples]), actor))


def enabled_packs(conn: psycopg.Connection) -> list[str]:
    s = load(conn)
    return s.packs if s.packs is not None else get_settings().packs
