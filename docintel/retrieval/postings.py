"""Access to the term-postings index (``unit_terms``; migrations/005). All lookups are B-tree equality or range
scans on (tenant_id, term[, unit_id]), which PostgreSQL uses under row-level security.

Intersections start from the rarest term and seek the other terms only for the surviving units, so common words
cost little. Phrase and adjacency checks run afterwards on the candidate units only (``verify``).
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import psycopg

from docintel.config import get_settings

DF_CAP = 200_000                 # document frequencies are counted up to this many units
BM25_K1 = 1.2
BM25_B = 0.75
_TENANT_UNITS_TTL = 60.0
_tenant_units_cache: dict[str, tuple[float, int]] = {}
_cache_lock = threading.Lock()


@dataclass
class UnitHit:
    unit_id: str
    document_id: str
    tf: dict[str, int] = field(default_factory=dict)      # term -> frequency in the unit


def _upper(prefix: str) -> str:
    """The smallest string greater than every string starting with ``prefix`` (byte order, "C" collation)."""
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


class Postings:
    def __init__(self, conn: psycopg.Connection, tenant_id: str):
        self.conn = conn
        self.tenant_id = tenant_id

    # -------------------------------------------------------------------------------------------- statistics

    def df(self, terms: Iterable[str]) -> dict[str, int]:
        terms = sorted(set(terms))
        if not terms:
            return {}
        # counting stops at DF_CAP: a word in that many units already has an IDF near zero, and an exact count of a
        # word in millions of units would cost a scan of its whole posting list on every query
        rows = self.conn.execute(
            "SELECT t.term, (SELECT count(*) FROM (SELECT 1 FROM unit_terms u WHERE u.term = t.term LIMIT %s) c) AS n "
            "FROM unnest(%s::text[]) AS t(term)", (DF_CAP, terms)).fetchall()
        out = dict.fromkeys(terms, 0)
        out.update({r["term"]: int(r["n"]) for r in rows})
        return out

    def total_units(self) -> int:
        """Number of units of the tenant (cached for a minute; used only for IDF)."""
        now = time.monotonic()
        with _cache_lock:
            hit = _tenant_units_cache.get(self.tenant_id)
            if hit and now - hit[0] < _TENANT_UNITS_TTL:
                return hit[1]
        n = int(self.conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"])
        with _cache_lock:
            _tenant_units_cache[self.tenant_id] = (now, n)
            if len(_tenant_units_cache) > 10000:
                _tenant_units_cache.clear()
        return n

    def lengths(self, unit_ids: Iterable[str]) -> dict[str, int]:
        ids = list(set(unit_ids))
        if not ids:
            return {}
        rows = self.conn.execute("SELECT id, n_terms FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()
        return {r["id"]: int(r["n_terms"]) for r in rows}

    # -------------------------------------------------------------------------------------------- lookups

    def _fetch(self, term: str, scope: tuple[str, ...] | None, units: list[str] | None, cap: int | None) -> list[dict]:
        sql = "SELECT unit_id, document_id, tf FROM unit_terms WHERE term = %s"
        args: list = [term]
        if units is not None:
            sql += " AND unit_id = ANY(%s)"
            args.append(units)
        if scope is not None:
            sql += " AND document_id = ANY(%s::uuid[])"
            args.append(list(scope))
        if cap is not None:                       # keep the units where the term matters most, deterministically
            sql += " ORDER BY unit_id LIMIT %s"
            args.append(cap)
        return self.conn.execute(sql, args).fetchall()

    def all_of(self, terms: list[str], scope: tuple[str, ...] | None = None, cap: int | None = None) -> dict[str, UnitHit]:
        """Units containing every term (each term may be a list of alternatives: "w:x|f:x" means w:x or f:x)."""
        groups = [t.split("|") for t in dict.fromkeys(terms) if t]
        if not groups:
            return {}
        df = self.df(t for g in groups for t in g)
        groups.sort(key=lambda g: sum(df.get(t, 0) for t in g))
        if sum(df.get(t, 0) for t in groups[0]) == 0:
            return {}
        hits: dict[str, UnitHit] = {}
        for t in groups[0]:
            for r in self._fetch(t, scope, None, cap):
                h = hits.setdefault(r["unit_id"], UnitHit(r["unit_id"], str(r["document_id"])))
                h.tf[t] = int(r["tf"])
        for g in groups[1:]:
            if not hits:
                break
            keep: dict[str, UnitHit] = {}
            ids = list(hits)
            for t in g:
                for r in self._fetch(t, None, ids, None):
                    h = hits[r["unit_id"]]
                    h.tf[t] = int(r["tf"])
                    keep[r["unit_id"]] = h
            hits = keep
        return hits

    def any_of(self, terms: list[str], scope: tuple[str, ...] | None = None, per_term_cap: int = 2000) -> dict[str, UnitHit]:
        hits: dict[str, UnitHit] = {}
        for t in dict.fromkeys(terms):
            for r in self._fetch(t, scope, None, per_term_cap):
                h = hits.setdefault(r["unit_id"], UnitHit(r["unit_id"], str(r["document_id"])))
                h.tf[t] = int(r["tf"])
        return hits

    def prefix(self, prefix: str, scope: tuple[str, ...] | None = None, cap: int = 500) -> dict[str, UnitHit]:
        """Units with a term starting with ``prefix`` (e.g. "w:0048" for "00481", "r:1840" for suffix "0481")."""
        sql = "SELECT term, unit_id, document_id, tf FROM unit_terms WHERE term >= %s AND term < %s"
        args: list = [prefix, _upper(prefix)]
        if scope is not None:
            sql += " AND document_id = ANY(%s::uuid[])"
            args.append(list(scope))
        rows = self.conn.execute(sql + " ORDER BY unit_id LIMIT %s", [*args, cap]).fetchall()
        hits: dict[str, UnitHit] = {}
        for r in rows:
            h = hits.setdefault(r["unit_id"], UnitHit(r["unit_id"], str(r["document_id"])))
            h.tf[r["term"]] = int(r["tf"])
        return hits

    def verify(self, unit_ids: list[str], condition: str, arg: str) -> set[str]:
        """Units among ``unit_ids`` whose stored vectors satisfy a fixed condition template (phrase checks)."""
        if not unit_ids:
            return set()
        out: set[str] = set()
        for i in range(0, len(unit_ids), 5000):
            rows = self.conn.execute(f"SELECT id FROM chunks WHERE id = ANY(%s) AND {condition}",
                                     (unit_ids[i:i + 5000], arg)).fetchall()
            out |= {r["id"] for r in rows}
        return out

    # -------------------------------------------------------------------------------------------- scoring

    def bm25(self, hits: dict[str, UnitHit], terms: list[str]) -> dict[str, float]:
        if not hits:
            return {}
        n = max(1, self.total_units())
        df = self.df(t for g in terms for t in g.split("|"))
        lengths = self.lengths(hits)
        avg = get_settings().lexical_avg_unit_terms
        scores: dict[str, float] = {}
        for uid, h in hits.items():
            dl = lengths.get(uid, avg)
            s = 0.0
            for t, tf in h.tf.items():
                d = df.get(t, 1)
                idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
                s += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg))
            scores[uid] = s
        return scores
