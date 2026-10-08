"""Domain packs: domain knowledge as data, merged into one ``Domain`` per set of enabled packs.

A pack is a directory with ``pack.yaml`` and any of ``taxonomy.yaml`` (document types and families),
``extraction.yaml`` (date and amount roles, identifier fields, party and person cues, clause types, question field
terms), ``concepts.yaml`` (concept variants for contextual retrieval), ``relations.yaml`` (relation patterns) and
``jurisdictions.yaml`` (gazetteer). Packs ship in this directory; deployments add their own with
``DOCINTEL_PACKS_DIR``. The ``core`` pack is always enabled and holds only generic vocabulary, so the engine works
without any domain pack.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
CORE = "core"
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


class PackError(ValueError):
    """A pack is missing, malformed or depends on a pack that is not available."""


@dataclass(frozen=True)
class DocType:
    label: str
    display: str
    family: str | None
    query_terms: tuple[str, ...]
    title_patterns: tuple[str, ...]
    keywords: dict[str, float] = field(default_factory=dict)
    description: str = ""
    title_weight: float = 5.0
    generic: bool = False          # the catch-all member of its family ("contract" among agreements)
    date_field: str | None = None  # the date a question about this type refers to by default ("invoices from 2025")
    pack: str = ""


@dataclass(frozen=True)
class Concept:
    name: str
    variants: tuple[str, ...]
    clause_type: str | None
    predicate: str | None
    pack: str


@dataclass(frozen=True)
class RelationPattern:
    predicate: str
    pattern: re.Pattern
    pack: str


def _dirs() -> list[Path]:
    extra = os.environ.get("DOCINTEL_PACKS_DIR")
    return ([Path(extra)] if extra else []) + [_HERE]


def _pack_dir(name: str) -> Path:
    if not _NAME.match(name):
        raise PackError(f"invalid pack name {name!r}")
    for d in _dirs():
        if (d / name / "pack.yaml").exists():
            return d / name
    raise PackError(f"pack {name!r} is not installed")


@cache
def available_packs() -> tuple[str, ...]:
    names = set()
    for d in _dirs():
        if d.exists():
            names |= {p.name for p in d.iterdir() if (p / "pack.yaml").exists() and _NAME.match(p.name)}
    return tuple(sorted(names))


@cache
def _load(name: str, file: str) -> dict[str, Any]:
    path = _pack_dir(name) / file
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise PackError(f"{name}/{file} must be a mapping")
    return data


def _resolve(names: tuple[str, ...]) -> tuple[str, ...]:
    """Enabled packs with their dependencies, core first, each once, in a stable order."""
    order: list[str] = []

    def visit(n: str, stack: tuple[str, ...]) -> None:
        if n in order:
            return
        if n in stack:
            raise PackError(f"pack dependency cycle: {' -> '.join((*stack, n))}")
        for dep in _load(n, "pack.yaml").get("depends_on") or []:
            visit(dep, (*stack, n))
        order.append(n)

    for n in (CORE, *names):
        visit(n, ())
    return tuple(order)


class Domain:
    """The merged knowledge of a set of packs. Immutable after construction; cached per pack set."""

    def __init__(self, packs: tuple[str, ...]):
        self.packs = packs
        self.types: dict[str, DocType] = {}
        self.families: dict[str, list[str]] = {}
        self.family_terms: dict[str, list[str]] = {}
        self.date_roles: dict[str, list[str]] = {}
        self.amount_roles: dict[str, list[str]] = {}
        self.identifier_fields: dict[str, list[str]] = {}
        self.party_cues: dict[str, str] = {}
        self.person_cues: dict[str, str] = {}
        self.clause_types: dict[str, list[str]] = {}
        self.field_terms: dict[str, list[str]] = {}
        self.lookup_terms: dict[str, list[str]] = {}
        self.concepts: dict[str, Concept] = {}
        self.relation_patterns: list[RelationPattern] = []
        self.jurisdictions: list[dict[str, Any]] = []
        for name in packs:
            self._merge(name)
        self._finish()

    @staticmethod
    def _extend(target: dict[str, list[str]], source: dict[str, Any] | None) -> None:
        for k, v in (source or {}).items():
            target.setdefault(k, [])
            target[k].extend(x for x in (v or []) if x not in target[k])

    def _merge(self, name: str) -> None:
        tax = _load(name, "taxonomy.yaml")
        self._extend(self.families, tax.get("families"))
        self._extend(self.family_terms, tax.get("family_terms"))
        for label, t in (tax.get("types") or {}).items():
            self.types[label] = DocType(
                label=label, display=t.get("display", label), family=None,
                query_terms=tuple(t.get("query_terms") or ()), title_patterns=tuple(t.get("title_patterns") or ()),
                keywords={k: float(v) for k, v in (t.get("keywords") or {}).items()},
                description=t.get("description", ""), title_weight=float(t.get("title_weight", 5.0)),
                generic=bool(t.get("generic", False)), date_field=t.get("date_field"), pack=name)
        ext = _load(name, "extraction.yaml")
        for key in ("date_roles", "amount_roles", "identifier_fields", "clause_types", "field_terms", "lookup_terms"):
            self._extend(getattr(self, key), ext.get(key))
        self.party_cues.update(ext.get("party_cues") or {})
        self.person_cues.update(ext.get("person_cues") or {})
        for cname, c in (_load(name, "concepts.yaml").get("concepts") or {}).items():
            prev = self.concepts.get(cname)
            variants = tuple(dict.fromkeys([*(prev.variants if prev else ()), *(v.lower() for v in c.get("variants") or ())]))
            self.concepts[cname] = Concept(cname, variants, c.get("clause_type") or (prev.clause_type if prev else None),
                                           c.get("predicate") or (prev.predicate if prev else None), name)
        for r in _load(name, "relations.yaml").get("relations") or []:
            try:
                raw = r["pattern"]
                rx = re.compile("".join(raw) if isinstance(raw, list) else str(raw), re.I)
            except (KeyError, re.error) as e:
                raise PackError(f"{name}/relations.yaml: invalid pattern for {r.get('predicate')}: {e}") from e
            self.relation_patterns.append(RelationPattern(str(r["predicate"]), rx, name))
        self.jurisdictions.extend(_load(name, "jurisdictions.yaml").get("jurisdictions") or [])

    def _finish(self) -> None:
        family_of = {t: fam for fam, members in self.families.items() for t in members}
        self.types = {k: DocType(**{**t.__dict__, "family": family_of.get(k)}) for k, t in self.types.items()}
        # amount role "amount" is the fallback and must come last
        if "amount" in self.amount_roles:
            self.amount_roles["amount"] = self.amount_roles.pop("amount")

    # ------------------------------------------------------------------ derived, computed once per Domain

    @cached_property
    def title_patterns(self) -> dict[str, list[re.Pattern]]:
        return {label: [re.compile(p, re.I | re.M) for p in t.title_patterns] for label, t in self.types.items()}

    @cached_property
    def gazetteer(self) -> list[tuple[str, str, str]]:
        """(folded alias, canonical name, code), longest aliases first."""
        from docintel import text as T
        out = []
        for j in self.jurisdictions:
            for alias in [j["name"], *(j.get("aliases") or [])]:
                out.append((T.fold(alias), j["name"], j["code"]))
        return sorted(out, key=lambda x: -len(x[0]))

    @cached_property
    def concept_index(self) -> list[tuple[str, Concept]]:
        """(variant, concept), longest variants first, for matching questions."""
        return sorted(((v, c) for c in self.concepts.values() for v in c.variants), key=lambda x: -len(x[0]))

    @cached_property
    def lookup_words(self) -> dict[str, str]:
        """Question phrase -> field name ("invoice number" -> invoice_number); longer phrases are matched first."""
        return {phrase: name for name, phrases in self.lookup_terms.items() for phrase in phrases}

    @cached_property
    def type_terms(self) -> list[tuple[str, list[str]]]:
        """(question term, document types) for type and family terms, longest terms first."""
        terms = [(term, [label]) for label, t in self.types.items() for term in t.query_terms]
        terms += [(w, list(self.families.get(fam, []))) for fam, words in self.family_terms.items() for w in words]
        return sorted(terms, key=lambda x: -len(x[0]))

    def type_label(self, label: str | None) -> str:
        t = self.types.get(label or "")
        return t.display if t else (label or "Unclassified").replace("_", " ").capitalize()


@cache
def get_domain(packs: tuple[str, ...] = ()) -> Domain:
    """The Domain for a set of enabled packs (core and dependencies are added automatically)."""
    for p in packs:
        _pack_dir(p)
    return Domain(_resolve(tuple(packs)))


def default_domain() -> Domain:
    from docintel.config import get_settings
    return get_domain(tuple(get_settings().packs))


def reset_packs() -> None:
    """Forget loaded packs (tests that install packs at runtime)."""
    available_packs.cache_clear()
    _load.cache_clear()
    get_domain.cache_clear()
