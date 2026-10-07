#!/usr/bin/env python3
"""Hardcoded-data scanner for production code (read-only).

Scans everything under ultimate/ except tests/, caches and this audit folder, and reports literal values that
couple production code to a specific corpus, customer, host or environment. Output is Markdown, used by
docs/HARDCODED_DATA_AUDIT.md. Re-run after each clean-up step; the counts should only go down.

    python3 -I ultimate/scripts/audit/hardcoded_scan.py ultimate > hardcoded_scan.md
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SKIP_PARTS = {"tests", "__pycache__", ".pytest_cache", "audit", "forensics"}
SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".json", ".example", ".txt", ".ultimate", ".env"}

# Corpus/customer vocabulary found in the forensics work. Matched case-insensitively on word boundaries.
CORPUS_TERMS = {
    "person": ["lisa riordan", "riordan", "chris dominguez", "dominguez", "david subar", "subar", "mitul",
               "thobhani", "azat", "jane doe"],
    "organization": ["storagechain", "storage chain", "freightpal", "metafesto", "curation media", "misfits",
                     "eightm", "esports now", "rh associates", "tech holding", "carnival", "coinstore", "xerox"],
    "location": ["austin", "dallas", "houston", "texas", "california", "berlin", "new york", "delaware"],
    "document_rule": ["nda", "mnda", "non-disclosure", "bank statement", "signed by", "governing law"],
}
PATTERNS = {
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url": re.compile(r"https?://[^\s\"'`)>]+"),
    "company_domain": re.compile(r"\b[\w.-]+\.(?:storagechain\.io|storagechain\.com)\b"),
    "absolute_path": re.compile(r"[\"'](/(?:app|home|root|opt|data|mnt|var|tmp)/[^\"']*)[\"']"),
    "model_id": re.compile(r"(?:sentence-transformers|BAAI|Salesforce|openai|facebook|microsoft|intfloat|google)/[\w.-]+"
                           r"|\byolov?\w*\.pt\b|\ben_core_web_\w+\b"),
    "tenant_or_user_literal": re.compile(r"[\"'](?:default_user|user\d{2,}|test_user|demo_user|admin_user|file_example\w*)[\"']"),
    "queue_name": re.compile(r"[\"'](ultimate_[a-z_]+|celery|metadata_[a-z_]+)[\"']"),
    "secret_like": re.compile(r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?key|token)\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']"),
    "env_default": re.compile(r"os\.(?:getenv|environ\.get)\(\s*[\"'](\w+)[\"']\s*,\s*[\"']([^\"']+)[\"']"),
}


def files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (SKIP_PARTS & set(p.parts)) and (p.suffix in SUFFIXES or p.name.startswith("Dockerfile")):
            yield p


def main(root: Path):
    hits = defaultdict(list)          # category -> [(file, line, text)]
    per_file = defaultdict(Counter)   # file -> Counter(category)
    term_counts = defaultdict(Counter)
    env_defaults = {}
    comment_counts = Counter()
    for p in files(root):
        rel = p.relative_to(root.parent).as_posix()
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        in_doc = False
        for n, line in enumerate(lines, 1):
            # Split executable text from comments/docstrings (heuristic, Python only).
            code = line
            if p.suffix == ".py":
                if line.count('"""') % 2 == 1 or line.count("'''") % 2 == 1:
                    was, in_doc = in_doc, not in_doc
                    code = "" if was or line.strip().startswith(('"""', "'''")) else line
                elif in_doc:
                    code = ""
                code = code.split("#", 1)[0] if "#" in code and not re.search(r"[\"'][^\"']*#", code) else code
            low = code.lower()
            for cat, terms in CORPUS_TERMS.items():
                for t in terms:
                    rx = rf"(?<![\w-]){re.escape(t)}(?![\w-])"
                    if re.search(rx, low):
                        term_counts[cat][t] += 1
                        per_file[rel][cat] += 1
                        hits[cat].append((rel, n, line.strip()[:110]))
                    elif re.search(rx, line.lower()):
                        comment_counts[cat] += 1
            for cat, rx in PATTERNS.items():
                for m in rx.finditer(line):
                    if cat == "ip_address" and (m.group(0).startswith(("0.", "127.", "255.")) or
                                                not all(int(x) < 256 for x in m.group(0).split("."))):
                        continue
                    if cat == "env_default":
                        env_defaults.setdefault(m.group(1), []).append((rel, n, m.group(2)[:60]))
                        continue
                    per_file[rel][cat] += 1
                    hits[cat].append((rel, n, m.group(0)[:110]))

    print("# Hardcoded-data scan\n")
    print(f"Root: `{root.name}/` (tests, caches and audit/forensics tooling excluded)\n")
    print("## Totals by category\n\n| Category | Occurrences | Files |\n|---|---|---|")
    for cat in list(CORPUS_TERMS) + [c for c in PATTERNS if c != "env_default"]:
        fs = {h[0] for h in hits[cat]}
        print(f"| {cat} | {len(hits[cat])} | {len(fs)} |")
    print(f"| env_default (distinct variables with a literal default) | {len(env_defaults)} | "
          f"{len({x[0] for v in env_defaults.values() for x in v})} |")
    print("\nCorpus vocabulary counts below are **executable code and string literals only**; mentions inside comments "
          "and docstrings are counted separately: " + ", ".join(f"{k} {v}" for k, v in comment_counts.items()) + ".")
    print("\n## Corpus vocabulary (occurrence counts)\n")
    for cat, c in term_counts.items():
        print(f"* **{cat}:** " + ", ".join(f"{t} {k}" for t, k in c.most_common()))
    print("\n## Files ranked by corpus coupling (person + organization + location + document_rule)\n")
    print("| File | person | org | location | doc rule | other literals |\n|---|---|---|---|---|---|")
    ranked = sorted(per_file.items(), key=lambda kv: -sum(kv[1][c] for c in CORPUS_TERMS))
    for f, c in ranked[:25]:
        other = sum(v for k, v in c.items() if k not in CORPUS_TERMS)
        print(f"| `{f}` | {c['person']} | {c['organization']} | {c['location']} | {c['document_rule']} | {other} |")
    for cat in ["ip_address", "company_domain", "url", "absolute_path", "model_id", "tenant_or_user_literal",
                "queue_name", "secret_like", "person"]:
        print(f"\n## {cat} (first 40)\n")
        for f, n, t in hits[cat][:40]:
            print(f"* `{f}:{n}` `{t}`")
    print("\n## Environment variables with literal defaults in code\n")
    print("| Variable | Defaults seen (file:line = value) |\n|---|---|")
    for k in sorted(env_defaults):
        vals = "; ".join(f"`{f}:{n}`={v!r}" for f, n, v in env_defaults[k][:3])
        more = f" (+{len(env_defaults[k]) - 3})" if len(env_defaults[k]) > 3 else ""
        distinct = len({v for _, _, v in env_defaults[k]})
        flag = " **conflicting defaults**" if distinct > 1 else ""
        print(f"| `{k}` | {vals}{more}{flag} |")
    print("\n```json")
    print(json.dumps({"totals": {c: len(h) for c, h in hits.items()}, "env_defaults": len(env_defaults)}))
    print("```")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "ultimate").resolve())
