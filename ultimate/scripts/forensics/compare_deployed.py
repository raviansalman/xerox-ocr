#!/usr/bin/env python3
"""Compare a deployed source tree with this repository and classify every difference by search area.

Read-only. Nothing is restored or copied. Usage (see docs/SEARCH_FORENSICS.md section 3):

    # on each API / worker host
    docker exec <container> tar c -C /app src ultimate_ui.py search_api.py > deployed_<host>.tar
    mkdir deployed && tar xf deployed_<host>.tar -C deployed

    python3 -I scripts/forensics/compare_deployed.py deployed . > deployed_vs_repo.md

The second argument is the repository's `ultimate/` directory.
"""
import ast
import difflib
import hashlib
import sys
from pathlib import Path

AREAS = [
    ("search", ("ultimate_ui.py", "search_api.py", "ultimate_vector_integration.py", "semantic_pipeline.py")),
    ("semantic_utils", ("semantic_utils.py",)),
    ("query understanding", ("query_enhancement.py", "temporal_engine.py")),
    ("ranking", ("constraint_ranking.py", "semantic_pipeline.py", "ultimate_ui.py")),
    ("metadata indexing", ("semantic_components.py", "semantic_pipeline.py")),
    ("milvus", ("vector_db_milvus_server.py",)),
    ("embeddings", ("embeddings.py", "embedder_service.py")),
    ("chunking", ("ultimate_vector_integration.py", "semantic_components.py")),
    ("ocr", ("ultimate_search_processor.py", "ultimate_tasks.py")),
]

# Names whose presence decides whether the dead search stack runs (docs/FORENSICS.md F1/F2).
PROBES = {
    "src/semantic/semantic_utils.py": ["LOCATION_PEERS"],
    "src/semantic/semantic_components.py": ["threading"],
}


def files(root: Path):
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*.py") if not {"__pycache__", "tests", "scripts", ".pytest_cache"} & set(p.parts)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def symbols(p: Path):
    """Top-level and class-level defs plus module-level assignments and imported names."""
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as e:
        return {f"<syntax error line {e.lineno}>"}
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.ClassDef):
            out.add(node.name)
            out.update(f"{node.name}.{n.name}" for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in node.names)
    return out


def areas_for(rel: str):
    name = rel.split("/")[-1]
    return [a for a, names in AREAS if name in names] or ["other"]


def main(deployed: Path, repo: Path):
    d, r = files(deployed), files(repo)
    print(f"# Deployed tree vs repository\n\ndeployed: `{deployed}`  repository: `{repo}`\n")
    print("## Decisive probes\n")
    for rel, names in PROBES.items():
        for n in names:
            dep = n in symbols(d[rel]) if rel in d else None
            rep = n in symbols(r[rel]) if rel in r else None
            print(f"* `{n}` in `{rel}`: deployed={dep} repository={rep}")
    print("\n## Files only in the deployed tree\n")
    for rel in sorted(set(d) - set(r)):
        print(f"* `{rel}` ({', '.join(areas_for(rel))}), {len(d[rel].read_text(errors='replace').splitlines())} lines")
    print("\n## Files only in the repository\n")
    for rel in sorted(set(r) - set(d)):
        print(f"* `{rel}` ({', '.join(areas_for(rel))})")
    print("\n## Files that differ\n")
    print("| File | Areas | Lines +/- | Symbols only deployed | Symbols only in repo |\n|---|---|---|---|---|")
    for rel in sorted(set(d) & set(r)):
        if sha(d[rel]) == sha(r[rel]):
            continue
        a = r[rel].read_text(errors="replace").splitlines()
        b = d[rel].read_text(errors="replace").splitlines()
        diff = list(difflib.unified_diff(a, b, lineterm="", n=0))
        plus = sum(1 for x in diff if x.startswith("+") and not x.startswith("+++"))
        minus = sum(1 for x in diff if x.startswith("-") and not x.startswith("---"))
        sd, sr = symbols(d[rel]), symbols(r[rel])
        print(f"| `{rel}` | {', '.join(areas_for(rel))} | +{plus}/-{minus} | "
              f"{', '.join(sorted(sd - sr)) or '-'} | {', '.join(sorted(sr - sd)) or '-'} |")
    print("\nFull diffs: `diff -ru <repo> <deployed>`. Do not copy deployed code into the repository "
          "without review (docs/SEARCH_FORENSICS.md section 10).")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
