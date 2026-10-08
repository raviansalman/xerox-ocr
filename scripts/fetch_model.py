"""Place a registry model in a folder the embedding service loads offline, and verify its weights.

    python scripts/fetch_model.py all-mpnet-base-v2 /models                          # download from the model hub
    python scripts/fetch_model.py all-mpnet-base-v2 /models --source deploy/models   # or copy a local folder
    python scripts/fetch_model.py ms-marco-MiniLM-L-6-v2 /models --reranker

The model ends up in <dest>/<key>/. Its model.safetensors must match the SHA-256 pinned in
docintel/model_registry.yaml (and the download uses the pinned revision), so a changed model is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key")
    ap.add_argument("dest")
    ap.add_argument("--source", help="folder with <key>/ subfolders to copy instead of downloading")
    ap.add_argument("--reranker", action="store_true", help="the key names a reranker, not an embedding model")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from docintel.model_registry import model_spec, reranker_spec

    spec = reranker_spec(a.key) if a.reranker else model_spec(a.key)
    target = Path(a.dest) / a.key
    local = Path(a.source) / a.key if a.source else None
    if local is not None and local.is_dir():
        shutil.copytree(local, target, dirs_exist_ok=True, symlinks=False)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(spec.name, revision=spec.revision or None, local_dir=str(target))
    weights = target / "model.safetensors"
    if not weights.exists():
        print(f"{weights} is missing", file=sys.stderr)
        return 1
    if spec.sha256 and _sha256(weights) != spec.sha256:
        print(f"checksum mismatch for {weights}: expected {spec.sha256}", file=sys.stderr)
        return 1
    if not a.reranker:
        from sentence_transformers import SentenceTransformer
        if SentenceTransformer(str(target), device="cpu").get_sentence_embedding_dimension() != spec.dimension:
            print("dimension does not match the registry", file=sys.stderr)
            return 1
    print(f"{spec.name} ready in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
