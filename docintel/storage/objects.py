"""Object storage for original files. Originals are content-addressed per tenant and never modified.

The filesystem backend works with local disks, NFS or any mounted volume (including S3 through a FUSE mount).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from docintel.config import get_settings
from docintel.security import is_valid_tenant_id


class ObjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def check_writable(self) -> None:
        """Raise when originals cannot be stored (missing or read-only volume, wrong owner)."""
        probe = self.root / "_incoming" / f".ready-{os.getpid()}"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
        probe.unlink()

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("object key escapes the storage root")
        return p

    def put_stream(self, tenant_id: str, stream: BinaryIO, max_bytes: int) -> tuple[str, str, int]:
        """Store a stream; returns (key, sha256, size). Raises ValueError when the size limit is exceeded."""
        if not is_valid_tenant_id(tenant_id):
            raise ValueError("invalid tenant id")
        tmp_dir = self.root / "_incoming"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        digest, size = hashlib.sha256(), 0
        fd, tmp = tempfile.mkstemp(dir=tmp_dir)
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"file exceeds the {max_bytes} byte limit")
                    digest.update(chunk)
                    out.write(chunk)
            sha = digest.hexdigest()
            key = f"{tenant_id}/originals/{sha[:2]}/{sha}"
            dest = self._path(key)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                os.unlink(tmp)
            else:
                os.replace(tmp, dest)
            return key, sha, size
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def put_bytes(self, tenant_id: str, data: bytes, max_bytes: int) -> tuple[str, str, int]:
        import io
        return self.put_stream(tenant_id, io.BytesIO(data), max_bytes)

    def path(self, key: str) -> Path:
        p = self._path(key)
        if not p.exists():
            raise FileNotFoundError(key)
        return p

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def copy_to(self, key: str, dest: Path) -> Path:
        shutil.copyfile(self.path(key), dest)
        return dest


_store: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    global _store
    if _store is None:
        _store = ObjectStore(get_settings().storage_dir)
    return _store


def set_object_store(store: ObjectStore | None) -> None:
    global _store
    _store = store
