from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, hash: str) -> Path:
        return self.root / hash

    def exists(self, hash: str) -> bool:
        return self.path_for(hash).exists()

    def put(self, path: Path) -> str:
        h = file_hash(path)
        dest = self.path_for(h)
        if not dest.exists():
            shutil.copyfile(path, dest)
        return h
