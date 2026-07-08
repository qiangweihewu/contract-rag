from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from contract_rag.connectors.base import SourceRef

_DEFAULT_SUFFIXES = (".pdf", ".docx", ".txt")


class LocalFilesystemConnector:
    """Credential-free default connector: documents are files under a root directory.
    The drop-in source for the eval/demo data dir and the production default."""

    name = "local"

    def __init__(self, root: Path | str, *, suffixes: tuple[str, ...] = _DEFAULT_SUFFIXES) -> None:
        self.root = Path(root)
        self.suffixes = tuple(s.lower() for s in suffixes)

    def list_documents(self, *, prefix: str | None = None) -> Iterable[SourceRef]:
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in self.suffixes:
                continue
            if prefix and not p.name.startswith(prefix):
                continue
            yield SourceRef(id=str(p), name=p.name, uri=p.as_uri())

    def fetch(self, ref: SourceRef) -> Path:
        return Path(ref.id)
