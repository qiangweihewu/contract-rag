from __future__ import annotations

from pathlib import Path

from contract_rag.connectors.base import Connector, SourceRef
from contract_rag.connectors.local import LocalFilesystemConnector


def test_list_and_fetch(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("bravo")
    (tmp_path / "ignore.md").write_text("skip")
    conn = LocalFilesystemConnector(tmp_path, suffixes=(".txt",))
    assert isinstance(conn, Connector)

    refs = sorted(conn.list_documents(), key=lambda r: r.name)
    assert [r.name for r in refs] == ["a.txt", "b.txt"]

    path = conn.fetch(refs[0])
    assert path.read_text() == "alpha"


def test_prefix_filter(tmp_path: Path):
    (tmp_path / "contract_1.txt").write_text("x")
    (tmp_path / "other.txt").write_text("y")
    conn = LocalFilesystemConnector(tmp_path, suffixes=(".txt",))
    names = [r.name for r in conn.list_documents(prefix="contract_")]
    assert names == ["contract_1.txt"]
