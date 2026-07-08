from pathlib import Path

from contract_rag.ingest.store import Store, file_hash


def test_identical_bytes_hash_identically(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    assert file_hash(a) == file_hash(b)


def test_put_is_idempotent_and_dedups(tmp_path: Path):
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    store = Store(root=tmp_path / "store")

    h1 = store.put(src)
    h2 = store.put(src)

    assert h1 == h2
    assert store.exists(h1)
    assert store.path_for(h1).read_bytes() == b"%PDF-1.4 fake"
    assert len(list((tmp_path / "store").iterdir())) == 1
