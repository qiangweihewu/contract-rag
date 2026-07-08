from contract_rag.eval.golden import GoldenDoc
from contract_rag.eval.versioning import (
    build_manifest,
    diff_manifests,
    read_manifest,
    version_id,
    write_manifest,
)


def _g(doc_id, facts):
    return GoldenDoc(doc_id=doc_id, source_pdf=f"{doc_id}.pdf", facts=facts)


def test_version_id_is_order_independent():
    a = [_g("d1", {"counterparty": "Acme"}), _g("d2", {"governing_law": "NY"})]
    b = list(reversed(a))
    assert version_id(a) == version_id(b)


def test_version_id_changes_when_a_fact_changes():
    a = [_g("d1", {"counterparty": "Acme"})]
    b = [_g("d1", {"counterparty": "Beta"})]
    assert version_id(a) != version_id(b)


def test_manifest_roundtrip(tmp_path):
    golden = [_g("d1", {"counterparty": "Acme"}), _g("d2", {"governing_law": "NY"})]
    manifest = build_manifest(golden, created_at="2026-06-30")
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    back = read_manifest(path)
    assert back.version == manifest.version
    assert back.n_docs == 2
    assert back.doc_ids == ["d1", "d2"]


def test_diff_detects_added_and_removed():
    old = build_manifest([_g("d1", {"counterparty": "Acme"})])
    new = build_manifest([_g("d1", {"counterparty": "Acme"}), _g("d2", {"governing_law": "NY"})])
    diff = diff_manifests(old, new)
    assert diff["added"] == ["d2"]
    assert diff["removed"] == []
    assert diff["version_changed"] is True
