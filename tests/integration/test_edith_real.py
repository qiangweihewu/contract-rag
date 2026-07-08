"""Gated on EDITH_DIR pointing at a real EDiTh snapshot (MASTER_INDEX.csv, and
by_entity/ PDFs or by_entity.tar.gz). Validates the schema mapping and exercises the
per-page probe + routing analysis on one real mixed document — no OCR here (that's
the `python -m contract_rag.eval.edith` harness with EDITH_PARSE_SIZE)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_dir = os.environ.get("EDITH_DIR", "")
pytestmark = pytest.mark.skipif(
    not (_dir and (Path(_dir) / "MASTER_INDEX.csv").exists()),
    reason="EDITH_DIR not set to an EDiTh snapshot",
)


def test_index_has_mixed_docs():
    from contract_rag.eval.edith import load_index, select_mixed

    entries = load_index(Path(_dir))
    assert entries, "empty MASTER_INDEX"
    mixed = select_mixed(entries)
    assert mixed, "no mixed/scanned docs found — schema drift on the `format` column?"
    # README declares ~85 mixed docs; we accept the labelled subset as-is
    assert all(e.rel_path().startswith("by_entity/") for e in mixed)


def test_probe_pages_on_a_real_mixed_doc_is_split():
    pytest.importorskip("pypdfium2")
    from contract_rag.config import Settings
    from contract_rag.eval.edith import (
        _ensure_pdfs,
        analyze_routing,
        load_index,
        select_mixed,
    )
    from contract_rag.parse.probe import probe_pages

    entries = load_index(Path(_dir))
    mixed = select_mixed(entries)
    if not any(e.format == "mixed" for e in mixed):
        pytest.skip("snapshot has no `mixed` docs, only scanned fallback")

    # find the first mixed doc whose PDF we can materialize
    doc = None
    for e in mixed:
        if e.format != "mixed":
            continue
        _ensure_pdfs(Path(_dir), [e])
        if (Path(_dir) / e.rel_path()).exists():
            doc = e
            break
    if doc is None:
        pytest.skip("no mixed PDF could be materialized from the snapshot")

    pages = probe_pages(Path(_dir) / doc.rel_path())
    assert len(pages) >= 2
    # a genuinely mixed doc has BOTH a digital and a scanned page
    assert any(p.has_text for p in pages)
    assert any(not p.has_text for p in pages)
    r = analyze_routing(doc.filename, pages, Settings())
    # the single-engine decision necessarily misroutes at least one page group
    assert r.n_misrouted_pages >= 1
