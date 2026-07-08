from contract_rag.clean.pipeline import clean_ir
from contract_rag.clean.quality import compute_quality_score
from contract_rag.eval.dirtify import (
    dirtify,
    inject_hyphenation,
    inject_mojibake,
    inject_near_duplicates,
    inject_repeated_headers,
    inject_whitespace_noise,
)
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, bid="b"):
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _clean_fixture():
    # includes non-ASCII so mojibake injection produces detectable garble
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf",
                      blocks=[_b("The café agreement is binding under §2.", "a"),
                              _b("Governing law shall be New York — exclusively.", "b"),
                              _b("Payment milestones are enumerated below.", "c")],
                      metadata={})


def test_inject_mojibake_then_fix_unicode_round_trips():
    dirty = inject_mojibake(_clean_fixture(), seed=1, rate=1.0)
    assert any(compute_quality_score(DocumentIR(
        doc_id="d", source_uri="x", file_hash="h", mime_type="application/pdf",
        blocks=[b], metadata={})).garble_ratio > 0 for b in dirty.blocks)


def test_mojibake_garbles_and_is_recoverable_on_pure_ascii():
    # the bug: utf-8->latin-1 was a no-op on ASCII, so real contracts never got garbled
    ascii_ir = DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                          mime_type="application/pdf",
                          blocks=[_b("The parties agree to the terms set forth herein.", "a")],
                          metadata={})
    dirty = inject_mojibake(ascii_ir, seed=1, rate=1.0)
    assert compute_quality_score(dirty).garble_ratio > 0          # ASCII text now garbled
    cleaned = clean_ir(dirty)
    assert compute_quality_score(cleaned).garble_ratio == 0       # ftfy + whitespace recovered it
    assert "parties agree to the terms" in cleaned.blocks[0].text


def test_inject_repeated_headers_adds_header_blocks():
    out = inject_repeated_headers(_clean_fixture(), copies=3)
    assert sum(1 for b in out.blocks if b.type is BlockType.HEADER) == 3


def test_inject_near_duplicates_adds_duplicates():
    base = _clean_fixture()
    out = inject_near_duplicates(base, seed=1, rate=1.0)
    assert len(out.blocks) > len(base.blocks)


def test_dirtify_is_seed_deterministic():
    a = dirtify(_clean_fixture(), seed=7)
    b = dirtify(_clean_fixture(), seed=7)
    assert [x.text for x in a.blocks] == [x.text for x in b.blocks]


def test_clean_recovers_quality_after_dirtify():
    clean = _clean_fixture()
    dirty = dirtify(clean, seed=3)
    cleaned = clean_ir(dirty)
    dq = compute_quality_score(dirty).quality_score
    cq = compute_quality_score(cleaned).quality_score
    assert cq > dq                                   # cleaning lifts quality
    assert not any(b.type is BlockType.HEADER for b in cleaned.blocks)  # headers gone


def _realistic_ascii_contract(n: int = 20):
    blocks = [_b(f"Section {i}. The parties agree to the terms and obligations set "
                 f"forth in this clause number {i} of the agreement.", f"b{i}") for i in range(n)]
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_dirtify_is_harsh_enough_to_drop_below_review_line_and_recover():
    # The default dirtify must produce a genuinely degraded document (well past the
    # needs_review 0.75 line) whose quality cleaning then substantially recovers — the
    # spec's dirty->clean story. (Floor ~0.5 on clean digital contracts: table-integrity
    # 0.25 + confidence 0.15 are maxed and untouchable by *recoverable* dirt.)
    ir = _realistic_ascii_contract(20)
    dirty = dirtify(ir, seed=1)
    cleaned = clean_ir(dirty)
    dq = compute_quality_score(dirty).quality_score
    cq = compute_quality_score(cleaned).quality_score
    assert dq < 0.70                                  # harsher than the old ~0.80 on this fixture
    assert cq > 0.90                                  # cleaning recovers
    assert cq - dq > 0.25                             # a substantial, demonstrable lift
