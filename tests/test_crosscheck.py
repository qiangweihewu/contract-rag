from __future__ import annotations

from contract_rag.clean.crosscheck import (
    CrosscheckReport,
    annotate_report,
    critical_tokens,
    crosscheck,
)
from contract_rag.clean.quality import compute_quality_score
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _ir(texts: list[str], engine: str = "paddleocr") -> DocumentIR:
    blocks = [
        DocBlock(block_id=f"#/b{i}", type=BlockType.PARAGRAPH, text=t,
                 confidence=1.0, source_engine=engine)
        for i, t in enumerate(texts)
    ]
    return DocumentIR(doc_id="d", source_uri="file:///d.pdf", file_hash="h",
                      mime_type="application/pdf", blocks=blocks)


def test_critical_tokens_formatting_invariant_and_digit_only():
    toks = critical_tokens("Total $1,200.50 due January 6, 2012 (12%)")
    assert "1200.50" in toks          # currency + thousands dropped, decimal kept
    assert "12%" in toks              # percent kept
    assert "2012" in toks and "6" in toks
    assert not any(t.isalpha() for t in toks)  # pure-alpha excluded


def test_crosscheck_flags_missing_critical_token():
    primary = _ir(["Payment due within 30 days"])
    verifier = _ir(["Payment of $12,000 due within 30 days"], engine="dots")
    cc = crosscheck(primary, verifier)
    assert cc.flagged and cc.missing_count == 1
    assert cc.missing_tokens == ["12000"]
    assert cc.verifier_engine == "dots"


def test_crosscheck_no_flag_when_primary_complete_or_verifier_empty():
    primary = _ir(["Total 12000 in 2024"])
    assert not crosscheck(primary, _ir(["Total $12,000 in 2024"], "dots")).flagged
    assert not crosscheck(primary, _ir([], "dots")).flagged


def test_crosscheck_min_missing_threshold():
    primary = _ir(["no numbers here"])
    verifier = _ir(["values 111 and 222"], "dots")
    assert crosscheck(primary, verifier).flagged
    assert not crosscheck(primary, verifier, min_missing=3).flagged


def test_annotate_report_is_additive_and_byte_identical():
    q = compute_quality_score(_ir(["Total 12000"]))
    cc = CrosscheckReport(missing_tokens=["99"], missing_count=1, flagged=True)
    q2 = annotate_report(cc, q)
    assert q2.crosscheck_missing_count == 1 and q2.crosscheck_flagged is True
    assert q.crosscheck_missing_count is None  # original untouched
    assert q2.quality_score == q.quality_score and q2.needs_review == q.needs_review
    assert q2.model_copy(update={"crosscheck_missing_count": None,
                                 "crosscheck_flagged": None}) == q
