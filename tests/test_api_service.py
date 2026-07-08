from contract_rag.api.service import Diagnosis, Problem, diagnose_ir, quality_problems
from contract_rag.clean.quality import compute_quality_score
from contract_rag.ir import BlockType, DocBlock, DocumentIR


def _block(bid, text, conf=1.0):
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    confidence=conf, source_engine="docling")


def _ir(blocks, doc_id="d"):
    return DocumentIR(doc_id=doc_id, source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_quality_problems_empty_for_clean_doc():
    ir = _ir([_block("b1", "This Agreement is governed by the laws of New York.")])
    assert quality_problems(compute_quality_score(ir)) == []


def test_quality_problems_flags_garble():
    ir = _ir([_block("b1", "Ã‚Â the Ã‚Â contract Ã¢â‚¬ text Ã‚Â is Ã‚Â garbled Ã‚Â here Ã‚Â now")])
    codes = {p.code for p in quality_problems(compute_quality_score(ir))}
    assert "garble" in codes
    assert all(isinstance(p, Problem) for p in quality_problems(compute_quality_score(ir)))


def test_diagnose_ir_reports_lift_and_doc_id():
    dirty = "Ã‚Â the Ã‚Â contract Ã¢â‚¬ text Ã‚Â is Ã‚Â garbled Ã‚Â here Ã‚Â and Ã‚Â there"
    ir = _ir([_block("b1", dirty)], doc_id="contract-7")
    diag = diagnose_ir(ir, redactions=2)
    assert isinstance(diag, Diagnosis)
    assert diag.doc_id == "contract-7"
    assert diag.redactions == 2
    # cleaning recovers mojibake → cleaned quality strictly higher than raw
    assert diag.cleaned_quality.quality_score > diag.raw_quality.quality_score
    assert round(diag.delta, 3) == round(
        diag.cleaned_quality.quality_score - diag.raw_quality.quality_score, 3
    )
    assert any(p.code == "garble" for p in diag.problems)


def test_diagnose_ir_doc_id_falls_back_to_ir():
    ir = _ir([_block("b1", "clean text here")], doc_id="from-ir")
    assert diagnose_ir(ir).doc_id == "from-ir"
