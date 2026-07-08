from __future__ import annotations

from contract_rag.verticals.nda.eval import NDA_GOLDEN_DIR, evaluate_nda, text_to_ir


def test_text_to_ir_builds_paragraph_blocks():
    ir = text_to_ir(NDA_GOLDEN_DIR / "nda_01.txt")
    assert len(ir.blocks) >= 5
    assert ir.blocks[0].source_engine == "synthetic"
    assert any("Disclosing Party" in b.text for b in ir.blocks)


def test_evaluate_nda_runs_over_golden_set():
    agg = evaluate_nda()
    assert agg["n_docs"] == 8
    assert "field_f1" in agg and "source_accuracy" in agg
