from __future__ import annotations

from contract_rag.chunk.chunker import chunk_ir
from contract_rag.enrich.enricher import enrich_chunks
from contract_rag.extract.verify import verify
from contract_rag.verticals.nda.eval import NDA_GOLDEN_DIR, evaluate_nda, text_to_ir
from contract_rag.verticals.nda.vertical import NDAVertical


def test_field_f1_and_source_attribution():
    agg = evaluate_nda()
    # synthetic set: the rules are designed to clear this with margin (honest floor, not rigged 1.0)
    assert agg["field_f1"] >= 0.70, f"field_f1={agg['field_f1']:.3f}"
    # rule extractor cites verbatim spans -> source-attribution is 1.0 by construction
    assert agg["source_accuracy"] == 1.0, f"source_accuracy={agg['source_accuracy']:.3f}"


def test_whole_pipeline_is_vertical_agnostic_for_nda():
    v = NDAVertical()
    ir = text_to_ir(NDA_GOLDEN_DIR / "nda_01.txt")
    # extraction
    facts = v.rule_extractor.extract(ir)
    # verify() works for free with the nda vertical
    report = verify(facts, ir, vertical=v)
    assert report.checks["governing_law"].passed is True
    # enrich_chunks works for free with the nda vertical
    enriched = enrich_chunks(chunk_ir(ir), vertical=v)
    assert enriched
    assert any(c.clause_type in {"confidentiality", "governing_law", "return_of_materials", "term"}
               for c in enriched)
