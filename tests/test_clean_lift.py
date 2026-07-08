from pathlib import Path

from contract_rag.clean.lift import format_cleaning_lift, measure_cleaning_lift
from contract_rag.config import Settings
from contract_rag.extract.extractor import FakeExtractor
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _fixture_ir(_p):
    b = lambda t, i: DocBlock(block_id=i, type=BlockType.PARAGRAPH, text=t,
                              bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                              confidence=1.0, source_engine="docling")
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf",
                      blocks=[b("The café agreement is binding under §2.", "#/b/1"),
                              b("Governing law shall be New York.", "#/b/2")],
                      metadata={})


def test_measure_cleaning_lift_quality_recovers(tmp_path: Path):
    gdir = tmp_path / "golden_set"
    gdir.mkdir()
    (gdir / "d.json").write_text(
        '{"doc_id":"d","source_pdf":"d.pdf",'
        '"facts":{"counterparty":"Acme Inc.","effective_date":"","governing_law":""}}'
    )
    settings = Settings(golden_set_dir=gdir)
    canned = ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#/b/1", confidence=0.9),
        effective_date=ExtractedClause(), governing_law=ExtractedClause(),
    )

    result = measure_cleaning_lift(settings, FakeExtractor(canned), parse_fn=_fixture_ir, seed=0)

    assert "dirty_f1" in result and "cleaned_f1" in result
    dq, cq = result["quality_pairs"][0]
    assert cq > dq                                    # cleaning recovers quality
    report = format_cleaning_lift(result)
    assert "field_f1" in report
    assert "quality_score" in report
