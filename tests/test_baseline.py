from pathlib import Path

import pytest

from contract_rag.baseline import format_report, run_baseline
from contract_rag.config import Settings
from contract_rag.extract.extractor import FakeExtractor
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _stub_ir() -> DocumentIR:
    return DocumentIR(
        doc_id="msa-acme", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH,
                     text="entered into by Acme Inc.",
                     bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                     confidence=1.0, source_engine="docling"),
        ],
        metadata={},
    )


def test_run_baseline_scores_against_golden(tmp_path: Path):
    gdir = tmp_path / "golden_set"
    gdir.mkdir()
    (gdir / "msa-acme.json").write_text(
        '{"doc_id":"msa-acme","source_pdf":"msa-acme.pdf",'
        '"facts":{"counterparty":"Acme Inc.","effective_date":"","governing_law":""}}'
    )
    settings = Settings(golden_set_dir=gdir)

    canned = ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#/b/1", confidence=0.9),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(),
    )
    agg = run_baseline(settings, extractor=FakeExtractor(canned), parse_fn=lambda _pdf: _stub_ir())
    assert agg["per_field"]["counterparty"] == 1.0
    assert agg["source_accuracy"] == 1.0
    assert "field_f1" in format_report(agg)


def test_run_baseline_raises_on_empty_golden_set(tmp_path: Path):
    gdir = tmp_path / "golden_set"
    gdir.mkdir()
    settings = Settings(golden_set_dir=gdir)
    canned = ContractFacts(
        counterparty=ExtractedClause(),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(),
    )
    with pytest.raises(ValueError, match="golden set is empty"):
        run_baseline(settings, extractor=FakeExtractor(canned), parse_fn=lambda _pdf: _stub_ir())
