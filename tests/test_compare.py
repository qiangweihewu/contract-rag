from pathlib import Path

from contract_rag.compare import compare_parsers, format_comparison
from contract_rag.config import Settings
from contract_rag.extract.extractor import FakeExtractor
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _ir(text: str) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH, text=text,
                         bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                         confidence=1.0, source_engine="docling")],
        metadata={},
    )


def test_compare_parsers_returns_two_aggregates(tmp_path: Path):
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
    extractor = FakeExtractor(canned)

    docling_agg, router_agg = compare_parsers(
        settings,
        extractor,
        docling_fn=lambda _p: _ir("entered into by Acme Inc."),
        router_fn=lambda _p: _ir("entered into by Acme Inc."),
    )
    assert docling_agg["per_field"]["counterparty"] == 1.0
    assert router_agg["per_field"]["counterparty"] == 1.0
    report = format_comparison(docling_agg, router_agg)
    assert "field_f1" in report
    assert "delta" in report.lower()


def test_compare_parsers_defaults_router_fn_to_router_parse(tmp_path, monkeypatch):
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

    calls = []

    def fake_parse(path, s):
        calls.append((path, s))
        return _ir("entered into by Acme Inc.")

    monkeypatch.setattr("contract_rag.parse.router.parse", fake_parse)

    # router_fn omitted -> default branch must lazily use router.parse, closed over settings
    docling_agg, router_agg = compare_parsers(
        settings, FakeExtractor(canned), docling_fn=lambda _p: _ir("entered into by Acme Inc.")
    )
    assert calls, "default router_fn never called router.parse"
    assert calls[0][1] is settings  # the lambda closed over settings
    assert router_agg["per_field"]["counterparty"] == 1.0
