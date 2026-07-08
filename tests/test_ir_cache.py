from pathlib import Path

from contract_rag.eval.ir_cache import ir_cache
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _ir(stem: str) -> DocumentIR:
    return DocumentIR(
        doc_id=stem, source_uri=f"file:///{stem}.pdf", file_hash="h",
        mime_type="application/pdf",
        blocks=[DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH, text=f"body of {stem}",
                         bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                         confidence=1.0, source_engine="docling")],
        metadata={},
    )


def test_ir_cache_parses_once_then_reuses(tmp_path):
    calls = {"n": 0}

    def fake_parse(p: Path) -> DocumentIR:
        calls["n"] += 1
        return _ir(p.stem)

    parse = ir_cache(tmp_path, fake_parse)
    first = parse(Path("data/foo.pdf"))
    second = parse(Path("data/foo.pdf"))

    assert calls["n"] == 1                              # parsed once, served from cache after
    assert (tmp_path / "foo.ir.json").exists()
    assert first.doc_id == second.doc_id == "foo"
    assert second.blocks[0].text == "body of foo"      # round-trips faithfully through JSON


def test_ir_cache_keys_by_pdf_name(tmp_path):
    def fake_parse(p: Path) -> DocumentIR:
        return _ir(p.stem)

    parse = ir_cache(tmp_path, fake_parse)
    a = parse(Path("data/a.pdf"))
    b = parse(Path("data/b.pdf"))

    assert a.doc_id == "a" and b.doc_id == "b"          # distinct docs cached separately
    assert (tmp_path / "a.ir.json").exists()
    assert (tmp_path / "b.ir.json").exists()
