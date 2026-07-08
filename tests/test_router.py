from pathlib import Path

from contract_rag.config import Settings
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR
from contract_rag.parse.probe import DocProfile, PageProfile
from contract_rag.parse.router import (
    contiguous_segments,
    page_route,
    parse,
    parse_per_page,
    route,
)


def _profile(cov: float) -> DocProfile:
    return DocProfile(page_count=10, pages_with_text=int(cov * 10), text_coverage=cov)


def _pp(page: int, has_text: bool) -> PageProfile:
    return PageProfile(page=page, char_count=500 if has_text else 0, has_text=has_text)


def test_digital_routes_to_docling():
    assert route(_profile(0.95), Settings()) == "docling"


def test_hard_doc_with_endpoint_routes_to_vlm():
    s = Settings(vlm_endpoint="http://gpu:10000/v1")
    assert route(_profile(0.1), s) == "vlm"


def test_hard_doc_without_endpoint_falls_back_to_paddle():
    assert route(_profile(0.1), Settings(vlm_endpoint=None)) == "paddleocr"


def test_threshold_is_boundary_inclusive_for_docling():
    # coverage exactly at the 0.8 default threshold -> docling
    assert route(_profile(0.8), Settings()) == "docling"


def test_parse_dispatches_to_selected_adapter(tmp_path):
    from contract_rag.ir import DocumentIR

    sentinel = DocumentIR(
        doc_id="x", source_uri="file:///x", file_hash="h",
        mime_type="application/pdf", blocks=[], metadata={"engine": "fake-docling"},
    )
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    ir = parse(
        pdf,
        Settings(),
        probe_fn=lambda _p: _profile(0.95),
        adapters={"docling": lambda _p, _s: sentinel},
    )
    assert ir.metadata["engine"] == "fake-docling"


# ---------------------------------------------------------------- per-page routing

def test_page_route_digital_vs_scanned():
    assert page_route(_pp(1, True), Settings()) == "docling"
    assert page_route(_pp(1, False), Settings()) == "paddleocr"
    assert page_route(_pp(1, False), Settings(vlm_endpoint="http://gpu/v1")) == "vlm"


def test_contiguous_segments_groups_runs():
    assert contiguous_segments(["docling", "docling", "paddleocr"]) == [
        ("docling", [0, 1]),
        ("paddleocr", [2]),
    ]
    assert contiguous_segments(["paddleocr"]) == [("paddleocr", [0])]
    # order preserved; a re-entrant engine starts a new segment
    assert contiguous_segments(["docling", "paddleocr", "docling"]) == [
        ("docling", [0]),
        ("paddleocr", [1]),
        ("docling", [2]),
    ]
    assert contiguous_segments([]) == []


def _seg_block(bid: str, page: int, engine: str, parent: str | None = None) -> DocBlock:
    return DocBlock(
        block_id=bid,
        type=BlockType.PARAGRAPH,
        text=f"{engine} p{page}",
        bbox=BoundingBox(page=page, x0=0, y0=0, x1=1, y1=1),
        parent_id=parent,
        confidence=1.0,
        source_engine=engine,
    )


def _ir(blocks: list[DocBlock]) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///d", file_hash="h",
        mime_type="application/pdf", blocks=blocks,
    )


def test_parse_per_page_single_segment_is_identical_to_single_route(tmp_path):
    """Regression guard: a pure-digital doc takes ONE segment, so the per-page path
    must return the docling adapter's IR unchanged (no split, no id-prefixing) —
    exactly what single-route parse would produce."""
    pdf = tmp_path / "pure.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    sentinel = _ir([_seg_block("#/texts/0", 1, "docling")])
    split_calls: list = []

    out = parse_per_page(
        pdf,
        Settings(),
        page_probe_fn=lambda _p: [_pp(1, True), _pp(2, True)],
        adapters={"docling": lambda _p, _s: sentinel},
        split_fn=lambda p, pages, o: split_calls.append(pages) or o,
    )
    assert out is sentinel                 # untouched adapter output
    assert out.blocks[0].block_id == "#/texts/0"  # not prefixed
    assert split_calls == []               # never split a pure doc


def test_parse_per_page_pure_scanned_routes_all_paddle(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    sentinel = _ir([_seg_block("#/ocr/0", 1, "paddleocr")])
    out = parse_per_page(
        pdf,
        Settings(),
        page_probe_fn=lambda _p: [_pp(1, False), _pp(2, False)],
        adapters={"paddleocr": lambda _p, _s: sentinel},
        split_fn=lambda p, pages, o: o,
    )
    assert out is sentinel


def test_parse_per_page_mixed_splits_merges_and_remaps_pages(tmp_path):
    """A 3-page mixed doc (2 digital + 1 scanned) → docling on pages [0,1], paddle on
    page [2]. Merged IR must: keep block order, remap sub-PDF pages to original page
    numbers, prefix ids so the two segments never collide, and stamp per-block engine."""
    pdf = tmp_path / "mixed.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    split_calls: list[list[int]] = []

    def fake_split(_p, pages, out) -> Path:
        split_calls.append(list(pages))
        # encode the segment's original pages in the returned path so the fake
        # adapter can emit a block per sub-page with sub-PDF-local page numbers
        return Path(str(out) + ".pages=" + ",".join(map(str, pages)))

    def _adapter(engine: str):
        def fn(p: Path, _s) -> DocumentIR:
            pages = [int(x) for x in str(p).split(".pages=")[1].split(",")]
            blocks = [
                # local (sub-PDF) 1-based page; ids unique within a segment but the
                # "#/b/0" of segment 0 collides with segment 1's until prefixed
                _seg_block(f"#/b/{local}", local + 1, engine, parent="#/b/root")
                for local in range(len(pages))
            ]
            return _ir(blocks)

        return fn

    out = parse_per_page(
        pdf,
        Settings(),
        page_probe_fn=lambda _p: [_pp(1, True), _pp(2, True), _pp(3, False)],
        adapters={"docling": _adapter("docling"), "paddleocr": _adapter("paddleocr")},
        split_fn=fake_split,
    )

    assert split_calls == [[0, 1], [2]]
    assert out.metadata == {"routing": "per_page", "segments": 2}
    # 2 docling blocks (pages 1,2) + 1 paddle block (page 3), in order
    assert [b.source_engine for b in out.blocks] == ["docling", "docling", "paddleocr"]
    assert [b.bbox.page for b in out.blocks] == [1, 2, 3]
    # segment-prefixed ids: the "#/b/0" from both segments no longer collides
    ids = [b.block_id for b in out.blocks]
    assert ids == ["#s0:/b/0", "#s0:/b/1", "#s1:/b/0"]
    assert len(set(ids)) == 3
    # parent refs prefixed with the SAME segment tag so they still resolve
    assert [b.parent_id for b in out.blocks] == ["#s0:/b/root", "#s0:/b/root", "#s1:/b/root"]
