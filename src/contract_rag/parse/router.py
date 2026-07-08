from __future__ import annotations

from pathlib import Path
from typing import Callable

from contract_rag.config import Settings
from contract_rag.ir import DocBlock, DocumentIR
from contract_rag.parse.probe import (
    DocProfile,
    PageProfile,
    probe_document,
    probe_pages,
    profile_from_counts,
)


def route(profile: DocProfile, settings: Settings) -> str:
    if profile.text_coverage >= settings.text_coverage_threshold:
        return "docling"
    if settings.vlm_endpoint:
        return "vlm"
    if settings.franken_bin:
        return "frankenocr"
    return "paddleocr"


def page_route(page: PageProfile, settings: Settings) -> str:
    """Route a single page: a digital (`has_text`) page behaves like a coverage-1.0
    document (→ docling), a scanned page like coverage-0.0 (→ vlm/paddle). Reuses the
    document-level `route()` so the per-page and whole-doc decisions stay consistent."""
    profile = profile_from_counts(page_count=1, pages_with_text=1 if page.has_text else 0)
    return route(profile, settings)


def contiguous_segments(routes: list[str]) -> list[tuple[str, list[int]]]:
    """Group consecutive same-engine page indices (0-based) into contiguous segments,
    preserving reading order. `["docling","docling","paddleocr"]` →
    `[("docling",[0,1]),("paddleocr",[2])]`. Pure."""
    segments: list[tuple[str, list[int]]] = []
    for i, eng in enumerate(routes):
        if segments and segments[-1][0] == eng:
            segments[-1][1].append(i)
        else:
            segments.append((eng, [i]))
    return segments


def _default_adapters() -> dict[str, Callable[[Path, Settings], DocumentIR]]:
    from contract_rag.parse.docling_parser import parse_with_docling
    from contract_rag.parse.franken_parser import parse_with_franken
    from contract_rag.parse.paddle_parser import parse_with_paddle
    from contract_rag.parse.vlm_parser import parse_with_vlm

    return {
        "docling": lambda p, _s: parse_with_docling(p),
        "vlm": lambda p, s: parse_with_vlm(p, s),
        "frankenocr": lambda p, s: parse_with_franken(p, s),
        "paddleocr": lambda p, _s: parse_with_paddle(p),
    }


def split_pdf_pages(path: Path, pages_0based: list[int], out_path: Path) -> Path:
    """Write a sub-PDF containing only `pages_0based` (in that order) to `out_path`,
    using pypdfium2 (already a runtime dep via probe). The seam the per-page router
    uses to hand each contiguous page-range to the right parser."""
    import pypdfium2 as pdfium

    src = pdfium.PdfDocument(str(path))
    try:
        new = pdfium.PdfDocument.new()
        new.import_pages(src, pages=list(pages_0based))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new.save(str(out_path))
        new.close()
    finally:
        src.close()
    return out_path


def _remap_segment_blocks(
    blocks: list[DocBlock], seg_idx: int, seg_pages_0based: list[int]
) -> list[DocBlock]:
    """Rewrite a segment IR's blocks into the merged document's coordinate space:
    - `bbox.page` (1-based within the sub-PDF) → original 1-based page number;
    - `block_id`/`parent_id` prefixed with the segment index so ids stay unique when
      two segments share a parser's id scheme (e.g. two docling runs both emit
      `#/texts/0`). Pure and immutable (rebuilds each block)."""

    def _pref(bid: str | None) -> str | None:
        return None if bid is None else f"#s{seg_idx}:{bid.lstrip('#')}"

    out: list[DocBlock] = []
    for b in blocks:
        bbox = b.bbox
        if bbox is not None and 1 <= bbox.page <= len(seg_pages_0based):
            bbox = bbox.model_copy(update={"page": seg_pages_0based[bbox.page - 1] + 1})
        out.append(
            b.model_copy(
                update={
                    "block_id": _pref(b.block_id),
                    "parent_id": _pref(b.parent_id),
                    "bbox": bbox,
                }
            )
        )
    return out


def parse_per_page(
    path: Path,
    settings: Settings,
    page_probe_fn: Callable[[Path], list[PageProfile]] | None = None,
    adapters: dict[str, Callable[[Path, Settings], DocumentIR]] | None = None,
    split_fn: Callable[[Path, list[int], Path], Path] | None = None,
    min_chars: int = 1,
) -> DocumentIR:
    """Route each page to the parser its text-layer warrants, then merge back into one
    `DocumentIR` (block order + original page numbers + per-block `source_engine`
    preserved). A pure-digital or pure-scanned doc collapses to a single segment and
    is parsed exactly as the single-route `parse()` would — no split, byte-identical.
    Seams (`page_probe_fn`, `adapters`, `split_fn`) keep it unit-testable dep-free."""
    import tempfile

    page_probe_fn = page_probe_fn or (lambda p: probe_pages(p, min_chars=min_chars))
    if adapters is None:
        adapters = _default_adapters()
    split_fn = split_fn or split_pdf_pages

    pages = page_probe_fn(path)
    routes = [page_route(pp, settings) for pp in pages]
    segments = contiguous_segments(routes)

    if len(segments) <= 1:  # pure doc — identical to the single-route path
        engine = segments[0][0] if segments else route(probe_document(path), settings)
        return adapters[engine](path, settings)

    merged: list[DocBlock] = []
    with tempfile.TemporaryDirectory() as d:
        for seg_idx, (engine, seg_pages) in enumerate(segments):
            sub = split_fn(Path(path), seg_pages, Path(d) / f"seg{seg_idx}.pdf")
            seg_ir = adapters[engine](Path(sub), settings)
            merged.extend(_remap_segment_blocks(seg_ir.blocks, seg_idx, seg_pages))

    from contract_rag.ingest.store import file_hash

    h = file_hash(Path(path))
    return DocumentIR(
        doc_id=h,
        source_uri=Path(path).resolve().as_uri(),
        file_hash=h,
        mime_type="application/pdf",
        blocks=merged,
        metadata={"routing": "per_page", "segments": len(segments)},
    )


def parse(
    path: Path,
    settings: Settings,
    probe_fn: Callable[[Path], DocProfile] | None = None,
    adapters: dict[str, Callable[[Path, Settings], DocumentIR]] | None = None,
    per_page: bool = False,
    page_probe_fn: Callable[[Path], list[PageProfile]] | None = None,
    split_fn: Callable[[Path, list[int], Path], Path] | None = None,
) -> DocumentIR:
    """Single-route by default (byte-identical to before). `per_page=True` opts into
    the mixed-document router, which probes each page and dispatches page-ranges to the
    parser each needs — for pure docs the two paths agree exactly."""
    if per_page:
        return parse_per_page(
            path, settings, page_probe_fn=page_probe_fn, adapters=adapters, split_fn=split_fn
        )
    probe_fn = probe_fn or probe_document
    adapters = adapters or _default_adapters()
    engine = route(probe_fn(path), settings)
    return adapters[engine](path, settings)
