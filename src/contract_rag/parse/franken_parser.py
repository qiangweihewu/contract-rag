from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

from contract_rag.config import Settings
from contract_rag.ingest.store import file_hash
from contract_rag.ir import DocBlock, DocumentIR
from contract_rag.parse.markdown_ir import markdown_to_blocks

_PAGE_SEP = "<PAGE>"


def _count_pdf_pages(path: Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def _run_focr(path: Path, multi_page: bool, settings: Settings) -> dict:
    """One `focr` subprocess invocation for the whole document — each invocation
    loads ~3.9GB of weights, so calling it once per page would be prohibitively
    slow. `--multi-page` is only needed (and only passed) for multi-page docs."""
    binary = settings.franken_bin or "focr"
    args = [binary, "ocr", str(path), "--json"]
    if multi_page:
        args.append("--multi-page")
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"focr exited {proc.returncode}: {stderr}")
    return json.loads(proc.stdout)


def _rebuild_for_page(blocks: list[DocBlock], page: int) -> list[DocBlock]:
    """Renumber a single page's blocks (fresh from `markdown_to_blocks`, so their
    ids/parent_ids are only unique within that call) into a page-prefixed id space
    that stays unique across the whole document, and stamp `source_engine`. Pure —
    rebuilds via `model_copy`, matching the router's segment-prefixing convention."""
    id_map = {b.block_id: f"#/franken/p{page}/{i}" for i, b in enumerate(blocks)}
    out: list[DocBlock] = []
    for b in blocks:
        out.append(
            b.model_copy(
                update={
                    "block_id": id_map[b.block_id],
                    "parent_id": id_map.get(b.parent_id) if b.parent_id else None,
                    "source_engine": "frankenocr",
                }
            )
        )
    return out


def franken_json_to_blocks(payload: dict, n_pages: int) -> list[DocBlock]:
    """Convert a focr `--json` payload into `DocBlock`s, splitting the joined
    `markdown` on `<PAGE>` for multi-page results. If the separator count doesn't
    match `n_pages` (unexpected output shape), fall back to treating the whole
    markdown as a single page-1 document rather than raising — a parse should
    degrade gracefully, not sink the whole document over a page-count mismatch."""
    markdown = payload.get("markdown", "") or ""
    segments = [markdown]
    if n_pages > 1:
        candidate = [seg.strip() for seg in markdown.split(_PAGE_SEP)]
        if len(candidate) == n_pages:
            segments = candidate

    blocks: list[DocBlock] = []
    for page_idx, seg in enumerate(segments, start=1):
        blocks.extend(_rebuild_for_page(markdown_to_blocks(seg.strip()), page_idx))
    return blocks


def franken_layout_by_page(payload: dict) -> dict[str, list]:
    """Pull the per-page layout-classification arrays out of a focr payload for
    `DocumentIR.metadata["franken_layout"]` (additive — used later for coverage
    experiments, not consumed by any current layer). Keys are stringified page
    numbers so the metadata dict stays JSON-serializable."""
    if "pages" in payload:
        return {str(p.get("page")): p.get("layout", []) for p in payload["pages"]}
    if "layout" in payload:
        return {"1": payload["layout"]}
    return {}


def parse_with_franken(
    path: Path,
    settings: Settings,
    runner: Callable[[Path, bool], dict] | None = None,
    page_count_fn: Callable[[Path], int] | None = None,
) -> DocumentIR:
    path = Path(path)
    page_count_fn = page_count_fn or _count_pdf_pages
    if runner is None:
        runner = lambda p, mp: _run_focr(p, mp, settings)  # noqa: E731

    n_pages = page_count_fn(path)
    multi_page = n_pages > 1
    payload = runner(path, multi_page)

    blocks = franken_json_to_blocks(payload, n_pages)
    layout = franken_layout_by_page(payload)

    h = file_hash(path)
    return DocumentIR(
        doc_id=h,
        source_uri=path.resolve().as_uri(),
        file_hash=h,
        mime_type="application/pdf",
        blocks=blocks,
        metadata={"franken_layout": layout},
    )
