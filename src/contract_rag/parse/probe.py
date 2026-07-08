from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class DocProfile(BaseModel):
    page_count: int
    pages_with_text: int
    text_coverage: float


class PageProfile(BaseModel):
    """Per-page text-layer probe. `has_text` is the digital-vs-scanned signal the
    document-level `DocProfile.text_coverage` averages away — a mixed PDF (digital
    body + scanned annex) has some pages True and some False."""

    page: int  # 1-based
    char_count: int
    has_text: bool


def profile_from_counts(page_count: int, pages_with_text: int) -> DocProfile:
    coverage = (pages_with_text / page_count) if page_count else 0.0
    return DocProfile(
        page_count=page_count, pages_with_text=pages_with_text, text_coverage=coverage
    )


def profile_from_pages(pages: list[PageProfile]) -> DocProfile:
    """Roll per-page profiles up into the document-level profile the router uses."""
    return profile_from_counts(len(pages), sum(1 for p in pages if p.has_text))


def _page_char_count(page) -> int:
    """Character count for one already-open pypdfium2 page; the page is always
    closed. Any failure while reading it (corrupt page data, decode error, missing
    text layer) degrades to 0 chars instead of propagating — that page then reads
    as scanned (`has_text=False`) rather than crashing the whole probe."""
    try:
        textpage = page.get_textpage()
        try:
            text = textpage.get_text_range()
        finally:
            textpage.close()
        return len(text.strip())
    except Exception:
        return 0
    finally:
        page.close()


def _page_char_counts(path: Path) -> list[int]:
    """Stripped-text character count for every page (pypdfium2, one pass). A
    single corrupt page degrades to 0 chars (see `_page_char_count`) rather than
    crashing the whole probe."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return [_page_char_count(page) for page in pdf]
    finally:
        pdf.close()


def probe_pages(path: Path, min_chars: int = 1) -> list[PageProfile]:
    """Per-page text-layer profile. A page counts as digital (`has_text`) when its
    stripped text layer has at least `min_chars` characters; scanned/image-only pages
    have none. `min_chars=1` reproduces `probe_document`'s truthiness test exactly."""
    return [
        PageProfile(page=i, char_count=c, has_text=c >= min_chars)
        for i, c in enumerate(_page_char_counts(path), start=1)
    ]


def probe_document(path: Path) -> DocProfile:
    counts = _page_char_counts(path)
    return profile_from_counts(len(counts), sum(1 for c in counts if c > 0))
