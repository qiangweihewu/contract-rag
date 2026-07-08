from __future__ import annotations

import re
import unicodedata
from typing import Callable

from contract_rag.ir import BlockType, DocumentIR

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WS = re.compile(r"\s+")


def _map_text(ir: DocumentIR, fn: Callable[[str], str]) -> DocumentIR:
    return ir.model_copy(
        update={"blocks": [b.model_copy(update={"text": fn(b.text)}) for b in ir.blocks]}
    )


def fix_unicode(ir: DocumentIR) -> DocumentIR:
    import ftfy

    return _map_text(ir, lambda t: unicodedata.normalize("NFC", ftfy.fix_text(t)))


def dehyphenate(ir: DocumentIR) -> DocumentIR:
    return _map_text(ir, lambda t: _HYPHEN_BREAK.sub(r"\1\2", t))


def strip_whitespace_noise(ir: DocumentIR) -> DocumentIR:
    survivors = []
    for b in ir.blocks:
        if b.type is BlockType.TABLE:
            survivors.append(b)          # preserve table rows verbatim
            continue
        new_text = _WS.sub(" ", b.text).strip()
        if new_text:
            survivors.append(b.model_copy(update={"text": new_text}))
    return ir.model_copy(update={"blocks": survivors})


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def strip_headers_footers(ir: DocumentIR, repeat_threshold: float = 0.5) -> DocumentIR:
    survivors = [b for b in ir.blocks if b.type not in (BlockType.HEADER, BlockType.FOOTER)]
    pages = {b.bbox.page for b in survivors if b.bbox is not None}
    if len(pages) >= 2:
        n_pages = len(pages)
        text_pages: dict[str, set[int]] = {}
        for b in survivors:
            if b.bbox is None:
                continue
            key = _norm(b.text)
            if key and len(key) <= 80:
                text_pages.setdefault(key, set()).add(b.bbox.page)
        repeated = {k for k, ps in text_pages.items() if len(ps) / n_pages >= repeat_threshold}
        survivors = [
            b for b in survivors if b.bbox is None or _norm(b.text) not in repeated
        ]
    return ir.model_copy(update={"blocks": survivors})


def _shingles(text: str, k: int = 3) -> frozenset[str]:
    toks = _norm(text).split()
    if not toks:
        return frozenset()
    if len(toks) < k:
        return frozenset([" ".join(toks)])
    return frozenset(" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1))


def dedupe_blocks(ir: DocumentIR, jaccard_threshold: float = 0.9) -> DocumentIR:
    survivors = []
    seen: list[frozenset[str]] = []
    for b in ir.blocks:
        sh = _shingles(b.text)
        is_dup = any(
            sh and prev and len(sh & prev) / len(sh | prev) >= jaccard_threshold
            for prev in seen
        )
        if not is_dup:
            survivors.append(b)
            seen.append(sh)
    return ir.model_copy(update={"blocks": survivors})
