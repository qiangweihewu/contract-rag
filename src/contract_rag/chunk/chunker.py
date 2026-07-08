"""Structure-aware chunking: group body blocks under their heading, split on size,
keep tables whole, and carry source block_ids through so attribution survives retrieval."""
from __future__ import annotations

from contract_rag.chunk.models import Chunk
from contract_rag.ir import BlockType, DocumentIR

_HEADINGS = (BlockType.TITLE, BlockType.HEADING)
_FURNITURE = (BlockType.HEADER, BlockType.FOOTER)


def chunk_ir(ir: DocumentIR, max_chars: int = 1000) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading: str | None = None
    buf: list[str] = []
    buf_ids: list[str] = []
    buf_len = 0
    buf_page: int | None = None

    def emit(text: str, ids: list[str], page: int | None) -> None:
        text = text.strip()
        if not text or not ids:
            return
        chunks.append(Chunk(chunk_id=f"{ir.doc_id}#c{len(chunks)}", doc_id=ir.doc_id,
                            text=text, block_ids=list(ids), heading=heading, page=page))

    def flush() -> None:
        nonlocal buf, buf_ids, buf_len, buf_page
        emit("\n".join(buf), buf_ids, buf_page)
        buf, buf_ids, buf_len, buf_page = [], [], 0, None

    for b in ir.blocks:
        if b.type in _HEADINGS:
            flush()
            heading = b.text.strip() or heading
            continue
        if b.type in _FURNITURE or not b.text.strip():
            continue
        if b.type is BlockType.TABLE:
            flush()
            emit(b.text, [b.block_id], b.bbox.page if b.bbox else None)
            continue
        if buf_ids and buf_len + len(b.text) > max_chars:
            flush()
        if buf_page is None and b.bbox is not None:
            buf_page = b.bbox.page
        buf.append(b.text)
        buf_ids.append(b.block_id)
        buf_len += len(b.text)

    flush()
    return chunks
