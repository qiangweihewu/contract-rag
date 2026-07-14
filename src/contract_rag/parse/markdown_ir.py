from __future__ import annotations

import re

from contract_rag.ir import BlockType, DocBlock

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def _new_block(
    i: int, btype: BlockType, text: str, parent_id: str | None,
    *, engine: str, id_prefix: str,
) -> DocBlock:
    return DocBlock(
        block_id=f"{id_prefix}/{i}",
        type=btype,
        text=text,
        bbox=None,
        parent_id=parent_id,
        confidence=1.0,
        source_engine=engine,
    )


def markdown_to_blocks(
    md: str, *, engine: str = "unlimited-ocr", id_prefix: str = "#/vlm"
) -> list[DocBlock]:
    lines = md.splitlines()
    blocks: list[DocBlock] = []
    # heading stack: list of (level, block_id) for parent resolution
    heading_stack: list[tuple[int, str]] = []
    para_buf: list[str] = []
    table_buf: list[str] = []
    i = 0

    def parent_for(level: int) -> str | None:
        for lvl, bid in reversed(heading_stack):
            if lvl < level:
                return bid
        return None

    def flush_para() -> None:
        nonlocal i
        if para_buf:
            text = " ".join(s.strip() for s in para_buf).strip()
            if text:
                blocks.append(_new_block(i, BlockType.PARAGRAPH, text, parent_for(99), engine=engine, id_prefix=id_prefix))
                i += 1
            para_buf.clear()

    def flush_table() -> None:
        nonlocal i
        if table_buf:
            blocks.append(_new_block(i, BlockType.TABLE, "\n".join(table_buf), parent_for(99), engine=engine, id_prefix=id_prefix))
            i += 1
            table_buf.clear()

    for raw in lines:
        if _TABLE_ROW.match(raw):
            flush_para()
            table_buf.append(raw.rstrip())
            continue
        flush_table()

        if not raw.strip():
            flush_para()
            continue

        m = _HEADING.match(raw)
        if m:
            flush_para()
            level = len(m.group(1))
            text = m.group(2).strip()
            btype = BlockType.TITLE if level == 1 else BlockType.HEADING
            block = _new_block(i, btype, text, parent_for(level), engine=engine, id_prefix=id_prefix)
            blocks.append(block)
            heading_stack[:] = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, block.block_id))
            i += 1
            continue

        lm = _LIST.match(raw)
        if lm:
            flush_para()
            blocks.append(_new_block(i, BlockType.LIST_ITEM, lm.group(1).strip(), parent_for(99), engine=engine, id_prefix=id_prefix))
            i += 1
            continue

        para_buf.append(raw)

    flush_para()
    flush_table()
    return blocks
