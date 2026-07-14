from contract_rag.ir import BlockType
from contract_rag.parse.markdown_ir import markdown_to_blocks

SAMPLE = """\
# Master Services Agreement

This Agreement is entered into by Acme Inc.

## Payment Terms

| Milestone | Amount |
| --- | --- |
| Kickoff | $10,000 |
| Delivery | $20,000 |

- Net 30 payment
- Late fee 1.5%
"""


def test_headings_paragraphs_lists_and_table_block():
    blocks = markdown_to_blocks(SAMPLE)
    types = [b.type for b in blocks]
    assert BlockType.TITLE in types
    assert BlockType.HEADING in types
    assert BlockType.PARAGRAPH in types
    assert types.count(BlockType.TABLE) == 1  # the whole pipe-table is ONE block
    assert types.count(BlockType.LIST_ITEM) == 2


def test_table_block_preserves_raw_rows_not_flattened():
    table = next(b for b in markdown_to_blocks(SAMPLE) if b.type is BlockType.TABLE)
    assert "Kickoff" in table.text and "$10,000" in table.text
    assert "|" in table.text  # structure preserved


def test_vlm_blocks_carry_provenance_and_no_bbox():
    blocks = markdown_to_blocks(SAMPLE)
    assert all(b.source_engine == "unlimited-ocr" for b in blocks)
    assert all(b.bbox is None for b in blocks)
    assert all(b.block_id.startswith("#/vlm/") for b in blocks)


def test_heading_nesting_sets_parent_id():
    blocks = markdown_to_blocks(SAMPLE)
    title = next(b for b in blocks if b.type is BlockType.TITLE)
    heading = next(b for b in blocks if b.type is BlockType.HEADING)
    assert heading.parent_id == title.block_id


def test_markdown_to_blocks_engine_and_prefix_params():
    blocks = markdown_to_blocks(
        "# Title\n\nBody para.", engine="dots.ocr", id_prefix="#/vlm/p2"
    )
    assert [b.block_id for b in blocks] == ["#/vlm/p2/0", "#/vlm/p2/1"]
    assert all(b.source_engine == "dots.ocr" for b in blocks)
    # parent link uses the prefixed id
    assert blocks[1].parent_id == "#/vlm/p2/0"


def test_markdown_to_blocks_defaults_unchanged():
    blocks = markdown_to_blocks("# Title\n\nBody para.")
    assert [b.block_id for b in blocks] == ["#/vlm/0", "#/vlm/1"]
    assert all(b.source_engine == "unlimited-ocr" for b in blocks)
