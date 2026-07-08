from contract_rag.ir import BlockType
from contract_rag.parse.docling_parser import (
    build_ir_from_items,
    docling_label_to_block_type,
)


def test_label_mapping_covers_known_labels():
    assert docling_label_to_block_type("section_header") is BlockType.HEADING
    assert docling_label_to_block_type("paragraph") is BlockType.PARAGRAPH
    assert docling_label_to_block_type("table") is BlockType.TABLE
    assert docling_label_to_block_type("totally_unknown") is BlockType.PARAGRAPH


def test_build_ir_from_items_produces_blocks_with_provenance():
    items = [
        {"label": "title", "text": "Master Services Agreement", "page": 1,
         "bbox": (10, 20, 300, 40), "self_ref": "#/texts/0", "parent_ref": None},
        {"label": "paragraph", "text": "This Agreement is entered into by Acme Inc.",
         "page": 1, "bbox": (10, 50, 300, 70), "self_ref": "#/texts/1", "parent_ref": "#/texts/0"},
    ]
    ir = build_ir_from_items(doc_id="d1", source_uri="file:///x.pdf", file_hash_str="abc", items=items)
    assert ir.doc_id == "d1"
    assert ir.file_hash == "abc"
    assert len(ir.blocks) == 2
    assert ir.blocks[0].type is BlockType.TITLE
    assert ir.blocks[1].parent_id == "#/texts/0"
    assert all(b.source_engine == "docling" for b in ir.blocks)
    assert all(0.0 <= b.confidence <= 1.0 for b in ir.blocks)
