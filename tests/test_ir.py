import pytest
from pydantic import ValidationError

from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(block_id: str = "b1", text: str = "hello") -> DocBlock:
    return DocBlock(
        block_id=block_id,
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
        confidence=0.9,
        source_engine="docling",
    )


def test_documentir_round_trips_through_json():
    ir = DocumentIR(
        doc_id="d1",
        source_uri="file:///tmp/a.pdf",
        file_hash="abc",
        mime_type="application/pdf",
        blocks=[_block()],
        metadata={"counterparty": "Acme"},
    )
    restored = DocumentIR.model_validate_json(ir.model_dump_json())
    assert restored == ir
    assert restored.blocks[0].type is BlockType.PARAGRAPH


def test_confidence_must_be_within_unit_interval():
    with pytest.raises(ValidationError):
        DocBlock(
            block_id="b",
            type=BlockType.PARAGRAPH,
            text="x",
            bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
            confidence=1.5,
            source_engine="docling",
        )


def test_docblock_bbox_is_optional_and_round_trips_as_none():
    b = DocBlock(
        block_id="#/vlm/0",
        type=BlockType.PARAGRAPH,
        text="from a VLM, no coordinates",
        confidence=1.0,
        source_engine="unlimited-ocr",
    )
    assert b.bbox is None
    restored = DocBlock.model_validate_json(b.model_dump_json())
    assert restored.bbox is None
    assert restored == b
