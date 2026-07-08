from __future__ import annotations

from pathlib import Path

from contract_rag.ingest.store import file_hash
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

_LABEL_MAP: dict[str, BlockType] = {
    "title": BlockType.TITLE,
    "section_header": BlockType.HEADING,
    "heading": BlockType.HEADING,
    "paragraph": BlockType.PARAGRAPH,
    "text": BlockType.PARAGRAPH,
    "table": BlockType.TABLE,
    "list_item": BlockType.LIST_ITEM,
    "page_footer": BlockType.FOOTER,
    "page_header": BlockType.HEADER,
    "caption": BlockType.FIGURE_CAPTION,
}


def docling_label_to_block_type(label: str) -> BlockType:
    return _LABEL_MAP.get(label.lower(), BlockType.PARAGRAPH)


def build_ir_from_items(doc_id: str, source_uri: str, file_hash_str: str, items: list[dict]) -> DocumentIR:
    blocks: list[DocBlock] = []
    for i, item in enumerate(items):
        x0, y0, x1, y1 = item["bbox"]
        blocks.append(
            DocBlock(
                block_id=item.get("self_ref") or f"#/block/{i}",
                type=docling_label_to_block_type(item["label"]),
                text=item["text"],
                bbox=BoundingBox(page=item["page"], x0=x0, y0=y0, x1=x1, y1=y1),
                parent_id=item.get("parent_ref"),
                confidence=item.get("confidence", 1.0),
                source_engine="docling",
            )
        )
    return DocumentIR(
        doc_id=doc_id, source_uri=source_uri, file_hash=file_hash_str,
        mime_type="application/pdf", blocks=blocks, metadata={},
    )


def _docling_converter():
    """A converter with OCR off: this is the native-digital-PDF branch (high text
    coverage). OCR is the PaddleOCR/VLM branch's job; running it here is redundant
    and pulls in a broken rapidocr engine config."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def parse_with_docling(path: Path) -> DocumentIR:
    path = Path(path)
    doc = _docling_converter().convert(path).document
    items: list[dict] = []
    for el, _level in doc.iterate_items():
        text = getattr(el, "text", "") or ""
        if not text.strip():
            continue
        prov = getattr(el, "prov", None)
        if prov:
            bbox = prov[0].bbox
            page = prov[0].page_no
            box = (bbox.l, bbox.t, bbox.r, bbox.b)
        else:
            page, box = 1, (0.0, 0.0, 0.0, 0.0)
        parent = getattr(el, "parent", None)
        items.append({
            "label": str(getattr(el, "label", "text")),
            "text": text, "page": page, "bbox": box,
            "self_ref": getattr(el, "self_ref", None),
            "parent_ref": getattr(parent, "cref", None) if parent else None,
        })
    h = file_hash(path)
    return build_ir_from_items(
        doc_id=h, source_uri=path.resolve().as_uri(),
        file_hash_str=h, items=items,
    )
