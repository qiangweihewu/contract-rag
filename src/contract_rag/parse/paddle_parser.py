from __future__ import annotations

from pathlib import Path

from contract_rag.ingest.store import file_hash
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

_RENDER_DPI = 300


def lines_to_blocks(lines: list[dict]) -> list[DocBlock]:
    blocks: list[DocBlock] = []
    for i, ln in enumerate(lines):
        x0, y0, x1, y1 = ln["box"]
        blocks.append(
            DocBlock(
                block_id=f"#/ocr/{i}",
                type=BlockType.PARAGRAPH,
                text=ln["text"],
                bbox=BoundingBox(page=ln["page"], x0=x0, y0=y0, x1=x1, y1=y1),
                confidence=ln["conf"],
                source_engine="paddleocr",
            )
        )
    return blocks


def predict_result_to_lines(res, page_no: int) -> list[dict]:
    """paddleocr >= 3 `predict()` result (dict-like with parallel `rec_texts` /
    `rec_scores` / `rec_boxes` arrays) -> the neutral line dicts `lines_to_blocks`
    consumes. Pure: takes any mapping, so unit tests pass a plain dict."""
    lines: list[dict] = []
    for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
        x0, y0, x1, y1 = (float(v) for v in box)
        lines.append(
            {"text": str(text), "box": (x0, y0, x1, y1), "conf": float(score), "page": page_no}
        )
    return lines


def legacy_result_to_lines(result_page, page_no: int) -> list[dict]:
    """paddleocr 2.x `ocr()` page result ([[quad, (text, conf)], ...]) -> line dicts."""
    lines: list[dict] = []
    for line in result_page or []:
        quad, (text, conf) = line
        xs = [pt[0] for pt in quad]
        ys = [pt[1] for pt in quad]
        lines.append(
            {
                "text": text,
                "box": (min(xs), min(ys), max(xs), max(ys)),
                "conf": float(conf),
                "page": page_no,
            }
        )
    return lines


_OCR_SINGLETON: tuple | None = None  # (engine, api) — model load is seconds, reuse it


def _select_api(version: str) -> str:
    """Pick the OCR call surface from the installed `paddleocr.__version__` string:
    'predict' (PP-OCRv5+ pipelines) for >=3, the legacy 'ocr' API for 2.x. Pure and
    testable without the dependency installed — replaces the old
    `except (ValueError, TypeError)` construction-failure guess, which could
    misattribute an unrelated model/network error to "must be 2.x"."""
    try:
        major = int(version.strip().split(".")[0])
    except (ValueError, IndexError):
        raise ValueError(f"unparseable paddleocr version {version!r}") from None
    return "predict" if major >= 3 else "ocr"


def _get_ocr() -> tuple:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        import paddleocr

        api = _select_api(paddleocr.__version__)
        try:
            if api == "predict":  # paddleocr >= 3; 2.x rejects these kwargs
                engine = paddleocr.PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            else:
                engine = paddleocr.PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception as exc:
            raise RuntimeError(
                f"failed to construct PaddleOCR (detected api={api!r} for "
                f"paddleocr {paddleocr.__version__}): {exc}"
            ) from exc
        _OCR_SINGLETON = (engine, api)
    return _OCR_SINGLETON


def _run_paddle(path: Path) -> list[dict]:
    import tempfile

    import pypdfium2 as pdfium

    ocr, api = _get_ocr()
    lines: list[dict] = []

    pdf = pdfium.PdfDocument(str(path))
    try:
        with tempfile.TemporaryDirectory() as d:
            for page_no, page in enumerate(pdf, start=1):
                try:
                    pil = page.render(scale=_RENDER_DPI / 72).to_pil()
                finally:
                    page.close()
                img = Path(d) / f"p{page_no}.png"
                pil.save(img)
                if api == "predict":
                    for res in ocr.predict(str(img)):
                        lines.extend(predict_result_to_lines(res, page_no))
                else:
                    result = ocr.ocr(str(img), cls=True)
                    lines.extend(legacy_result_to_lines(result[0], page_no))
    finally:
        pdf.close()
    return lines


def parse_with_paddle(path: Path) -> DocumentIR:
    path = Path(path)
    lines = _run_paddle(path)
    h = file_hash(path)
    return DocumentIR(
        doc_id=h,
        source_uri=path.resolve().as_uri(),
        file_hash=h,
        mime_type="application/pdf",
        blocks=lines_to_blocks(lines),
        metadata={},
    )
