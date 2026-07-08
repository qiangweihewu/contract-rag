from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path

from contract_rag.config import Settings
from contract_rag.ingest.store import file_hash
from contract_rag.ir import DocumentIR
from contract_rag.parse.markdown_ir import markdown_to_blocks

_PROMPT = "Multi page parsing."
_MAX_ATTEMPTS = 2       # 1 retry on top of the first attempt
_BACKOFF_SECONDS = 1.0


def build_vlm_ir(doc_id: str, source_uri: str, file_hash_str: str, markdown: str) -> DocumentIR:
    return DocumentIR(
        doc_id=doc_id,
        source_uri=source_uri,
        file_hash=file_hash_str,
        mime_type="application/pdf",
        blocks=markdown_to_blocks(markdown),
        metadata={},
    )


def _pdf_to_image_b64(path: Path, dpi: int = 300) -> list[str]:
    import pypdfium2 as pdfium

    out: list[str] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        with tempfile.TemporaryDirectory() as d:
            for idx, page in enumerate(pdf):
                try:
                    pil = page.render(scale=dpi / 72).to_pil()
                finally:
                    page.close()
                img = Path(d) / f"page_{idx:04d}.png"
                pil.save(img)
                out.append(base64.b64encode(img.read_bytes()).decode("utf-8"))
    finally:
        pdf.close()
    return out


def _post_vlm_request(settings: Settings, content, post_fn, sleep_fn=time.sleep):
    """POST to the VLM endpoint with a bounded retry (network blips / transient
    5xx must not sink the whole parse on the first hiccup) and the configurable
    `settings.vlm_timeout` (was a hardcoded 1200s)."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = post_fn(
                f"{settings.vlm_endpoint}/chat/completions",
                json={
                    "model": "Unlimited-OCR",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                },
                timeout=settings.vlm_timeout,
            )
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 — deliberately broad: any transport failure retries
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                sleep_fn(_BACKOFF_SECONDS * (attempt + 1))
    raise ConnectionError(
        f"VLM endpoint {settings.vlm_endpoint!r} failed after {_MAX_ATTEMPTS} attempts"
    ) from last_exc


def _extract_markdown(resp) -> str:
    """Defensive pull of the OCR markdown out of the chat-completion response —
    raises a clear ValueError instead of an opaque KeyError/IndexError when the
    endpoint returns an unexpected shape."""
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"unexpected VLM response shape: {exc}") from exc


def parse_with_vlm(path: Path, settings: Settings, post_fn=None) -> DocumentIR:
    if not settings.vlm_endpoint:
        raise ValueError("parse_with_vlm requires settings.vlm_endpoint (VLM_ENDPOINT)")
    if post_fn is None:
        import requests

        post_fn = requests.post

    path = Path(path)
    images = _pdf_to_image_b64(path)
    content = [{"type": "text", "text": _PROMPT}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        for b64 in images
    ]
    resp = _post_vlm_request(settings, content, post_fn)
    markdown = _extract_markdown(resp)
    h = file_hash(path)
    return build_vlm_ir(
        doc_id=h,
        source_uri=path.resolve().as_uri(),
        file_hash_str=h,
        markdown=markdown,
    )
