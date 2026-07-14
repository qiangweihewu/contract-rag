from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path

from contract_rag.config import Settings
from contract_rag.ingest.store import file_hash
from contract_rag.ir import DocumentIR
from contract_rag.parse.markdown_ir import markdown_to_blocks

_MAX_ATTEMPTS = 2       # 1 retry on top of the first attempt
_BACKOFF_SECONDS = 1.0


def build_vlm_ir(
    doc_id: str,
    source_uri: str,
    file_hash_str: str,
    markdown: str = "",
    *,
    blocks: list | None = None,
) -> DocumentIR:
    return DocumentIR(
        doc_id=doc_id,
        source_uri=source_uri,
        file_hash=file_hash_str,
        mime_type="application/pdf",
        blocks=blocks if blocks is not None else markdown_to_blocks(markdown),
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
                    "model": settings.vlm_model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    **(
                        {"max_tokens": settings.vlm_max_tokens}
                        if settings.vlm_max_tokens is not None
                        else {}
                    ),
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


def _page_content(settings: Settings, b64: str) -> list[dict]:
    return [
        {"type": "text", "text": settings.vlm_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]


def parse_with_vlm(path: Path, settings: Settings, post_fn=None, render_fn=None) -> DocumentIR:
    if not settings.vlm_endpoint:
        raise ValueError("parse_with_vlm requires settings.vlm_endpoint (VLM_ENDPOINT)")
    if post_fn is None:
        import requests

        post_fn = requests.post
    if render_fn is None:
        render_fn = _pdf_to_image_b64

    path = Path(path)
    blocks: list = []
    for page_no, b64 in enumerate(render_fn(path), start=1):
        resp = _post_vlm_request(settings, _page_content(settings, b64), post_fn)
        markdown = _extract_markdown(resp)
        if settings.vlm_raw_dir is not None:
            raw_page = settings.vlm_raw_dir / path.stem / f"page_{page_no:04d}.md"
            raw_page.parent.mkdir(parents=True, exist_ok=True)
            raw_page.write_text(markdown)
        blocks.extend(
            markdown_to_blocks(
                markdown, engine=settings.vlm_model, id_prefix=f"#/vlm/p{page_no}"
            )
        )
    h = file_hash(path)
    return build_vlm_ir(
        doc_id=h, source_uri=path.resolve().as_uri(), file_hash_str=h, blocks=blocks
    )
