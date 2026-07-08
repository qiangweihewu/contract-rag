"""FastAPI service over the contract-rag pipeline + the F2 free-diagnosis hook.

Thin shell: every endpoint ingests an upload (parse → redact) then calls a pure
service function. Collaborators are injected via create_app(...) so tests use a
fake parse + FakeExtractor and never run docling or hit a network. Each request
opens an obs Tracer span tree, so GET /v1/metrics is a live SLO dashboard."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from contract_rag.api.service import diagnose_ir
from contract_rag.api.slo import check_slo
from contract_rag.clean.pipeline import clean_ir
from contract_rag.config import Settings, assert_backend_allowed, get_settings
from contract_rag.demo.ask import answer_question
from contract_rag.ingest.pipeline import ingest_document
from contract_rag.ir import DocumentIR
from contract_rag.obs.metrics import aggregate_traces
from contract_rag.obs.store import InMemoryTraceStore
from contract_rag.obs.tracer import Tracer

if TYPE_CHECKING:
    from contract_rag.agent.models import AgentResult, AgentTask

_UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB, streamed so a huge upload is rejected mid-read


def _default_extractor(settings: Settings):
    """Credential-free default: rule extractor unless a real backend is configured
    (mirrors demo/report.py). get_extractor enforces the external-LLM gate."""
    from contract_rag.extract.rules import RuleExtractor

    if settings.extract_backend in ("fake", "rule"):
        return RuleExtractor()
    from contract_rag.extract.extractor import get_extractor

    return get_extractor(settings)


def create_app(
    *,
    settings: Settings | None = None,
    parse_fn: Callable[[Path, Settings], DocumentIR] | None = None,
    extractor=None,
    embedder=None,
    agent: Callable[["DocumentIR", "AgentTask"], "AgentResult"] | None = None,
    tracer: Tracer | None = None,
    trace_store: InMemoryTraceStore | None = None,
):
    settings = settings or get_settings()
    store = trace_store or InMemoryTraceStore()
    tracer = tracer or Tracer(store=store)

    def _extractor():
        # Lazy: an injected extractor is used as-is; otherwise the default is built
        # on first /extract use, so a gated-backend misconfig (openai without the
        # ALLOW_EXTERNAL_LLM gate) can't crash startup — /ready reports it as 503.
        nonlocal extractor
        if extractor is None:
            extractor = _default_extractor(settings)
        return extractor

    def _real_parse(path: Path, s: Settings) -> DocumentIR:
        from contract_rag.parse.router import parse

        return parse(path, s)

    parse_fn = parse_fn or _real_parse

    def _default_agent(ir: DocumentIR, task: "AgentTask") -> "AgentResult":
        """Credential-free real agent: clean->chunk->enrich->hybrid-index over the IR,
        RulePlanner + the configured extractor + hashing embedder, run via run_agent."""
        from contract_rag.agent.planner import RulePlanner
        from contract_rag.agent.runner import build_agent_tools, run_agent
        from contract_rag.chunk.chunker import chunk_ir
        from contract_rag.enrich.enricher import enrich_chunks
        from contract_rag.index.hybrid import build_index

        chunks = enrich_chunks(chunk_ir(ir))
        index = build_index(chunks, embedder)
        tools = build_agent_tools(ir, index, _extractor())
        return run_agent(task, RulePlanner(), tools, tracer=tracer)

    agent_fn = agent or _default_agent

    app = FastAPI(title="contract-rag", version="0.0.0")
    app.state.trace_store = store

    def _save_upload(file: UploadFile) -> Path:
        """Stream the upload to a temp file in bounded chunks so an oversized file is
        rejected (413) mid-read instead of being fully buffered in memory first."""
        max_bytes = settings.max_upload_mb * 1024 * 1024
        suffix = Path(file.filename or "upload").suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = Path(tmp.name)
        try:
            total = 0
            while chunk := file.file.read(_UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds the {settings.max_upload_mb}MB limit",
                    )
                tmp.write(chunk)
        except Exception:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
        tmp.close()
        return tmp_path

    def _ingest(file: UploadFile, trace) -> tuple[DocumentIR, int]:
        path = _save_upload(file)
        try:
            with tracer.span(trace, "parse"):
                try:
                    res = ingest_document(path, settings, parse_fn=parse_fn)
                except Exception as exc:
                    raise HTTPException(
                        status_code=400, detail=f"failed to parse document: {exc}"
                    ) from exc
            return res.ir, len(res.redactions)
        finally:
            path.unlink(missing_ok=True)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        try:
            assert_backend_allowed(settings)
        except PermissionError as exc:
            return JSONResponse({"ready": False, "reason": str(exc)}, status_code=503)
        return {"ready": True, "backend": settings.extract_backend}

    @app.post("/v1/diagnose")
    def diagnose(file: UploadFile = File(...)):
        trace = tracer.start(doc_id=file.filename or "upload")
        try:
            ir, redactions = _ingest(file, trace)
            with tracer.span(trace, "diagnose"):
                diag = diagnose_ir(ir, doc_id=file.filename, redactions=redactions)
            return diag.model_dump()
        finally:
            tracer.finish(trace)

    @app.post("/v1/extract")
    def extract(file: UploadFile = File(...)):
        from contract_rag.extract.verify import verify

        trace = tracer.start(doc_id=file.filename or "upload")
        try:
            ir, _ = _ingest(file, trace)
            with tracer.span(trace, "clean"):
                cleaned = clean_ir(ir)
            with tracer.span(trace, "extract"):
                facts = _extractor().extract(cleaned)
            report = verify(facts, cleaned)
            return {
                "facts": facts.model_dump(),
                "verification": {k: c.model_dump() for k, c in report.checks.items()},
            }
        finally:
            tracer.finish(trace)

    @app.post("/v1/ask")
    def ask(file: UploadFile = File(...), q: str = Form(...), k: int = Form(5)):
        trace = tracer.start(doc_id=file.filename or "upload")
        try:
            ir, _ = _ingest(file, trace)
            with tracer.span(trace, "retrieve"):
                hits = answer_question(ir, q, embedder=embedder, k=k)
            return {"query": q, "results": [h.model_dump() for h in hits]}
        finally:
            tracer.finish(trace)

    @app.post("/v1/agent")
    def agent_route(
        file: UploadFile = File(...), q: str = Form(...), field: str | None = Form(None)
    ):
        from contract_rag.agent.models import AgentTask

        trace = tracer.start(doc_id=file.filename or "upload")
        try:
            ir, _ = _ingest(file, trace)
            with tracer.span(trace, "clean"):
                cleaned = clean_ir(ir)
            task = AgentTask(question=q, field=field, doc_id=file.filename or "doc")
            result = agent_fn(cleaned, task)
            ans = result.state.answer
            return {
                "answer": ans.value if ans else "",
                "confidence": ans.confidence if ans else 0.0,
                "citations": [
                    {"block_id": c.block_id, "text": c.text}
                    for c in (ans.citations if ans else [])
                ],
                "status": result.state.status.value,
                "trace_id": result.trace_id,
            }
        finally:
            tracer.finish(trace)

    @app.get("/v1/metrics")
    def metrics() -> dict:
        agg = aggregate_traces(store.all())
        return {"metrics": agg, "slo": check_slo(agg).model_dump()}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    return app


_INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>contract-rag — free data-quality diagnosis</title>
<style>body{font-family:Georgia,serif;max-width:42rem;margin:4rem auto;padding:0 1.2rem;
color:#222}h1{font-weight:600}button{font:inherit;padding:.5rem 1rem;cursor:pointer}
small{color:#666}</style></head><body>
<h1>Is your contract RAG returning garbage?</h1>
<p>Upload a contract (PDF/DOCX). We score its data quality, clean it, and list
exactly what's wrong — before/after, with the recoverable lift.</p>
<form action="/v1/diagnose" method="post" enctype="multipart/form-data">
<input type="file" name="file" required>
<button type="submit">Diagnose</button></form>
<p><small>PII is redacted at ingest. The credential-free pipeline runs with no
keys. See <code>/v1/metrics</code> for live SLO.</small></p>
</body></html>"""


# Module-level app for `uvicorn contract_rag.api.app:app` (credential-free defaults).
app = create_app()
