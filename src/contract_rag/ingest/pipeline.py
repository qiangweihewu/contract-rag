"""Ingest front door: parse a document, then (by default) redact PII before the
IR flows downstream. Redaction-at-ingest is the spec's Day-1 compliance must;
the raw parse path (eval/baseline) stays unredacted by calling parse directly."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from contract_rag.config import Settings
from contract_rag.ir import DocumentIR
from contract_rag.parse.router import parse
from contract_rag.security.pii import PIIMatch
from contract_rag.security.redact import redact_ir


class IngestResult(BaseModel):
    ir: DocumentIR
    redactions: list[PIIMatch]


def ingest_document(
    path: Path,
    settings: Settings,
    *,
    parse_fn: Callable[[Path, Settings], DocumentIR] = parse,
    redact: bool | None = None,
) -> IngestResult:
    do_redact = settings.redact_pii if redact is None else redact
    ir = parse_fn(path, settings)
    if not do_redact:
        return IngestResult(ir=ir, redactions=[])
    result = redact_ir(ir)
    return IngestResult(ir=result.ir, redactions=result.matches)
