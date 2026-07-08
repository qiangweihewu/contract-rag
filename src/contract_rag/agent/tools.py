"""Agent tools — each composes an existing contract-rag layer behind a uniform
`name` + `run(inp: dict) -> dict` interface so the planner/runner stay generic
while each tool validates its own typed Pydantic input/output internally."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical

from pydantic import BaseModel

from contract_rag.chunk.models import Chunk
from contract_rag.ir import DocumentIR
from contract_rag.security.search import search_as


class Tool(Protocol):
    name: str
    def run(self, inp: dict) -> dict: ...


class RetrieveInput(BaseModel):
    query: str
    k: int = 5


class RetrieveOutput(BaseModel):
    chunks: list[Chunk]


class RetrieveTool:
    name = "retrieve"

    def __init__(self, index, principal=None) -> None:
        self.index = index
        self.principal = principal

    def run(self, inp: dict) -> dict:
        args = RetrieveInput(**inp)
        if self.principal is not None:
            chunks = search_as(self.index, args.query, self.principal, k=args.k)
        else:
            chunks = self.index.search(args.query, k=args.k)
        return RetrieveOutput(chunks=list(chunks)).model_dump()


class ExtractFieldInput(BaseModel):
    field: str


class ExtractFieldOutput(BaseModel):
    field: str
    value: str
    source_block_id: str | None
    confidence: float


class ExtractFieldTool:
    name = "extract_field"

    def __init__(self, extractor, ir: DocumentIR) -> None:
        self.extractor = extractor
        self.ir = ir

    def run(self, inp: dict) -> dict:
        args = ExtractFieldInput(**inp)
        facts = self.extractor.extract(self.ir)
        clause = getattr(facts, args.field)
        return ExtractFieldOutput(
            field=args.field, value=clause.value,
            source_block_id=clause.source_block_id, confidence=clause.confidence,
        ).model_dump()


class CheckClauseInput(BaseModel):
    clause_type: str


class CheckClauseOutput(BaseModel):
    clause_type: str
    present: bool
    evidence_block_ids: list[str]


class CheckClauseTool:
    name = "check_clause"

    def __init__(self, ir: DocumentIR, vertical: Vertical | None = None) -> None:
        from contract_rag.verticals.registry import default_vertical
        self.ir = ir
        self.vertical = vertical or default_vertical()

    def run(self, inp: dict) -> dict:
        args = CheckClauseInput(**inp)
        hits: list[str] = []
        for b in self.ir.blocks:
            probe = Chunk(chunk_id=b.block_id, doc_id=self.ir.doc_id, text=b.text,
                          block_ids=[b.block_id])
            if self.vertical.classify_clause(probe) == args.clause_type:
                hits.append(b.block_id)
        return CheckClauseOutput(
            clause_type=args.clause_type, present=bool(hits), evidence_block_ids=hits
        ).model_dump()


class CiteInput(BaseModel):
    block_id: str


class CiteOutput(BaseModel):
    block_id: str
    text: str


class CiteTool:
    name = "cite"

    def __init__(self, ir: DocumentIR) -> None:
        self.ir = ir

    def run(self, inp: dict) -> dict:
        args = CiteInput(**inp)
        text = next((b.text for b in self.ir.blocks if b.block_id == args.block_id), "")
        return CiteOutput(block_id=args.block_id, text=text).model_dump()
