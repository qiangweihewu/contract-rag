"""'Ask a question over your contract' — the retrieval half of the demo.

Cleans → chunks → enriches → hybrid-indexes a single contract's IR, then returns
the top-k clauses for a query. Each result carries its source `block_ids`, so the
answer stays *sourced* (the project's whole thesis). Pure + unit-tested; the
Streamlit app is a thin shell over `answer_question`."""
from __future__ import annotations

from pydantic import BaseModel, Field

from contract_rag.chunk.chunker import chunk_ir
from contract_rag.clean.pipeline import clean_ir
from contract_rag.enrich import definitions as _definitions
from contract_rag.enrich.enricher import enrich_chunks
from contract_rag.index.embed import Embedder
from contract_rag.index.hybrid import build_index
from contract_rag.ir import DocumentIR


class RetrievedClause(BaseModel):
    rank: int
    text: str
    heading: str | None = None
    clause_type: str | None = None        # rule-based, set by enrich/
    block_ids: list[str]                  # provenance — which source blocks back this answer
    definition_block_ids: list[str] = Field(default_factory=list)  # provenance of injected defs


def answer_question(
    ir: DocumentIR,
    query: str,
    *,
    embedder: Embedder | None = None,
    k: int = 5,
    reranker=None,
    allowed_tags: list[str] | None = None,
    clean: bool = True,
    inject_definitions: bool = False,
) -> list[RetrievedClause]:
    """Top-k clauses relevant to `query`, retrieved over the (cleaned) contract.

    Hybrid BM25 + dense fusion via `build_index`; the same ABAC `allowed_tags` and
    `reranker` seams as the eval. Empty query or empty document → no results. The
    default `HashingEmbedder` keeps this credential-free; pass an `OpenAIEmbedder`
    for semantic recall.

    `inject_definitions` (default off, additive): when True, extracts defined terms
    from the doc and injects their definitions into any chunk that USES the term
    (retrieval-only text — `text`/`block_ids` are never touched), between enrichment
    and indexing. Off by default so results are byte-identical to before."""
    if not query.strip():
        return []
    doc = clean_ir(ir) if clean else ir
    chunks = enrich_chunks(chunk_ir(doc))
    if not chunks:
        return []
    if inject_definitions:
        chunks = _definitions.inject_definitions(chunks, _definitions.extract_definitions(doc))
    hits = build_index(chunks, embedder).search(
        query, k=k, allowed_tags=allowed_tags, reranker=reranker
    )
    return [
        RetrievedClause(rank=i + 1, text=c.text, heading=c.heading,
                        clause_type=c.clause_type, block_ids=list(c.block_ids),
                        definition_block_ids=list(c.definition_block_ids))
        for i, c in enumerate(hits)
    ]
