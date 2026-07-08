"""Hybrid retrieval: Reciprocal Rank Fusion of BM25 (lexical) + dense (semantic),
with clause-level ABAC filtering. RRF needs no score calibration between retrievers —
it fuses ranks, so a lexical hit and a semantic hit reinforce each other."""
from __future__ import annotations

from contract_rag.chunk.models import Chunk
from contract_rag.index.bm25 import BM25Index
from contract_rag.index.dense import DenseIndex
from contract_rag.index.embed import Embedder, HashingEmbedder

_CANDIDATES = 20


class HybridIndex:
    def __init__(self, bm25: BM25Index, dense: DenseIndex):
        self.bm25 = bm25
        self.dense = dense

    def add(self, chunks: list[Chunk]) -> None:
        self.bm25.add(chunks)
        self.dense.add(chunks)

    def search(
        self, query: str, k: int = 5, rrf_k: int = 60,
        allowed_tags: list[str] | None = None, reranker=None,
    ) -> list[Chunk]:
        ranked_lists = [
            [c for c, _ in self.bm25.search(query, k=_CANDIDATES)],
            [c for c, _ in self.dense.search(query, k=_CANDIDATES)],
        ]
        scores: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, c in enumerate(ranked):
                by_id[c.chunk_id] = c
                scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        fused = sorted(by_id.values(), key=lambda c: -scores[c.chunk_id])
        if allowed_tags is not None:
            allowed = set(allowed_tags)
            fused = [c for c in fused if set(c.permission_tags) & allowed]
        if reranker is not None:                       # rerank the candidate pool, then cut to k
            fused = reranker.rerank(query, fused[:_CANDIDATES])
        return fused[:k]


def build_index(chunks: list[Chunk], embedder: Embedder | None = None) -> HybridIndex:
    idx = HybridIndex(BM25Index(), DenseIndex(embedder or HashingEmbedder()))
    idx.add(chunks)
    return idx
