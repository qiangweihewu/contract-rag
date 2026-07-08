"""Dense (embedding) retriever — delegates storage to a VectorStore (in-memory default,
or the pgvector adapter), so the backend is swappable without touching retrieval."""
from __future__ import annotations

from contract_rag.chunk.models import Chunk
from contract_rag.index.embed import Embedder
from contract_rag.index.store import InMemoryVectorStore, VectorStore


class DenseIndex:
    def __init__(self, embedder: Embedder, store: VectorStore | None = None):
        self.embedder = embedder
        self.store: VectorStore = store or InMemoryVectorStore()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.store.add(chunks, self.embedder.embed([c.index_text() for c in chunks]))

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        return self.store.query(self.embedder.embed([query])[0], k)
