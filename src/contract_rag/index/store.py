"""Pluggable vector store behind DenseIndex: in-memory default, pgvector adapter optional.
Same add()/query() surface so they're interchangeable."""
from __future__ import annotations

import math
from typing import Protocol

from contract_rag.chunk.models import Chunk


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def query(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def query(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        scored = [(c, cosine(vector, v)) for c, v in zip(self.chunks, self.vectors)]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
