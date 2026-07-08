"""Pure-Python BM25 lexical retriever over chunks (no deps, no infra)."""
from __future__ import annotations

import math

from contract_rag.chunk.models import Chunk
from contract_rag.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.tokens: list[list[str]] = []
        self.df: dict[str, int] = {}
        self.avgdl = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            toks = tokenize(c.index_text())
            self.chunks.append(c)
            self.tokens.append(toks)
            for w in set(toks):
                self.df[w] = self.df.get(w, 0) + 1
        total = sum(len(t) for t in self.tokens)
        self.avgdl = total / len(self.tokens) if self.tokens else 0.0

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.avgdl:
            return []
        n = len(self.tokens)
        q = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk, toks in zip(self.chunks, self.tokens):
            tf: dict[str, int] = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            dl = len(toks)
            score = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df.get(w, 0) + 0.5) / (self.df.get(w, 0) + 0.5))
                score += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
