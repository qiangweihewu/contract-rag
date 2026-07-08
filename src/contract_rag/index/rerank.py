"""Final reranking stage after RRF fusion. `Reranker` protocol with a dependency-free
LexicalReranker default and a gated LLMReranker (the same credential-free-floor /
gated-upgrade pattern as embedders). A cross-encoder reranker fits the same protocol."""
from __future__ import annotations

from typing import Protocol

from contract_rag.chunk.models import Chunk
from contract_rag.config import Settings
from contract_rag.obs.counters import CounterStore
from contract_rag.text import tokenize

_DEGRADED_METRIC = "rerank.degraded"  # incremented whenever a reranker silently falls back


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...


class LexicalReranker:
    """Reorder by query-term overlap density — free, deterministic baseline."""

    name = "lexical"

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        q = set(tokenize(query))
        if not q:
            return list(chunks)

        def score(c: Chunk) -> float:
            toks = tokenize(c.index_text())
            return sum(t in q for t in toks) / (len(toks) or 1)

        return sorted(chunks, key=score, reverse=True)


class CrossEncoderReranker:
    """Semantic reranking via a sentence-transformers cross-encoder — a local, no-API
    alternative to LLMReranker behind the same protocol. A cross-encoder scores each
    (query, passage) pair jointly, so it is more accurate than the bi-encoder used for
    retrieval but too slow to run over the whole corpus — ideal as a final reorder of
    the fused candidate pool. `model` is an injectable seam so unit tests skip the
    sentence-transformers/torch download."""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model=None,
        counter: CounterStore | None = None,
    ):
        if model is not None:
            self._model = model
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        self._counter = counter

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) <= 1:
            return list(chunks)
        pairs = [(query, c.index_text()) for c in chunks]
        try:
            scores = self._model.predict(pairs)
        except Exception:
            if self._counter is not None:
                self._counter.incr(_DEGRADED_METRIC)
            return list(chunks)  # degrade gracefully — a rerank failure must not break retrieval
        return [c for _, c in sorted(zip(scores, chunks), key=lambda sc: -float(sc[0]))]


_PROMPT = (
    "Rank the passages by how well they answer the query. Return passage numbers "
    "(0-indexed) best first, comma-separated. Query: {query}\n\n{passages}"
)


class LLMReranker:
    """Relevance reranking via an LLM (gated). A cross-encoder is the heavier alternative
    behind this same Reranker protocol."""

    name = "llm"

    def __init__(
        self, settings: Settings, model: str = "gpt-5-mini", counter: CounterStore | None = None
    ):
        if not settings.allow_external_llm:
            raise PermissionError("LLM reranker requires ALLOW_EXTERNAL_LLM=true")
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self._counter = counter

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) <= 1:
            return list(chunks)
        # NOTE: uses c.text only — unlike Lexical/CrossEncoder this reranker does NOT
        # see index_extra (e.g. injected definitions), by design (keeps the gated LLM
        # call scoped to display text; revisit if that gap matters in practice).
        passages = "\n".join(f"[{i}] {c.text[:400]}" for i, c in enumerate(chunks))
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _PROMPT.format(query=query, passages=passages)}],
            )
        except Exception:
            if self._counter is not None:
                self._counter.incr(_DEGRADED_METRIC)
            return list(chunks)  # degrade gracefully — a rerank failure must not break retrieval
        order: list[int] = []
        for tok in resp.choices[0].message.content.replace(" ", "").split(","):
            if tok.isdigit() and int(tok) < len(chunks) and int(tok) not in order:
                order.append(int(tok))
        order += [i for i in range(len(chunks)) if i not in order]  # append any missed
        return [chunks[i] for i in order]
