"""Embedders behind one protocol: a deterministic, dependency-free HashingEmbedder
(default / CI / tests) and a gated OpenAIEmbedder (real semantic vectors).

Mirrors the extractor pattern: credential-free floor + gated upgrade. Select via
`get_embedder(settings, kind)`."""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from contract_rag.config import Settings
from contract_rag.text import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _bucket(word: str, dim: int) -> int:
    # hashlib (not builtin hash) so vectors are stable across processes/runs
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % dim


class HashingEmbedder:
    """Deterministic hashed bag-of-tokens vector. Not semantic, but real, fast, and
    free — proves the dense/hybrid plumbing without an API or model download."""

    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in tokenize(t):
                v[_bucket(w, self.dim)] += 1.0
            out.append(_normalize(v))
        return out


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, settings: Settings, model: str = "text-embedding-3-small"):
        if not settings.allow_external_llm:
            raise PermissionError(
                "openai embedder requires ALLOW_EXTERNAL_LLM=true; refusing to send "
                "documents to a third party by default."
            )
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [_normalize(d.embedding) for d in resp.data]


def get_embedder(settings: Settings, kind: str = "hashing") -> Embedder:
    if kind == "openai":
        return OpenAIEmbedder(settings)
    if kind == "hashing":
        return HashingEmbedder()
    raise NotImplementedError(f"embedder {kind!r} not available")
