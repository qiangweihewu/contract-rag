"""Cross-encoder reranker: a local (no-API) semantic alternative to LLMReranker
behind the same Reranker protocol. Unit-tested with an injected fake model so no
sentence-transformers/torch download is needed."""

from contract_rag.chunk.models import Chunk
from contract_rag.index.rerank import CrossEncoderReranker
from contract_rag.obs.counters import InMemoryCounterStore


def _c(cid, text):
    return Chunk(chunk_id=cid, doc_id="d", text=text, block_ids=[cid])


class _FakeCE:
    """Mimics sentence-transformers CrossEncoder.predict: one score per (query, passage)."""

    def predict(self, pairs):
        return [1.0 if "termination" in passage.lower() else 0.0 for _query, passage in pairs]


def test_cross_encoder_reranker_orders_by_model_score():
    chunks = [_c("a", "unrelated boilerplate text"), _c("b", "termination notice period clause")]
    out = CrossEncoderReranker(model=_FakeCE()).rerank("when can we terminate", chunks)
    assert out[0].chunk_id == "b"
    assert {c.chunk_id for c in out} == {"a", "b"}     # no chunk dropped


def test_cross_encoder_reranker_degrades_on_model_error():
    class _Boom:
        def predict(self, pairs):
            raise RuntimeError("model unavailable")

    chunks = [_c("a", "first"), _c("b", "second")]
    out = CrossEncoderReranker(model=_Boom()).rerank("q", chunks)
    assert [c.chunk_id for c in out] == ["a", "b"]      # original order preserved


def test_cross_encoder_reranker_records_degrade_metric_on_error():
    # A silent fallback must be observable — otherwise a dead reranker degrades
    # retrieval quality with no on-call signal (the audit's A-section finding).
    class _Boom:
        def predict(self, pairs):
            raise RuntimeError("model unavailable")

    counter = InMemoryCounterStore()
    chunks = [_c("a", "first"), _c("b", "second")]
    CrossEncoderReranker(model=_Boom(), counter=counter).rerank("q", chunks)
    assert counter.value("rerank.degraded") == 1


def test_cross_encoder_reranker_no_metric_recorded_on_success():
    counter = InMemoryCounterStore()
    chunks = [_c("a", "unrelated"), _c("b", "termination notice period clause")]
    CrossEncoderReranker(model=_FakeCE(), counter=counter).rerank("terminate", chunks)
    assert counter.value("rerank.degraded") == 0


def test_cross_encoder_reranker_works_without_counter():
    class _Boom:
        def predict(self, pairs):
            raise RuntimeError("model unavailable")

    chunks = [_c("a", "first"), _c("b", "second")]
    out = CrossEncoderReranker(model=_Boom()).rerank("q", chunks)  # no counter injected
    assert [c.chunk_id for c in out] == ["a", "b"]


def test_cross_encoder_reranker_single_chunk_is_noop():
    chunks = [_c("a", "only one")]
    assert CrossEncoderReranker(model=_FakeCE()).rerank("q", chunks) == chunks


def test_cross_encoder_reranker_satisfies_protocol_name():
    assert CrossEncoderReranker(model=_FakeCE()).name == "cross_encoder"
