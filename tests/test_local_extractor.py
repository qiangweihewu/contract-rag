"""Unit tests for the on-device extraction backends (`local` = vLLM/SGLang,
`mlx` = Ollama). Both speak OpenAI-compatible HTTP via instructor and emit the
identical `ContractFacts`. Tests inject a fake instructor client through the DI
seam so they stay dep-free (no openai/instructor import, no network)."""

import importlib.util

import pytest

from contract_rag.config import Settings
from contract_rag.extract.extractor import LocalExtractor, MLXExtractor
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

_openai_missing = importlib.util.find_spec("openai") is None


def _ir() -> DocumentIR:
    return DocumentIR(
        doc_id="d1", source_uri="file:///x.pdf", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH,
                     text="Entered into by Acme Inc.",
                     bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                     confidence=1.0, source_engine="docling")
        ],
        metadata={},
    )


def _canned() -> ContractFacts:
    return ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#/b/1", confidence=0.8),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(),
    )


class _FakeInstructorClient:
    """Mimics instructor's `client.chat.completions.create(...)` surface and
    records the call so tests can assert the model + prompt wiring."""

    def __init__(self, facts: ContractFacts):
        self._facts = facts
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, *, model, response_model, messages):
                outer.calls.append(
                    {"model": model, "response_model": response_model, "messages": messages}
                )
                return outer._facts

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_local_extractor_uses_local_model_and_block_tagged_prompt():
    client = _FakeInstructorClient(_canned())
    ext = LocalExtractor(Settings(extract_backend="local", local_model="Qwen3-14B"), client=client)

    facts = ext.extract(_ir())

    assert facts.counterparty.value == "Acme Inc."
    call = client.calls[0]
    assert call["model"] == "Qwen3-14B"
    assert call["response_model"] is ContractFacts
    assert "[#/b/1]" in call["messages"][0]["content"]   # block-tagged context


def test_mlx_extractor_uses_mlx_model():
    client = _FakeInstructorClient(_canned())
    ext = MLXExtractor(Settings(extract_backend="mlx", mlx_model="qwen3:8b"), client=client)

    ext.extract(_ir())

    assert client.calls[0]["model"] == "qwen3:8b"


def test_on_device_extractors_construct_without_external_llm_flag():
    # allow_external_llm defaults False; the local path must not require it.
    LocalExtractor(Settings(extract_backend="local"), client=_FakeInstructorClient(_canned()))
    MLXExtractor(Settings(extract_backend="mlx"), client=_FakeInstructorClient(_canned()))


@pytest.mark.skipif(_openai_missing, reason="get_extractor builds a real OpenAI-compatible client")
def test_get_extractor_routes_local_and_mlx():
    from contract_rag.extract.extractor import get_extractor

    assert isinstance(get_extractor(Settings(extract_backend="local")), LocalExtractor)
    assert isinstance(get_extractor(Settings(extract_backend="mlx")), MLXExtractor)
