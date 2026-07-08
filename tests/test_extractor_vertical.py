from __future__ import annotations

from contract_rag.config import Settings
from contract_rag.extract.extractor import get_extractor
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract.rules import RuleExtractor
from contract_rag.verticals.contract.schema import ContractFacts


def test_rule_backend_returns_the_vertical_rule_extractor():
    ex = get_extractor(Settings(extract_backend="rule"))
    assert isinstance(ex, RuleExtractor)


def test_fake_backend_returns_empty_facts_of_vertical_model():
    ex = get_extractor(Settings(extract_backend="fake"))
    out = ex.extract(_min_ir())  # FakeExtractor ignores ir, returns its canned empty facts
    assert isinstance(out, ContractFacts)
    assert out.counterparty.value == ""


def test_instructor_extractor_uses_vertical_facts_model_as_response_model():
    from contract_rag.extract.extractor import _InstructorExtractor

    captured = {}

    class FakeCompletions:
        def create(self, *, model, response_model, messages):
            captured["response_model"] = response_model
            # supply the 3 required fields so ContractFacts() doesn't raise
            _e = ExtractedClause
            return response_model(counterparty=_e(), effective_date=_e(), governing_law=_e())

    class FakeClient:
        class chat:  # noqa: N801
            completions = FakeCompletions()

    ex = _InstructorExtractor(FakeClient(), "m")  # defaults to contract vertical
    ex.extract(_min_ir())
    assert captured["response_model"] is ContractFacts


def _min_ir():
    from contract_rag.ir import BlockType, DocBlock, DocumentIR
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", metadata={}, blocks=[
        DocBlock(block_id="b1", type=BlockType.PARAGRAPH, text="hello",
                 confidence=1.0, source_engine="docling")])
