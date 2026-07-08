from contract_rag.config import Settings
from contract_rag.extract.extractor import LocalExtractor
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.obs.pricing import PRICES, estimate_tokens


def _ir():
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf",
                      blocks=[DocBlock(block_id="b1", type=BlockType.PARAGRAPH,
                                       text="Entered into by Acme Inc.", confidence=1.0,
                                       source_engine="docling")],
                      metadata={})


def _facts():
    return ContractFacts(counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b1"),
                         effective_date=ExtractedClause(), governing_law=ExtractedClause())


class _CreateOnlyClient:
    """Fake instructor client exposing ONLY `.create` (like the existing test fakes)."""
    def __init__(self, facts):
        outer = self
        class _Comps:
            def create(self, *, model, response_model, messages):
                return outer._facts
        class _Chat:
            completions = _Comps()
        self._facts = facts
        self.chat = _Chat()


class _CompletionClient:
    """Fake instructor client exposing `create_with_completion` -> (facts, completion.usage)."""
    def __init__(self, facts, total_tokens):
        outer = self
        class _Usage:
            total_tokens = 0
        class _Completion:
            usage = _Usage()
        class _Comps:
            def create_with_completion(self, *, model, response_model, messages):
                comp = _Completion(); comp.usage.total_tokens = outer._tokens
                return outer._facts, comp
        class _Chat:
            completions = _Comps()
        self._facts = facts
        self._tokens = total_tokens
        self.chat = _Chat()


def test_estimate_tokens_is_roughly_chars_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 40) == 10


def test_create_only_client_falls_back_to_estimate():
    ext = LocalExtractor(Settings(extract_backend="local"), client=_CreateOnlyClient(_facts()))
    ext.extract(_ir())
    assert ext.last_tokens > 0          # estimated from the prompt
    assert ext.last_cost_usd == 0.0     # local_model unpriced -> free (credential-free floor)


def test_completion_client_uses_real_usage_and_prices_it():
    PRICES["test-model-cost"] = 2.0  # $2 / 1K tokens
    ext = LocalExtractor(
        Settings(extract_backend="local", local_model="test-model-cost"),
        client=_CompletionClient(_facts(), total_tokens=1500),
    )
    ext.extract(_ir())
    assert ext.last_tokens == 1500
    assert round(ext.last_cost_usd, 4) == 3.0  # 1500/1000 * 2.0
