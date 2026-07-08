"""Unit tests for the `constrained` extraction backend — server-side json_schema
structured decoding over an OpenAI-compatible endpoint (vLLM/SGLang/Ollama).
A fake OpenAI-SDK-shaped client is injected through the DI seam so tests stay
dep-free (no openai import, no network). Covers: happy path + request wiring,
schema strictness, the two documented bounded repairs (missing `#` prefix on
source_block_id; an accidental extra nesting level), validation failure → empty
field (miss, never invention), repair counting, token accounting, vertical
genericity (NDAFacts), and the no-gate posture."""

import importlib.util
import json

import pytest

from contract_rag.config import Settings
from contract_rag.extract.constrained import (
    ConstrainedExtractor,
    strict_schema,
    validate_with_repairs,
)
from contract_rag.extract.schema import ContractFacts
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR
from contract_rag.obs.counters import InMemoryCounterStore

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


def _payload(counterparty=None) -> str:
    fields = {name: {"value": "", "source_block_id": None, "confidence": 0.0}
              for name in ContractFacts.FIELD_NAMES}
    if counterparty is not None:
        fields["counterparty"] = counterparty
    return json.dumps(fields)


class _FakeClient:
    """Mimics the raw OpenAI SDK `client.chat.completions.create(...)` surface
    (choices[0].message.content + usage.total_tokens) and records the call."""

    def __init__(self, content: str, total_tokens: int | None = None):
        self.calls: list[dict] = []
        outer = self

        class _Msg:
            pass

        class _Choice:
            message = _Msg()

        class _Usage:
            pass

        class _Completion:
            choices = [_Choice()]
            usage = _Usage() if total_tokens is not None else None

        _Choice.message.content = content
        if total_tokens is not None:
            _Completion.usage.total_tokens = total_tokens
        completion = _Completion()

        class _Completions:
            def create(self, *, model, messages, response_format):
                outer.calls.append({"model": model, "messages": messages,
                                    "response_format": response_format})
                return completion

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _extractor(content: str, total_tokens: int | None = None, **kwargs) -> ConstrainedExtractor:
    return ConstrainedExtractor(
        Settings(extract_backend="constrained"),
        client=_FakeClient(content, total_tokens), **kwargs,
    )


# --- happy path + wiring ---------------------------------------------------


def test_happy_path_returns_facts_and_wires_request():
    good = {"value": "Acme Inc.", "source_block_id": "#/b/1", "confidence": 0.9}
    ext = _extractor(_payload(good))

    facts = ext.extract(_ir())

    assert isinstance(facts, ContractFacts)
    assert facts.counterparty.value == "Acme Inc."
    assert facts.counterparty.source_block_id == "#/b/1"
    assert ext.last_repairs == {}
    call = ext._client.calls[0]
    assert call["model"] == "Qwen3-14B"                       # falls back to local_model
    assert "[#/b/1]" in call["messages"][0]["content"]        # block-tagged context
    rf = call["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "ContractFacts"
    assert rf["json_schema"]["schema"]["additionalProperties"] is False


def test_constrained_model_and_endpoint_settings_win_over_local():
    ext = ConstrainedExtractor(
        Settings(extract_backend="constrained",
                 constrained_model="qwen2.5:32b-instruct", local_model="Qwen3-14B"),
        client=_FakeClient(_payload()),
    )
    ext.extract(_ir())
    assert ext._client.calls[0]["model"] == "qwen2.5:32b-instruct"


def test_constructs_without_external_llm_flag():
    # allow_external_llm defaults False; the constrained (local-server) path is ungated.
    ConstrainedExtractor(Settings(extract_backend="constrained"), client=_FakeClient("{}"))


# --- schema strictness ------------------------------------------------------


def test_strict_schema_tightens_every_object():
    schema = strict_schema(ContractFacts)
    clause = schema["$defs"]["ExtractedClause"]
    for obj in (schema, clause):
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
    # keywords grammar compilers reject are gone; Pydantic re-enforces bounds later
    dumped = json.dumps(schema)
    for kw in ("default", "minimum", "maximum"):
        assert f'"{kw}"' not in dumped


# --- documented repairs -----------------------------------------------------


def test_hash_prefix_repair_restores_documented_mlx_bug():
    # the model cites "/b/1" instead of "#/b/1" — repaired because "#/b/1" exists in the IR
    ext = _extractor(_payload({"value": "Acme Inc.", "source_block_id": "/b/1",
                               "confidence": 0.9}))
    facts = ext.extract(_ir())
    assert facts.counterparty.source_block_id == "#/b/1"
    assert ext.last_repairs == {"hash_prefix": 1}


def test_hash_prefix_repair_never_creates_an_attribution():
    # cited id matches no block even with '#' — left untouched (verify() will quarantine)
    ext = _extractor(_payload({"value": "Acme Inc.", "source_block_id": "/nope",
                               "confidence": 0.9}))
    facts = ext.extract(_ir())
    assert facts.counterparty.source_block_id == "/nope"
    assert ext.last_repairs == {}


def test_unwrap_field_repair_recovers_extra_nesting():
    # the documented TOOLS-mode failure: an extra ExtractedClause wrapper per field
    wrapped = {"ExtractedClause": {"value": "Acme Inc.", "source_block_id": "#/b/1",
                                   "confidence": 0.9}}
    ext = _extractor(_payload(wrapped))
    facts = ext.extract(_ir())
    assert facts.counterparty.value == "Acme Inc."
    assert ext.last_repairs == {"unwrap_field": 1}


def test_unwrap_root_repair_recovers_wrapped_facts_object():
    inner = json.loads(_payload({"value": "Acme Inc.", "source_block_id": "#/b/1",
                                 "confidence": 0.9}))
    ext = _extractor(json.dumps({"ContractFacts": inner}))
    facts = ext.extract(_ir())
    assert facts.counterparty.value == "Acme Inc."
    assert ext.last_repairs == {"unwrap_root": 1}


def test_repairs_compose_unwrap_then_hash_prefix():
    wrapped = {"ExtractedClause": {"value": "Acme Inc.", "source_block_id": "/b/1",
                                   "confidence": 0.9}}
    ext = _extractor(_payload(wrapped))
    facts = ext.extract(_ir())
    assert facts.counterparty.source_block_id == "#/b/1"
    assert ext.last_repairs == {"unwrap_field": 1, "hash_prefix": 1}


# --- failure → miss, never invention ----------------------------------------


def test_invalid_field_degrades_to_empty_clause_others_survive():
    bad = {"value": ["not", "a", "string"], "source_block_id": 3, "confidence": "x"}
    payload = json.loads(_payload(bad))
    payload["governing_law"] = {"value": "State of New York",
                                "source_block_id": "#/b/1", "confidence": 0.8}
    ext = _extractor(json.dumps(payload))
    facts = ext.extract(_ir())
    assert facts.counterparty.value == ""                     # miss, not invention
    assert facts.counterparty.source_block_id is None
    assert facts.governing_law.value == "State of New York"   # one bad field ≠ a lost doc
    assert ext.last_repairs == {"field_dropped": 1}


def test_unparseable_payload_yields_all_empty_facts():
    ext = _extractor("not json at all")
    facts = ext.extract(_ir())
    assert all(getattr(facts, n).value == "" for n in ContractFacts.FIELD_NAMES)
    assert ext.last_repairs == {"payload_invalid": 1}


def test_fenced_payload_is_tolerated_and_counted():
    fenced = "```json\n" + _payload({"value": "Acme Inc.", "source_block_id": "#/b/1",
                                     "confidence": 0.9}) + "\n```"
    ext = _extractor(fenced)
    facts = ext.extract(_ir())
    assert facts.counterparty.value == "Acme Inc."
    assert ext.last_repairs == {"fence_strip": 1}


# --- accounting -------------------------------------------------------------


def test_token_accounting_uses_server_usage_when_present():
    ext = _extractor(_payload(), total_tokens=1234)
    ext.extract(_ir())
    assert ext.last_tokens == 1234


def test_token_accounting_estimates_without_usage():
    ext = _extractor(_payload())
    ext.extract(_ir())
    assert ext.last_tokens > 0  # prompt-length estimate, parity with _InstructorExtractor


def test_repairs_flow_into_injected_counter_store():
    counters = InMemoryCounterStore()
    ext = _extractor(_payload({"value": "Acme Inc.", "source_block_id": "/b/1",
                               "confidence": 0.9}), counters=counters)
    ext.extract(_ir())
    ext.extract(_ir())
    assert counters.value("extract.constrained.repair.hash_prefix") == 2


# --- vertical genericity ----------------------------------------------------


def test_generic_over_verticals_nda_facts():
    from contract_rag.verticals.nda.schema import NDAFacts
    from contract_rag.verticals.registry import get_vertical

    nda = get_vertical("nda")
    fields = {name: {"value": "", "source_block_id": None, "confidence": 0.0}
              for name in NDAFacts.FIELD_NAMES}
    fields["disclosing_party"] = {"value": "Acme Inc.", "source_block_id": "/b/1",
                                  "confidence": 0.9}
    ext = ConstrainedExtractor(Settings(extract_backend="constrained"),
                               vertical=nda, client=_FakeClient(json.dumps(fields)))
    facts = ext.extract(_ir())
    assert isinstance(facts, NDAFacts)
    assert facts.disclosing_party.value == "Acme Inc."
    assert facts.disclosing_party.source_block_id == "#/b/1"  # repair works cross-vertical
    assert ext._client.calls[0]["response_format"]["json_schema"]["name"] == "NDAFacts"


def test_validate_with_repairs_is_pure_and_reusable():
    data = {"counterparty": {"value": "Acme Inc.", "source_block_id": "#/b/1",
                             "confidence": 0.9}}
    facts, repairs = validate_with_repairs(data, ContractFacts, frozenset({"#/b/1"}))
    assert facts.counterparty.value == "Acme Inc."
    assert facts.effective_date.value == ""  # absent fields default to empty clauses
    assert repairs == {}


# --- routing ----------------------------------------------------------------


@pytest.mark.skipif(_openai_missing, reason="get_extractor builds a real OpenAI-compatible client")
def test_get_extractor_routes_constrained():
    from contract_rag.extract.extractor import get_extractor

    assert isinstance(get_extractor(Settings(extract_backend="constrained")),
                      ConstrainedExtractor)
