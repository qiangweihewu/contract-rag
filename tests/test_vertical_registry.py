from __future__ import annotations

from contract_rag.config import Settings
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.base import Vertical
from contract_rag.verticals.registry import (
    default_vertical, get_vertical, get_vertical_for, register_vertical,
)


def test_contract_is_registered_and_satisfies_protocol():
    v = get_vertical("contract")
    assert isinstance(v, Vertical)
    assert v.name == "contract"
    assert v.field_names[0] == "counterparty"
    assert "counterparty" in v.set_fields and "auto_renewal" in v.judgment_fields


def test_get_vertical_for_settings():
    assert get_vertical_for(Settings(vertical="contract")).name == "contract"
    assert default_vertical().name == "contract"


def test_unknown_vertical_raises():
    import pytest
    with pytest.raises(NotImplementedError):
        get_vertical("does-not-exist")


def test_register_vertical_roundtrip():
    v = get_vertical("contract")
    register_vertical(v)  # idempotent
    assert get_vertical("contract") is v


def test_default_vertical_resolves_configured_env(monkeypatch):
    import contract_rag.verticals.registry as reg
    saved = dict(reg._REGISTRY)
    try:
        class _Stub:
            name = "stub_v"
        reg.register_vertical(_Stub())
        monkeypatch.setenv("VERTICAL", "stub_v")
        assert reg.default_vertical().name == "stub_v"
    finally:
        reg._REGISTRY.clear()
        reg._REGISTRY.update(saved)
    monkeypatch.delenv("VERTICAL", raising=False)
    assert reg.default_vertical().name == "contract"


def test_canonicalize_and_entities_and_rule_extractor():
    v = get_vertical("contract")
    assert v.canonicalize_value("governing_law", "the State of New York") == "New York"
    assert v.entities("by and between Acme Inc. and Beta Corp.")
    ir = DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                    mime_type="application/pdf", metadata={}, blocks=[
        DocBlock(block_id="b1", type=BlockType.PARAGRAPH,
                 text="governed by the laws of California",
                 confidence=1.0, source_engine="docling")])
    assert v.rule_extractor.extract(ir).governing_law.value == "California"
    assert v.empty_facts().counterparty.value == ""
