from __future__ import annotations


def test_contract_facts_canonical_home():
    from contract_rag.extract.schema import ContractFacts as Shim
    from contract_rag.verticals.contract.schema import ContractFacts as Canonical
    assert Shim is Canonical
    assert Canonical.FIELD_NAMES[0] == "counterparty"
