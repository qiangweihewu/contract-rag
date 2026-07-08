from contract_rag.extract.schema import ContractFacts, ExtractedClause


def test_field_names_are_cuad_aligned():
    assert ContractFacts.FIELD_NAMES == (
        "counterparty",
        "effective_date",
        "governing_law",
        "total_value",
        "termination_notice_days",
        "auto_renewal",
    )
    assert ContractFacts.SET_FIELDS == ("counterparty",)
    assert ContractFacts.JUDGMENT_FIELDS == ("auto_renewal",)


def test_phase4_fields_default_to_empty():
    # 3-field construction stays valid; the new fields default to empty clauses
    facts = ContractFacts(
        counterparty=ExtractedClause(), effective_date=ExtractedClause(), governing_law=ExtractedClause(),
    )
    assert facts.total_value.value == ""
    assert facts.termination_notice_days.value == ""
    assert facts.auto_renewal.value == ""


def test_empty_clause_defaults_are_safe():
    c = ExtractedClause()
    assert c.value == ""
    assert c.source_block_id is None
    assert c.confidence == 0.0


def test_contractfacts_constructs_from_clauses():
    facts = ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#/b/1", confidence=0.9),
        effective_date=ExtractedClause(value="2026-01-01", source_block_id="#/b/3", confidence=0.8),
        governing_law=ExtractedClause(),
    )
    assert facts.counterparty.value == "Acme Inc."
    assert facts.governing_law.value == ""
