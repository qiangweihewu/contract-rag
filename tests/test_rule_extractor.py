from contract_rag.extract.rules import (
    RuleExtractor,
    find_counterparty,
    find_effective_date,
    find_governing_law,
)
from contract_rag.extract.schema import ContractFacts
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, bid):
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_find_governing_law_extracts_state_with_source():
    ir = _ir([
        _b("Recitals appear here first.", "#/b/0"),
        _b("This Agreement shall be governed by the laws of the State of New York.", "#/b/1"),
    ])
    clause = find_governing_law(ir)
    assert clause.value == "New York"
    assert clause.source_block_id == "#/b/1"
    assert clause.confidence > 0


def test_find_governing_law_empty_when_absent():
    clause = find_governing_law(_ir([_b("There is no choice-of-law clause here.", "#/b/0")]))
    assert clause.value == ""
    assert clause.confidence == 0.0


def test_find_effective_date_extracts_named_date_near_effective():
    clause = find_effective_date(_ir([_b("This Agreement is effective as of January 1, 2026.", "#/b/1")]))
    assert clause.value == "January 1, 2026"
    assert clause.source_block_id == "#/b/1"


def test_find_effective_date_extracts_numeric_date():
    clause = find_effective_date(_ir([_b("Effective Date: 03/15/2025 per the parties.", "#/b/1")]))
    assert clause.value == "03/15/2025"


def test_find_counterparty_captures_both_parties():
    ir = _ir([_b("This Agreement is entered into by and between Acme Inc. and Globex LLC.", "#/b/1")])
    clause = find_counterparty(ir)
    assert "Acme" in clause.value
    assert "Globex" in clause.value
    assert clause.source_block_id == "#/b/1"


def _norm_set(strings):
    from contract_rag.eval.golden import normalize  # the same canonicalizer the metric uses

    return {normalize(s) for s in strings}


def test_party_entities_extracts_corporate_names_past_aliases():
    from contract_rag.extract.rules import party_entities

    text = ('by and between Birch First Global Investments Inc., a Nevada corporation '
            '("Company") and Mount Knowledge Holdings Inc., a Delaware corporation')
    ents = _norm_set(party_entities(text))
    assert "birch first global investments inc" in ents
    assert "mount knowledge holdings inc" in ents


def test_party_entities_handles_all_caps_and_drops_aliases():
    from contract_rag.extract.rules import party_entities

    raw = ("['BIRCH FIRST GLOBAL INVESTMENTS INC.', 'MA', 'Marketing Affiliate', "
           "'MOUNT KNOWLEDGE HOLDINGS INC.', 'Company']")
    ents = _norm_set(party_entities(raw))
    assert "birch first global investments inc" in ents
    assert "mount knowledge holdings inc" in ents
    assert "ma" not in ents
    assert "company" not in ents          # bare defined-term alias, not an entity


def test_party_entities_handles_comma_before_suffix():
    from contract_rag.extract.rules import party_entities

    assert _norm_set(party_entities("Acme, Inc.")) == {"acme inc"}


def test_party_entities_drops_jurisdiction_descriptors_keeps_real_names():
    from contract_rag.extract.rules import party_entities

    text = "Commnet Wireless, LLC, a Delaware limited liability company, and AT&T Mobility LLC"
    ents = _norm_set(party_entities(text))
    assert "commnet wireless llc" in ents
    assert "att mobility llc" in ents
    assert not any(e.startswith("delaware") for e in ents)   # "Delaware limited ..." descriptor dropped
    # a real company that begins with a place name is still kept
    keep = _norm_set(party_entities("This is signed by New York Life Insurance Company today."))
    assert any("new york life" in e for e in keep)


def test_find_counterparty_returns_entity_set_through_paren_aliases():
    ir = _ir([_b('This is made by and between Acme Inc. (the "Buyer") and Globex LLC.', "#/b/1")])
    from contract_rag.extract.rules import party_entities

    ents = _norm_set(party_entities(find_counterparty(ir).value))
    assert ents == {"acme inc", "globex llc"}     # both parties, not truncated at "("


def test_rule_extractor_returns_full_contractfacts():
    ir = _ir([
        _b("This Agreement is entered into by and between Acme Inc. and Globex LLC.", "#/b/0"),
        _b("It is effective as of January 1, 2026.", "#/b/1"),
        _b("This Agreement shall be governed by the laws of the State of New York.", "#/b/2"),
    ])
    facts = RuleExtractor().extract(ir)
    assert isinstance(facts, ContractFacts)
    assert "Acme" in facts.counterparty.value
    assert facts.effective_date.value == "January 1, 2026"
    assert facts.governing_law.value == "New York"


def test_extracted_values_satisfy_source_attribution():
    # Each value must literally appear in its cited block — the metrics' source-attribution gate.
    from contract_rag.eval.metrics import source_attribution_ok

    ir = _ir([
        _b("This Agreement is entered into by and between Acme Inc. and Globex LLC.", "#/b/0"),
        _b("It is effective as of January 1, 2026.", "#/b/1"),
        _b("This Agreement shall be governed by the laws of the State of New York.", "#/b/2"),
    ])
    facts = RuleExtractor().extract(ir)
    ok = source_attribution_ok(facts, ir)
    populated = {k: v for k, v in ok.items() if getattr(facts, k).value}
    assert all(populated.values()), populated


def test_jurisdiction_in_finds_longest_match():
    from contract_rag.extract.rules import jurisdiction_in

    assert jurisdiction_in("under the laws of West Virginia hereby") == "West Virginia"
    assert jurisdiction_in("governed by the laws of the State of New York.") == "New York"
    assert jurisdiction_in("no place named here") is None


def test_find_governing_law_matches_jurisdiction_without_state_of():
    ir = _ir([_b("This Agreement is governed by California law.", "#/b/1")])
    assert find_governing_law(ir).value == "California"


def test_find_total_value_extracts_dollar_amount():
    from contract_rag.extract.rules import find_total_value

    ir = _ir([_b("The total contract value shall be $1,250,000.00 over the term.", "#/b/1")])
    c = find_total_value(ir)
    assert "1,250,000" in c.value
    assert c.source_block_id == "#/b/1"


def test_find_termination_notice_days_extracts_day_count():
    from contract_rag.extract.rules import find_termination_notice_days

    ir = _ir([_b("Either party may terminate upon ninety (90) days prior written notice.", "#/b/1")])
    c = find_termination_notice_days(ir)
    assert "90" in c.value
    assert c.source_block_id == "#/b/1"


def test_find_auto_renewal_detects_automatic_language():
    from contract_rag.extract.rules import find_auto_renewal

    yes = _ir([_b("This Agreement shall automatically renew for successive one-year terms.", "#/b/1")])
    assert find_auto_renewal(yes).value == "yes"
    no = _ir([_b("This Agreement has a fixed term of three years.", "#/b/1")])
    assert find_auto_renewal(no).value == ""


def test_rule_extractor_populates_all_six_fields():
    ir = _ir([
        _b("by and between Acme Inc. and Globex LLC.", "#/b/0"),
        _b("This is effective as of January 1, 2026.", "#/b/1"),
        _b("Governed by the laws of the State of New York.", "#/b/2"),
        _b("The total value is $500,000.", "#/b/3"),
        _b("Terminable upon thirty (30) days notice.", "#/b/4"),
        _b("It shall automatically renew for successive terms.", "#/b/5"),
    ])
    f = RuleExtractor().extract(ir)
    assert "Acme" in f.counterparty.value
    assert f.effective_date.value == "January 1, 2026"
    assert f.governing_law.value == "New York"
    assert "500,000" in f.total_value.value
    assert "30" in f.termination_notice_days.value
    assert f.auto_renewal.value == "yes"


def test_get_extractor_rule_backend_round_trips():
    from contract_rag.config import Settings
    from contract_rag.extract.extractor import get_extractor

    ext = get_extractor(Settings(extract_backend="rule"))
    facts = ext.extract(_ir([_b("This is governed by the laws of the State of Delaware.", "#/b/1")]))
    assert facts.governing_law.value == "Delaware"
