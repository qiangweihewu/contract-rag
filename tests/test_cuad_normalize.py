from contract_rag.verticals.contract.gold import (
    normalize_auto_renewal,
    normalize_counterparty,
    normalize_effective_date,
    normalize_facts,
    normalize_governing_law,
    normalize_termination_notice_days,
)


def test_normalize_governing_law_extracts_us_state_from_span_list():
    raw = "['This Agreement shall be governed by the laws of the State of Florida.']"
    assert normalize_governing_law(raw) == "Florida"


def test_normalize_governing_law_handles_two_word_state_and_province():
    assert normalize_governing_law("['... the laws of the State of New York.']") == "New York"
    assert normalize_governing_law("['... governed by laws of the Province of Ontario ...']") == "Ontario"


def test_normalize_governing_law_prefers_longest_jurisdiction():
    assert normalize_governing_law("['governed by the laws of West Virginia']") == "West Virginia"


def test_normalize_governing_law_passes_through_bare_jurisdiction():
    assert normalize_governing_law("New York") == "New York"


def test_normalize_governing_law_empty_when_no_known_jurisdiction():
    assert normalize_governing_law("['subject to all applicable law']") == ""
    assert normalize_governing_law("") == ""


def test_normalize_effective_date_pulls_date_from_span_list():
    assert normalize_effective_date("['November 15, 2012']") == "November 15, 2012"
    # CUAD spans sometimes have a stray space before the comma
    assert normalize_effective_date("['July 11 , 2006']") == "July 11 , 2006"


def test_normalize_effective_date_empty_when_span_is_prose():
    assert normalize_effective_date("['This agreement shall begin upon execution']") == ""


def test_normalize_counterparty_reduces_span_list_to_entity_set():
    raw = ("['BIRCH FIRST GLOBAL INVESTMENTS INC.', 'MA', 'Marketing Affiliate', "
           "'MOUNT KNOWLEDGE HOLDINGS INC.', 'Company']")
    out = normalize_counterparty(raw)
    assert "BIRCH FIRST GLOBAL INVESTMENTS INC" in out
    assert "MOUNT KNOWLEDGE HOLDINGS INC" in out
    assert "Marketing Affiliate" not in out      # aliases dropped


def test_normalize_termination_notice_days_extracts_integer():
    assert normalize_termination_notice_days("['ninety (90) days']") == "90"
    assert normalize_termination_notice_days("['30 days prior written notice']") == "30"
    assert normalize_termination_notice_days("") == ""


def test_normalize_auto_renewal_maps_to_yes_no():
    assert normalize_auto_renewal("['automatically renew for successive one-year terms']") == "yes"
    assert normalize_auto_renewal("['renewable only by mutual written agreement']") == "no"
    assert normalize_auto_renewal("") == ""


def test_normalize_facts_normalizes_all_three_fields():
    facts = {
        "counterparty": "['Acme Inc.', 'Buyer', 'Globex LLC']",
        "effective_date": "['November 15, 2012']",
        "governing_law": "['... the laws of the State of Texas.']",
    }
    out = normalize_facts(facts)
    assert "Acme Inc" in out["counterparty"] and "Globex LLC" in out["counterparty"]
    assert "Buyer" not in out["counterparty"]
    assert out["effective_date"] == "November 15, 2012"
    assert out["governing_law"] == "Texas"
