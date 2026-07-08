"""Legal-domain primitives shared across verticals — dep-free unit coverage for the
symbols `contract.rules` and `nda.*` both rely on."""
from __future__ import annotations

from contract_rag.verticals.legal_common import (
    _DATE,
    _MONTHS,
    _is_jurisdiction_descriptor,
    jurisdiction_in,
    party_entities,
)


def test_months_covers_all_twelve():
    for m in ("January", "June", "December"):
        assert m in _MONTHS


def test_date_matches_prose_and_slash_forms():
    assert _DATE.search("effective July 11, 2006").group(0) == "July 11, 2006"
    assert _DATE.search("dated 7/11/2006").group(0) == "7/11/2006"
    # tolerates the stray space-before-comma some CUAD spans carry
    assert _DATE.search("July 11 , 2006").group(0) == "July 11 , 2006"


def test_jurisdiction_in_finds_longest_match_first():
    assert jurisdiction_in("governed by the laws of New York") == "New York"
    assert jurisdiction_in("a West Virginia corporation") == "West Virginia"
    assert jurisdiction_in("no jurisdiction mentioned here") is None


def test_is_jurisdiction_descriptor_flags_state_plus_lowercase_word():
    assert _is_jurisdiction_descriptor("Delaware limited liability company") is True
    assert _is_jurisdiction_descriptor("New York limited partnership") is True
    # "New York Life ..." — capitalized word after the jurisdiction is a party name
    assert _is_jurisdiction_descriptor("New York Life Insurance") is False


def test_party_entities_extracts_corporate_names_and_dedupes():
    text = "by and between Acme Inc. and Beta Corp., and later again Acme Inc."
    ents = party_entities(text)
    assert ents == ["Acme Inc", "Beta Corp"]


def test_party_entities_skips_jurisdiction_descriptors():
    text = "Acme Inc., a Delaware limited liability company, and Beta Corp."
    ents = party_entities(text)
    assert "Delaware limited liability company" not in ents
    assert "Acme Inc" in ents
    assert "Beta Corp" in ents


def test_party_entities_empty_on_no_match():
    assert party_entities("no corporate suffix in this sentence") == []
