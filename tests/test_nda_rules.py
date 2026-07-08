from __future__ import annotations

from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.nda.rules import NDARuleExtractor, duration_in, return_signal


def _b(bid: str, text: str) -> DocBlock:
    return DocBlock(block_id=bid, type=BlockType.PARAGRAPH, text=text,
                    confidence=1.0, source_engine="synthetic")


def _ir() -> DocumentIR:
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="text/plain", metadata={}, blocks=[
        _b("#/b/0", 'This Non-Disclosure Agreement is entered into by and between '
                    'Acme Robotics Inc. ("Disclosing Party") and Beilan Systems LLC '
                    '("Receiving Party").'),
        _b("#/b/1", "This Agreement is effective as of March 3, 2025."),
        _b("#/b/2", "This Agreement shall remain in full force and effect for two (2) years."),
        _b("#/b/3", "The obligations of confidentiality shall survive for five (5) years."),
        _b("#/b/4", "Upon termination, the Receiving Party shall return or destroy all Confidential Information."),
        _b("#/b/5", "This Agreement shall be governed by the laws of the State of New York."),
    ])


def test_duration_and_return_helpers():
    assert duration_in("two (2) years") == "2 years"
    assert duration_in("thirty-six (36) months") == "36 months"
    assert duration_in("no duration here") == ""
    assert return_signal("shall return or destroy all materials") is True
    assert return_signal("nothing relevant") is False


def _ir_of(*texts: str) -> DocumentIR:
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="text/plain", metadata={},
                      blocks=[_b(f"#/b/{i}", t) for i, t in enumerate(texts)])


def test_rule_extractor_fields():
    f = NDARuleExtractor().extract(_ir())
    assert "Acme Robotics Inc" in f.disclosing_party.value
    assert f.disclosing_party.source_block_id == "#/b/0"
    assert "Beilan Systems LLC" in f.receiving_party.value
    assert f.effective_date.value == "March 3, 2025"
    assert duration_in(f.term.value) == "2 years"
    assert duration_in(f.confidentiality_period.value) == "5 years"
    assert f.return_of_materials.value == "yes"
    assert f.governing_law.value == "New York"
    # every non-empty scalar/set value is a verbatim span of its cited block
    assert f.governing_law.value in _ir().blocks[5].text


# ---------------------------------------------------------- party preamble fallback
# Real SEC NDAs rarely carry "Disclosing Party:"-style role labels; the preamble
# ("by and between X ... and Y") is the reliable party source. Labels still win.

def test_party_fallback_preamble_two_entities_first_is_disclosing():
    ir = _ir_of(
        "CONFIDENTIALITY AND NON-COMPETITION AGREEMENT",
        "This Agreement is entered into as of January 1, 2008, by and between "
        "Verso Paper Holdings LLC, a Delaware limited liability company ('Verso Paper'), "
        "and Acme Consulting Corp., a New York corporation ('Consultant').",
    )
    f = NDARuleExtractor().extract(ir)
    assert f.disclosing_party.value == "Verso Paper Holdings LLC"
    assert f.receiving_party.value == "Acme Consulting Corp"
    assert f.disclosing_party.source_block_id == "#/b/1"
    assert f.receiving_party.source_block_id == "#/b/1"
    # fallback confidence is marked lower than the label heuristic's 0.7
    assert f.disclosing_party.confidence < 0.7


def test_party_fallback_single_entity_goes_to_disclosing_only():
    ir = _ir_of(
        "This Agreement is made and entered as of January 30, 2006, between "
        "ASSET ACCEPTANCE CAPITAL CORP. a Delaware corporation (the 'Company'), "
        "and James C. Lee ('Employee')."
    )
    f = NDARuleExtractor().extract(ir)
    assert f.disclosing_party.value == "ASSET ACCEPTANCE CAPITAL CORP"
    assert f.receiving_party.value == ""


def test_party_fallback_by_and_among_collects_all_entities():
    ir = _ir_of(
        "THIS NON-DISCLOSURE AGREEMENT, dated this 11th day of January, 2012, is by and "
        "among First Financial Northwest, Inc. (the 'Company'), Stilwell Value LLC and "
        "Stilwell Partners LP (collectively, the 'Stilwell Group')."
    )
    f = NDARuleExtractor().extract(ir)
    assert f.disclosing_party.value == "First Financial Northwest, Inc"
    assert f.receiving_party.value == "Stilwell Value LLC; Stilwell Partners LP"


def test_party_labels_take_precedence_over_fallback():
    ir = _ir_of(
        'This NDA is entered into by and between Acme Robotics Inc. ("Disclosing Party") '
        'and Beilan Systems LLC ("Receiving Party").'
    )
    f = NDARuleExtractor().extract(ir)
    assert f.disclosing_party.value == "Acme Robotics Inc"
    assert f.receiving_party.value == "Beilan Systems LLC"
    assert f.disclosing_party.confidence == 0.7  # label heuristic, not the fallback


def test_party_fallback_fills_only_the_empty_role():
    # disclosing found via label; receiving has no label -> preamble fills it
    ir = _ir_of(
        'This NDA is made by and between Acme Robotics Inc. (the "Disclosing Party") '
        "and Beilan Systems LLC."
    )
    f = NDARuleExtractor().extract(ir)
    assert f.disclosing_party.value == "Acme Robotics Inc"
    assert f.disclosing_party.confidence == 0.7
    assert f.receiving_party.value == "Beilan Systems LLC"
    assert f.receiving_party.confidence < 0.7


def test_party_fallback_head_scan_for_parties_list_style():
    # "PARTIES:" list style with no 'between' anywhere (e.g. Nike employment NDAs)
    ir = _ir_of(
        "COVENANT NOT TO COMPETE AND NON-DISCLOSURE AGREEMENT",
        "PARTIES:",
        "Eric Dean Sprunk ('EMPLOYEE')",
        "and NIKE, Inc., divisions, subsidiaries and affiliates. ('NIKE'):",
    )
    f = NDARuleExtractor().extract(ir)
    assert f.disclosing_party.value == "NIKE, Inc"
    assert f.disclosing_party.source_block_id == "#/b/3"


def test_party_fallback_stays_empty_without_entities():
    ir = _ir_of("This Agreement is between the parties named on the signature page.")
    f = NDARuleExtractor().extract(ir)
    assert f.disclosing_party.value == ""
    assert f.receiving_party.value == ""


# ------------------------------------------------------------- effective date forms
# Real SEC NDAs write dates as legalese ordinals ("the 6th day of January, 2012"),
# month-ordinal ("April 6th, 2005"), day-month-year ("6 January 2012"), or a
# standalone letterhead/signature line ("December 11,2014", "Date: 5/12/09").

def _date_of(*texts: str) -> str:
    return NDARuleExtractor().extract(_ir_of(*texts)).effective_date.value


def test_date_ordinal_day_of_form():
    v = _date_of("This Agreement (the 'Agreement') dated this 6th day of January, 2012 "
                 "is entered into by and between the parties.")
    assert v == "6th day of January, 2012"


def test_date_day_of_form_without_comma_and_nbsp():
    v = _date_of("This Agreement is made as of the 4th day of May\xa02005 by and between the parties.")
    assert v == "4th day of May\xa02005"


def test_date_month_ordinal_form_with_nbsp_gap():
    v = _date_of("I, Hewes, Hap, as of \xa0April\xa06th\xa0\xa0\xa0, 2005, in consideration "
                 "of my continued employment agree as follows.")
    assert v == "April\xa06th\xa0\xa0\xa0, 2005"


def test_date_day_month_year_form():
    assert _date_of("This Agreement is dated 6 January 2012.") == "6 January 2012"


def test_date_executed_cue():
    v = _date_of("IN WITNESS WHEREOF, the parties hereto have executed this agreement "
                 "this 16th day of May, 2011.")
    assert v == "16th day of May, 2011"


def test_date_needs_a_nearby_cue_not_any_date():
    # a date deep in a definitions clause with no agreement-date cue must not win
    # over the properly cued one later in the document
    v = _date_of(
        "Information furnished on or after November 8, 2011 to the Recipient is covered.",
        "This Agreement is entered into as of March 1, 2015 between the parties.",
    )
    assert v == "March 1, 2015"


def test_date_standalone_letterhead_fallback():
    # letter-style NDA: no cue anywhere, the date is its own block
    v = _date_of("Ladies and Gentlemen:", "December\xa011,2014",
                 "The Confidential Information will be used solely for the Transaction.")
    assert v == "December\xa011,2014"


def test_date_signature_line_fallback():
    assert _date_of("I agree to the terms above.", "Date: 5/12/09") == "5/12/09"


def test_date_fallback_ignores_dates_embedded_in_prose():
    # no cue and the date sits inside a long uncued sentence -> stay empty
    v = _date_of("Information furnished on or after November 8, 2011 to the Recipient is covered.")
    assert v == ""


def test_date_absent_stays_empty():
    assert _date_of("This Agreement has no date at all.") == ""


# ----------------------------------------------------------------- term durations

def test_duration_word_numbers():
    assert duration_in("three years from the date of this letter") == "3 years"
    assert duration_in("a period of twelve months") == "12 months"
    # compound word numbers are NOT half-matched ("thirty-one" must not become 1)
    assert duration_in("thirty-one years") == ""


def test_term_shall_terminate_cue():
    ir = _ir_of(
        "20. Term. This Agreement shall terminate two years after the date of this Agreement."
    )
    f = NDARuleExtractor().extract(ir)
    assert f.term.value == "two years"
    assert f.term.source_block_id == "#/b/0"


def test_term_digit_form_still_extracted():
    ir = _ir_of("This Agreement shall remain in full force and effect for two (2) years.")
    f = NDARuleExtractor().extract(ir)
    assert duration_in(f.term.value) == "2 years"
