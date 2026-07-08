from contract_rag.enrich.definitions import Definition, extract_definitions
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, type=BlockType.PARAGRAPH, bid=None, page=1):
    return DocBlock(block_id=bid or text[:8], type=type, text=text,
                    bbox=BoundingBox(page=page, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir(blocks):
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                      mime_type="application/pdf", blocks=blocks, metadata={})


def test_empty_ir_returns_empty_list():
    assert extract_definitions(_ir([])) == []


# --- Pattern 1: quoted-term-defines --------------------------------------

def test_pattern1_means_straight_quotes():
    ir = _ir([_b('"Confidential Information" means any information disclosed by a party. '
                 'This is a new sentence.', bid="p1")])
    defs = extract_definitions(ir)
    assert len(defs) == 1
    d = defs[0]
    assert isinstance(d, Definition)
    assert d.term == "Confidential Information"
    assert d.block_id == "p1"
    assert d.definition.startswith("means any information disclosed by a party.")
    assert "This is a new sentence" not in d.definition


def test_pattern1_curly_quotes():
    ir = _ir([_b('“Effective Date” means the date first written above.', bid="p1")])
    defs = extract_definitions(ir)
    assert len(defs) == 1
    assert defs[0].term == "Effective Date"
    assert defs[0].definition.startswith("means the date first written above.")


def test_pattern1_shall_mean_variant():
    ir = _ir([_b('"Term" shall mean the period specified in Section 2.', bid="p1")])
    defs = extract_definitions(ir)
    assert defs[0].term == "Term"
    assert defs[0].definition.startswith("shall mean")


def test_pattern1_has_the_meaning_variant():
    ir = _ir([_b('"Losses" has the meaning set forth in Section 9.', bid="p1")])
    defs = extract_definitions(ir)
    assert defs[0].term == "Losses"
    assert defs[0].definition.startswith("has the meaning")


def test_pattern1_refers_to_variant():
    ir = _ir([_b('"Products" refers to the goods listed in Exhibit A.', bid="p1")])
    defs = extract_definitions(ir)
    assert defs[0].term == "Products"
    assert defs[0].definition.startswith("refers to")


def test_pattern1_is_defined_as_variant():
    ir = _ir([_b('"Fee" is defined as the amount payable under Section 4.', bid="p1")])
    defs = extract_definitions(ir)
    assert defs[0].term == "Fee"
    assert defs[0].definition.startswith("is defined as")


def test_pattern1_two_defs_in_one_block_do_not_bleed_into_each_other():
    ir = _ir([_b('"Alpha" means the first thing. "Beta" means the second thing.', bid="p1")])
    defs = {d.term: d.definition for d in extract_definitions(ir)}
    assert defs["Alpha"] == "means the first thing."
    assert defs["Beta"] == "means the second thing."


# --- Pattern 2: parenthetical referent ------------------------------------

def test_pattern2_the_quoted_parenthetical():
    ir = _ir([_b('This Agreement is between Acme Corp. and Beta LLC (the "Company"), '
                 'effective as of the date below.', bid="p1")])
    defs = extract_definitions(ir)
    assert len(defs) == 1
    d = defs[0]
    assert d.term == "Company"
    assert d.block_id == "p1"
    assert '(the "Company")' in d.definition


def test_pattern2_each_a_and_collectively_variants():
    ir = _ir([
        _b('Acme Corp (each a "Party") agrees to the terms below.', bid="p1"),
        _b('The undersigned parties (collectively, the "Parties") agree as follows.', bid="p2"),
        _b('This document (hereinafter "Agreement") governs the relationship.', bid="p3"),
    ])
    defs = extract_definitions(ir)
    terms = {d.term for d in defs}
    assert terms == {"Party", "Parties", "Agreement"}


# --- Pattern 3: definition-list entry under a Definitions heading --------

def test_pattern3_fires_under_definitions_heading():
    ir = _ir([
        _b("1. Definitions", type=BlockType.HEADING, bid="h1"),
        _b('"Confidential Information": Any non-public information disclosed by either party.',
           bid="p1"),
        _b("Personal Data: Any data relating to an identified individual.", bid="p2"),
    ])
    defs = extract_definitions(ir)
    terms = {d.term: d for d in defs}
    assert "Confidential Information" in terms
    assert terms["Confidential Information"].definition.startswith("Any non-public")
    assert terms["Confidential Information"].block_id == "p1"
    assert "Personal Data" in terms
    assert terms["Personal Data"].block_id == "p2"


def test_pattern3_does_not_fire_outside_definitions_section():
    ir = _ir([
        _b("Payment Terms", type=BlockType.HEADING, bid="h1"),
        _b("Invoice Amount: The total due each month.", bid="p1"),
    ])
    defs = extract_definitions(ir)
    assert defs == []


def test_pattern3_does_not_fire_with_no_preceding_heading():
    ir = _ir([_b("Governing Body: The entity administering this agreement.", bid="p1")])
    defs = extract_definitions(ir)
    assert defs == []


# --- Dedupe -----------------------------------------------------------------

def test_dedupe_textually_first_wins_across_patterns_within_one_block():
    # Pattern 2 (parenthetical) appears TEXTUALLY before pattern 1 ("means") for the
    # same term in the same block — the textually-first (parenthetical) must win,
    # regardless of pattern processing order.
    ir = _ir([_b('Acme Corp (the "Company") is a Delaware corporation. '
                 '"Company" means Acme Corp. and its subsidiaries.', bid="p1")])
    defs = extract_definitions(ir)
    assert len(defs) == 1
    d = defs[0]
    assert d.term == "Company"
    assert '(the "Company")' in d.definition          # the parenthetical sentence won
    assert not d.definition.startswith("means")


def test_pattern1_verb_requires_word_boundary():
    # "shall meaningfully" must not false-match the "shall mean" verb alternative.
    ir = _ir([_b('"Term" shall meaningfully alter the obligations of the parties.',
                 bid="p1")])
    assert extract_definitions(ir) == []


def test_dedupe_first_occurrence_wins_case_insensitive():
    ir = _ir([
        _b('"Confidential Information" means the first definition given here.', bid="p1"),
        _b('"CONFIDENTIAL INFORMATION" means a different, later definition.', bid="p2"),
    ])
    defs = extract_definitions(ir)
    assert len(defs) == 1
    assert defs[0].block_id == "p1"
    assert "first definition" in defs[0].definition


# --- 300-char cap ------------------------------------------------------------

def test_definition_capped_at_300_chars():
    long_tail = "x" * 400
    ir = _ir([_b(f'"Widget" means a device that does the following: {long_tail}.', bid="p1")])
    defs = extract_definitions(ir)
    assert len(defs[0].definition) == 300


# --- Furniture skipped -------------------------------------------------------

def test_header_and_footer_blocks_skipped():
    ir = _ir([
        _b('"Ignored Term" means this should never be extracted.', type=BlockType.HEADER, bid="hdr"),
        _b('"Also Ignored" means this should never be extracted either.', type=BlockType.FOOTER, bid="ftr"),
    ])
    defs = extract_definitions(ir)
    assert defs == []


# --- Term validation ----------------------------------------------------------

def test_rejects_lowercase_and_too_short_terms():
    ir = _ir([
        _b('"ok" means nothing useful because it is lowercase.', bid="p1"),
        _b('"AB" means nothing useful because it is too short.', bid="p2"),
    ])
    defs = extract_definitions(ir)
    assert defs == []
