"""Unit tests for the wrong_span re-attribution post-pass
(`contract_rag.extract.reattribute`). Hand-built IRs + facts, dep-free: covers a
mis-cited value being relocated to its real source block, a value present in no
block being left untouched (never invent), an already-correct citation being
left untouched, the counterparty entity-set containment path, and the
reading-order proximity tie-break among multiple candidate blocks."""
from __future__ import annotations

from contract_rag.extract.reattribute import ReattributingExtractor, reattribute_facts
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(block_id: str, text: str) -> DocBlock:
    return DocBlock(
        block_id=block_id, type=BlockType.PARAGRAPH, text=text,
        bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
        confidence=1.0, source_engine="docling",
    )


def _ir(*blocks: DocBlock) -> DocumentIR:
    return DocumentIR(
        doc_id="d1", source_uri="file:///x.pdf", file_hash="h",
        mime_type="application/pdf", blocks=list(blocks), metadata={},
    )


def _facts(**clauses) -> ContractFacts:
    base = {n: ExtractedClause() for n in ContractFacts.FIELD_NAMES}
    base.update(clauses)
    return ContractFacts(**base)


def test_wrong_block_is_repaired_to_the_correct_block():
    ir = _ir(
        _block("b0", "This Agreement is entered into as of January 1, 2020."),
        _block("b1", "The parties are XYZ Corp and Acme Inc., a Delaware corporation."),
        _block("b2", "This Agreement shall be governed by the laws of the State of New York."),
    )
    facts = _facts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b2", confidence=0.9),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired.counterparty.source_block_id == "b1"
    assert repaired.counterparty.value == "Acme Inc."  # value itself is never touched
    assert repairs == {"counterparty": 1}


def test_value_in_no_block_is_left_untouched():
    ir = _ir(
        _block("b0", "This Agreement is entered into as of January 1, 2020."),
        _block("b1", "The parties are Acme Inc."),
    )
    facts = _facts(
        governing_law=ExtractedClause(value="Texas", source_block_id="b0", confidence=0.9),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired.governing_law.source_block_id == "b0"  # untouched, never invented
    assert repaired.governing_law.value == "Texas"
    assert repairs == {}


def test_correct_citation_is_left_untouched():
    ir = _ir(
        _block("b0", "Preamble text."),
        _block("b1", "This Agreement shall be governed by the laws of the State of New York."),
    )
    facts = _facts(
        governing_law=ExtractedClause(value="State of New York", source_block_id="b1", confidence=0.9),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired is facts  # no updates -> same object, not just equal
    assert repairs == {}


def test_counterparty_entity_set_reattribution():
    ir = _ir(
        _block("b0", "Miscellaneous provisions."),
        _block("b1", "This Agreement is between Acme Inc. and Globex LLC."),
        _block("b2", "Notices shall be sent to the registered agent."),
    )
    facts = _facts(
        counterparty=ExtractedClause(
            value="Acme Inc.; Globex LLC", source_block_id="b2", confidence=0.5
        ),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired.counterparty.source_block_id == "b1"
    assert repairs == {"counterparty": 1}


def test_counterparty_partial_entity_match_is_not_attributed():
    # only one of the two entities appears in b1 -> not a qualifying block anywhere
    ir = _ir(
        _block("b0", "This Agreement is between Acme Inc. only."),
        _block("b1", "Wrong block."),
    )
    facts = _facts(
        counterparty=ExtractedClause(
            value="Acme Inc.; Globex LLC", source_block_id="b1", confidence=0.5
        ),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired.counterparty.source_block_id == "b1"  # left as-is; no block qualifies
    assert repairs == {}


def test_proximity_tie_break_prefers_nearest_block_in_reading_order():
    # "Acme Inc." appears in both b0 and b3; the wrong citation is b2, so the nearer
    # candidate (b3, distance 1) should win over the farther one (b0, distance 2).
    ir = _ir(
        _block("b0", "Recitals: Acme Inc. is a software company."),
        _block("b1", "Definitions section."),
        _block("b2", "Miscellaneous boilerplate — no party name here."),
        _block("b3", "Notices to Acme Inc. shall be sent to its registered office."),
        _block("b4", "Signature block."),
    )
    facts = _facts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b2", confidence=0.9),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired.counterparty.source_block_id == "b3"
    assert repairs == {"counterparty": 1}


def test_proximity_tie_break_other_direction():
    # symmetric check: wrong citation now closer to the earlier occurrence (b1 vs b0/b3).
    ir = _ir(
        _block("b0", "Recitals: Acme Inc. is a software company."),
        _block("b1", "Miscellaneous boilerplate — no party name here."),
        _block("b2", "Definitions section."),
        _block("b3", "Notices to Acme Inc. shall be sent to its registered office."),
    )
    facts = _facts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b1", confidence=0.9),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired.counterparty.source_block_id == "b0"
    assert repairs == {"counterparty": 1}


def test_judgment_fields_are_skipped():
    # auto_renewal is a JUDGMENT_FIELD: never span-attributed, so a "wrong" citation
    # (which isn't wrong by this pass's rules) must never be touched.
    ir = _ir(_block("b0", "This Agreement auto-renews annually."))
    facts = _facts(
        auto_renewal=ExtractedClause(value="yes", source_block_id="b0", confidence=0.9),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired is facts
    assert repairs == {}


def test_empty_clause_is_skipped():
    ir = _ir(_block("b0", "Acme Inc."))
    facts = _facts()  # every field empty

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired is facts
    assert repairs == {}


def test_invalid_original_source_block_id_still_finds_the_real_block():
    ir = _ir(
        _block("b0", "Recitals."),
        _block("b1", "The parties are Acme Inc."),
    )
    facts = _facts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#does-not-exist", confidence=0.5),
    )

    repaired, repairs = reattribute_facts(facts, ir)

    assert repaired.counterparty.source_block_id == "b1"
    assert repairs == {"counterparty": 1}


# --- ReattributingExtractor wrapper ------------------------------------------


class _FakeChild:
    def __init__(self, facts, last_tokens=42, last_cost_usd=0.01):
        self._facts = facts
        self.last_tokens = last_tokens
        self.last_cost_usd = last_cost_usd

    def extract(self, ir):
        return self._facts


def test_reattributing_extractor_wraps_a_child_and_repairs_its_output():
    ir = _ir(
        _block("b0", "Recitals."),
        _block("b1", "The parties are Acme Inc."),
    )
    child_facts = _facts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b0", confidence=0.9),
    )
    wrapper = ReattributingExtractor(_FakeChild(child_facts))

    out = wrapper.extract(ir)

    assert out.counterparty.source_block_id == "b1"
    assert wrapper.last_repairs == {"counterparty": 1}
    assert wrapper.last_tokens == 42          # passed through from the child
    assert wrapper.last_cost_usd == 0.01


def test_reattributing_extractor_no_op_when_already_correct():
    ir = _ir(_block("b0", "The parties are Acme Inc."))
    child_facts = _facts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="b0", confidence=0.9),
    )
    wrapper = ReattributingExtractor(_FakeChild(child_facts))

    out = wrapper.extract(ir)

    assert out.counterparty.source_block_id == "b0"
    assert wrapper.last_repairs == {}
