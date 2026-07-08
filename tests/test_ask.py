from contract_rag.demo.ask import RetrievedClause, answer_question
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(bid: str, btype: BlockType, text: str) -> DocBlock:
    return DocBlock(block_id=bid, type=btype, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir() -> DocumentIR:
    # Each clause under its own heading so chunk_ir emits one chunk per clause.
    return DocumentIR(
        doc_id="d1", source_uri="file:///x.pdf", file_hash="h", mime_type="application/pdf",
        blocks=[
            _block("#/b/1", BlockType.HEADING, "Governing Law"),
            _block("#/b/2", BlockType.PARAGRAPH,
                   "This Agreement shall be governed by the laws of the State of New York."),
            _block("#/b/3", BlockType.HEADING, "Termination"),
            _block("#/b/4", BlockType.PARAGRAPH,
                   "Either party may terminate this Agreement upon ninety (90) days written notice."),
            _block("#/b/5", BlockType.HEADING, "Parties"),
            _block("#/b/6", BlockType.PARAGRAPH,
                   "This Agreement is entered into by Acme Inc. and Globex LLC."),
        ],
        metadata={},
    )


def test_answer_question_returns_relevant_clause_with_provenance():
    res = answer_question(_ir(), "how many days notice to terminate", k=3)
    assert res, "expected at least one retrieved clause"
    assert isinstance(res[0], RetrievedClause)
    top = res[0]
    assert "ninety (90) days" in top.text          # the termination clause surfaced first
    assert top.block_ids == ["#/b/4"]              # provenance preserved through retrieval
    assert top.rank == 1


def test_answer_question_respects_k():
    res = answer_question(_ir(), "agreement", k=2)
    assert len(res) <= 2
    assert [r.rank for r in res] == list(range(1, len(res) + 1))


def test_answer_question_empty_query_returns_nothing():
    assert answer_question(_ir(), "   ", k=5) == []


def test_answer_question_empty_document_returns_nothing():
    empty = DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                       mime_type="application/pdf", blocks=[], metadata={})
    assert answer_question(empty, "anything", k=5) == []


def test_answer_question_carries_clause_type_and_heading():
    res = answer_question(_ir(), "which state law governs this agreement", k=3)
    gov = next(r for r in res if "New York" in r.text)
    assert gov.heading == "Governing Law"
    assert gov.clause_type is not None             # enrichment ran (rule-based clause_type)


def _ir_with_definition() -> DocumentIR:
    return DocumentIR(
        doc_id="d2", source_uri="file:///x.pdf", file_hash="h", mime_type="application/pdf",
        blocks=[
            _block("#/b/1", BlockType.HEADING, "Definitions"),
            _block("#/b/2", BlockType.PARAGRAPH,
                   '"Confidential Information" means any nonpublic information disclosed '
                   "by either party under this Agreement."),
            _block("#/b/3", BlockType.HEADING, "Termination"),
            _block("#/b/4", BlockType.PARAGRAPH,
                   "Upon breach involving Confidential Information, either party may "
                   "terminate this Agreement immediately."),
        ],
        metadata={},
    )


def test_answer_question_inject_definitions_flag_off_is_byte_identical():
    ir = _ir_with_definition()
    off1 = answer_question(ir, "terminate this agreement", k=3)
    off2 = answer_question(ir, "terminate this agreement", k=3, inject_definitions=False)
    assert off1 == off2
    for r in off1:
        assert r.definition_block_ids == []        # additive default, always empty off


def test_answer_question_inject_definitions_surfaces_definition_block_ids():
    ir = _ir_with_definition()
    res = answer_question(ir, "terminate this agreement", k=3, inject_definitions=True)
    term = next(r for r in res if "Confidential Information" in r.text and "terminate" in r.text.lower())
    assert term.definition_block_ids == ["#/b/2"]   # provenance of the injected definition
    assert "#/b/2" not in term.block_ids            # display text/attribution untouched
