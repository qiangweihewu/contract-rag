from contract_rag.chunk.models import Chunk
from contract_rag.enrich.definitions import Definition, inject_definitions
from contract_rag.index.bm25 import BM25Index


def _c(cid, text, heading=None, block_ids=None):
    return Chunk(chunk_id=cid, doc_id="d", text=text,
                 block_ids=block_ids if block_ids is not None else [cid], heading=heading)


def _d(term, definition, block_id):
    return Definition(term=term, definition=definition, block_id=block_id)


# --- no-op cases -------------------------------------------------------------

def test_no_definitions_is_noop_and_identity():
    chunks = [_c("c1", "This Agreement is governed by New York law.")]
    out = inject_definitions(chunks, [])
    assert out[0] is chunks[0]
    assert out[0].index_text() == f"{chunks[0].heading or ''} {chunks[0].text}"


def test_no_matching_definitions_is_noop_and_identity():
    chunks = [_c("c1", "This Agreement is governed by New York law.")]
    defs = [_d("Confidential Information", "means secret stuff.", "def1")]
    out = inject_definitions(chunks, defs)
    assert out[0] is chunks[0]
    assert out[0].index_text() == f"{chunks[0].heading or ''} {chunks[0].text}"


def test_index_text_matches_old_doc_text_format_with_no_extra():
    c = _c("c1", "some text", heading="Heading")
    assert c.index_text() == "Heading some text"


def test_index_text_with_extra_prepends_between_heading_and_text():
    c = Chunk(chunk_id="c1", doc_id="d", text="body", block_ids=["c1"],
              heading="H", index_extra="[DEFINITIONS: X]")
    assert c.index_text() == "H [DEFINITIONS: X] body"


# --- injection basics ---------------------------------------------------------

def test_match_injects_into_index_extra_and_definition_block_ids_leaves_text_untouched():
    chunk = _c("c1", "The Company shall indemnify the Customer.")
    defs = [_d("Company", "Acme Corp and its subsidiaries.", "def1")]
    out = inject_definitions([chunk], defs)
    c2 = out[0]
    assert c2 is not chunk
    assert c2.text == chunk.text
    assert c2.block_ids == chunk.block_ids
    assert '"Company" means Acme Corp and its subsidiaries.' in c2.index_extra
    assert c2.index_extra.startswith("[DEFINITIONS: ") and c2.index_extra.endswith("]")
    assert c2.definition_block_ids == ["def1"]
    assert c2.index_text() == f"{c2.heading or ''} {c2.index_extra} {c2.text}"


def test_same_chunk_definition_skipped():
    # The definition's block_id is already among the chunk's block_ids -> nothing to
    # inject even though the term is used in the text.
    chunk = _c("c1", 'The "Company" means Acme Corp, referenced here as Company.',
               block_ids=["c1", "def1"])
    defs = [_d("Company", "means Acme Corp.", "def1")]
    out = inject_definitions([chunk], defs)
    assert out[0] is chunk


# --- case sensitivity / word boundary / plural --------------------------------

def test_case_sensitivity_lowercase_usage_does_not_trigger():
    chunk = _c("c1", "This agreement shall remain in effect.")
    defs = [_d("Agreement", "means this contract.", "def1")]
    out = inject_definitions([chunk], defs)
    assert out[0] is chunk


def test_word_boundary_terminal_does_not_match_term():
    chunk = _c("c1", "The Terminal building is not part of this deal.")
    defs = [_d("Term", "means the duration of this Agreement.", "def1")]
    out = inject_definitions([chunk], defs)
    assert out[0] is chunk


def test_plural_parties_matches_term_party():
    chunk = _c("c1", "The Parties agree to the following terms.")
    defs = [_d("Party", "a signatory to this Agreement.", "def1")]
    out = inject_definitions([chunk], defs)
    assert out[0] is not chunk
    assert '"Party" means a signatory to this Agreement.' in out[0].index_extra


def test_plain_plural_s_and_possessive_match():
    chunk_s = _c("c1", "All Products shall be delivered on time.")
    chunk_poss = _c("c2", "Vendor's Product's warranty applies here.")
    defs = [_d("Product", "means any item sold.", "def1")]
    out_s = inject_definitions([chunk_s], defs)
    out_poss = inject_definitions([chunk_poss], defs)
    assert out_s[0] is not chunk_s
    assert out_poss[0] is not chunk_poss


# --- caps ----------------------------------------------------------------------

def test_max_defs_per_chunk_respected():
    chunk = _c("c1", "Alpha Beta Gamma Delta all appear once each in this sentence.")
    defs = [
        _d("Alpha", "means the first term.", "d1"),
        _d("Beta", "means the second term.", "d2"),
        _d("Gamma", "means the third term.", "d3"),
        _d("Delta", "means the fourth term.", "d4"),
    ]
    out = inject_definitions([chunk], defs, max_defs_per_chunk=3)
    c2 = out[0]
    assert len(c2.definition_block_ids) == 3


def test_max_chars_per_chunk_respected():
    chunk = _c("c1", "Alpha Beta Gamma all appear once each in this very sentence.")
    long_def = "x" * 300
    defs = [
        _d("Alpha", long_def, "d1"),
        _d("Beta", long_def, "d2"),
        _d("Gamma", long_def, "d3"),
    ]
    out = inject_definitions([chunk], defs, max_defs_per_chunk=3, max_chars_per_chunk=600)
    c2 = out[0]
    assert len(c2.index_extra) <= 700  # well under 3 * 300+ chars, cap enforced
    assert len(c2.definition_block_ids) < 3


# --- ranking ---------------------------------------------------------------------

def test_frequency_ranking_prefers_more_frequent_term_when_budget_forces_choice():
    # "Alpha" used 3x, "Beta" used 1x; cap at 1 definition -> Alpha must win.
    chunk = _c("c1", "Alpha interacts with Alpha and then Alpha again, unlike Beta.")
    defs = [
        _d("Alpha", "means the frequent term.", "d1"),
        _d("Beta", "means the rare term.", "d2"),
    ]
    out = inject_definitions([chunk], defs, max_defs_per_chunk=1)
    c2 = out[0]
    assert c2.definition_block_ids == ["d1"]
    assert "Alpha" in c2.index_extra
    assert "Beta" not in c2.index_extra


def test_ranking_tiebreak_by_first_use_position():
    # Both Alpha and Beta used once each -> tie on frequency; Beta appears first in
    # the text, so it wins when budget forces a choice.
    chunk = _c("c1", "Beta comes before Alpha in this sentence.")
    defs = [
        _d("Alpha", "means the second term.", "d1"),
        _d("Beta", "means the first term.", "d2"),
    ]
    out = inject_definitions([chunk], defs, max_defs_per_chunk=1)
    c2 = out[0]
    assert c2.definition_block_ids == ["d2"]


# --- multiple chunks: no-match ones stay identity, matches get copied -----------

def test_multiple_chunks_mixed_match_and_no_match():
    matched = _c("c1", "The Company shall pay all fees.")
    unmatched = _c("c2", "Nothing relevant appears in this sentence.")
    defs = [_d("Company", "means Acme Corp.", "def1")]
    out = inject_definitions([matched, unmatched], defs)
    assert out[0] is not matched
    assert out[1] is unmatched


# --- bm25/dense hoist regression -------------------------------------------------

def test_bm25_retrieves_chunk_via_injected_definition_text_only():
    chunk = _c("c1", "The Company shall pay all fees within thirty days.")
    other = _c("c2", "This clause discusses termination rights only.")
    defs = [_d("Company", "means Acme Corp and its worldwide subsidiaries.", "def1")]
    injected = inject_definitions([chunk, other], defs)

    idx = BM25Index()
    idx.add(injected)
    res = idx.search("worldwide subsidiaries", k=2)
    assert res
    assert res[0][0].chunk_id == "c1"
