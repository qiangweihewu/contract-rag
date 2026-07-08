from contract_rag.chunk.models import Chunk
from contract_rag.enrich.enricher import classify_clause, enrich_chunks, permission_tags


def _chunk(heading, text):
    return Chunk(chunk_id="c", doc_id="d", text=text, block_ids=["b"], heading=heading)


def test_classify_clause_detects_common_types():
    assert classify_clause(_chunk("Governing Law", "governed by the laws of New York")) == "governing_law"
    assert classify_clause(_chunk("Term", "either party may terminate on notice")) == "termination"
    assert classify_clause(_chunk("Fees", "Customer shall pay fees of $500")) == "payment"
    assert classify_clause(_chunk("Confidentiality", "keep proprietary information secret")) == "confidentiality"
    assert classify_clause(_chunk("Misc", "The parties hereby agree to the following.")) == "other"


def test_permission_tags_are_rule_based_abac():
    fin = _chunk("Fees", "fees of $5,000 payable monthly")
    fin = fin.model_copy(update={"clause_type": classify_clause(fin)})
    assert "finance" in permission_tags(fin)

    conf = _chunk("Confidentiality", "keep all confidential information secret")
    conf = conf.model_copy(update={"clause_type": classify_clause(conf)})
    assert "restricted" in permission_tags(conf)


def test_enrich_chunks_sets_type_and_tags_without_mutating_input():
    chunks = [_chunk("Governing Law", "governed by the laws of Delaware")]
    out = enrich_chunks(chunks)
    assert out[0].clause_type == "governing_law"
    assert out[0].permission_tags                  # non-empty
    assert chunks[0].clause_type is None           # original untouched (immutable)
