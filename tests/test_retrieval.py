from contract_rag.chunk.models import Chunk
from contract_rag.eval.golden import GoldenDoc
from contract_rag.eval.retrieval import evaluate_retrieval, supports
from contract_rag.index.embed import HashingEmbedder
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _b(text, type=BlockType.PARAGRAPH, bid=None):
    return DocBlock(block_id=bid or text[:8], type=type, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _contract_ir():
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
                      blocks=[
                          _b("Recitals", type=BlockType.HEADING, bid="h0"),
                          _b("The parties enter this distribution arrangement.", bid="r0"),
                          _b("Governing Law", type=BlockType.HEADING, bid="h1"),
                          _b("This Agreement is governed by the laws of the State of New York.", bid="g0"),
                          _b("Parties", type=BlockType.HEADING, bid="h2"),
                          _b("This Agreement is by and between Acme Inc. and Globex LLC.", bid="p0"),
                          _b("Indemnification", type=BlockType.HEADING, bid="h3"),
                          _b("Each party shall indemnify the other against third-party claims.", bid="i0"),
                      ], metadata={})


def test_supports_checks_gold_value_in_chunk():
    c = Chunk(chunk_id="c", doc_id="d", block_ids=["b"],
              text="governed by the laws of the State of New York")
    assert supports(c, "governing_law", "New York") is True
    assert supports(c, "governing_law", "California") is False


def test_evaluate_retrieval_reports_recall_per_method():
    golden = [GoldenDoc(doc_id="d", source_pdf="d.pdf",
                        facts={"governing_law": "New York", "counterparty": "Acme Inc.; Globex LLC"})]
    res = evaluate_retrieval(golden, ir_for=lambda _g: _contract_ir(),
                             embedder=HashingEmbedder(dim=128), k=3)
    for method in ("bm25", "dense", "hybrid"):
        assert 0.0 <= res["recall"][method] <= 1.0
    assert res["n"] == 2                          # two labeled fields evaluated
    assert res["recall"]["hybrid"] > 0            # the clauses are retrievable


def test_evaluate_retrieval_default_is_byte_identical_and_has_no_split_keys():
    golden = [GoldenDoc(doc_id="d", source_pdf="d.pdf",
                        facts={"governing_law": "New York", "counterparty": "Acme Inc.; Globex LLC"})]
    res1 = evaluate_retrieval(golden, ir_for=lambda _g: _contract_ir(),
                              embedder=HashingEmbedder(dim=128), k=3)
    res2 = evaluate_retrieval(golden, ir_for=lambda _g: _contract_ir(),
                              embedder=HashingEmbedder(dim=128), k=3)
    assert res1 == res2                            # deterministic, no seams => identical
    assert set(res1.keys()) == {"recall", "n", "k"}  # no defs_split keys leak in by default


def _ir_with_placeholder():
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
                      blocks=[
                          _b("Governing Law", type=BlockType.HEADING, bid="h0"),
                          _b("PLACEHOLDER", bid="g0"),
                      ], metadata={})


def test_evaluate_retrieval_applies_chunk_transform():
    """The honesty guard lives here: `supports()` only ever reads `chunk.text`, so a
    transform can change WHICH chunks rank top-k but can never fabricate a recall hit
    that isn't literally present in the (transformed) display text."""
    golden = [GoldenDoc(doc_id="d", source_pdf="d.pdf", facts={"governing_law": "Nevada"})]

    def mark(chunks, ir):
        return [
            c.model_copy(update={
                "text": c.text.replace("PLACEHOLDER",
                                       "This Agreement is governed by the laws of the State of Nevada."),
            })
            for c in chunks
        ]

    baseline = evaluate_retrieval(golden, ir_for=lambda _g: _ir_with_placeholder(),
                                  embedder=HashingEmbedder(dim=64), k=3)
    assert baseline["recall"]["bm25"] == 0.0       # "Nevada" isn't in the untransformed text

    transformed = evaluate_retrieval(golden, ir_for=lambda _g: _ir_with_placeholder(),
                                     embedder=HashingEmbedder(dim=64), k=3, chunk_transform=mark)
    assert transformed["recall"]["bm25"] == 1.0    # the transform's text is what got indexed


def _defs_ir():
    return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
                      blocks=[
                          _b("Definitions", type=BlockType.HEADING, bid="h0"),
                          _b('"Confidential Information" means any nonpublic information '
                             "disclosed by either party under this Agreement.", bid="d0"),
                          _b("Governing Law", type=BlockType.HEADING, bid="h1"),
                          _b("This Agreement, including all Confidential Information, is "
                             "governed by the laws of the State of New York.", bid="g0"),
                          _b("Parties", type=BlockType.HEADING, bid="h2"),
                          _b("This Agreement is by and between Acme Inc. and Globex LLC.", bid="p0"),
                      ], metadata={})


def test_evaluate_retrieval_defs_split_tallies_dependent_vs_independent():
    golden = [GoldenDoc(doc_id="d", source_pdf="d.pdf",
                        facts={"governing_law": "New York", "counterparty": "Acme Inc.; Globex LLC"})]
    res = evaluate_retrieval(golden, ir_for=lambda _g: _defs_ir(),
                             embedder=HashingEmbedder(dim=128), k=5, defs_split=True)

    assert "recall_defs_dependent" in res and "recall_defs_independent" in res
    assert "n_defs_dependent" in res and "n_defs_independent" in res
    # governing_law's supporting chunk uses "Confidential Information" -> dependent;
    # counterparty's supporting chunk doesn't -> independent.
    assert res["n_defs_dependent"] == 1
    assert res["n_defs_independent"] == 1
    assert res["recall_defs_dependent"]["bm25"] == 1.0
    assert res["recall_defs_independent"]["bm25"] == 1.0
    for method in ("bm25", "dense", "hybrid"):
        assert method in res["recall_defs_dependent"]
        assert method in res["recall_defs_independent"]
