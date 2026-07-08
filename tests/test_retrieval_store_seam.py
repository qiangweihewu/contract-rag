"""The dense-store seam on evaluate_retrieval: lets the benchmark swap the in-memory
vector store for a PgVectorStore (one fresh store per doc) without changing results."""

from contract_rag.eval.golden import GoldenDoc
from contract_rag.eval.retrieval import evaluate_retrieval
from contract_rag.index.embed import HashingEmbedder
from contract_rag.index.store import InMemoryVectorStore
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR


def _block(bid, btype, text):
    return DocBlock(block_id=bid, type=btype, text=text,
                    bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                    confidence=1.0, source_engine="docling")


def _ir_for(_g) -> DocumentIR:
    return DocumentIR(
        doc_id="d1", source_uri="file:///d1.pdf", file_hash="h", mime_type="application/pdf",
        blocks=[
            _block("#/b/1", BlockType.HEADING, "Governing Law"),
            _block("#/b/2", BlockType.PARAGRAPH,
                   "This Agreement shall be governed by the laws of the State of New York."),
            _block("#/b/3", BlockType.HEADING, "Termination"),
            _block("#/b/4", BlockType.PARAGRAPH,
                   "Either party may terminate upon ninety (90) days written notice."),
        ],
        metadata={},
    )


def _golden():
    return [GoldenDoc(doc_id="d1", source_pdf="d1.pdf", facts={"governing_law": "New York"})]


def test_evaluate_retrieval_builds_one_dense_store_per_doc():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return InMemoryVectorStore()

    res = evaluate_retrieval(_golden(), _ir_for, HashingEmbedder(), k=5, dense_store_factory=factory)

    assert calls["n"] == 1                    # one fresh store per document
    assert res["recall"]["dense"] == 1.0      # the New York clause is retrieved


def test_dense_store_factory_matches_default_inmemory():
    base = evaluate_retrieval(_golden(), _ir_for, HashingEmbedder(), k=5)
    seamed = evaluate_retrieval(_golden(), _ir_for, HashingEmbedder(), k=5,
                                dense_store_factory=InMemoryVectorStore)

    assert base["recall"] == seamed["recall"]   # the seam doesn't change results
