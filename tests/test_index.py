import math

from contract_rag.chunk.models import Chunk
from contract_rag.index.bm25 import BM25Index
from contract_rag.index.dense import DenseIndex
from contract_rag.index.embed import HashingEmbedder
from contract_rag.index.hybrid import HybridIndex, build_index


def _c(cid, text, heading=None, tags=None):
    return Chunk(chunk_id=cid, doc_id="d", text=text, block_ids=[cid], heading=heading,
                 permission_tags=tags or [])


_DOCS = [
    _c("law", "This Agreement shall be governed by the laws of the State of New York."),
    _c("pay", "Customer shall pay fees of $5,000 within thirty days of each invoice."),
    _c("term", "Either party may terminate this Agreement upon ninety days written notice."),
]


def test_bm25_ranks_lexical_match_first_and_empty_on_no_overlap():
    idx = BM25Index()
    idx.add(_DOCS)
    res = idx.search("governing law New York", k=3)
    assert res[0][0].chunk_id == "law"
    assert idx.search("zzzzz qqqqq", k=3) == []


def test_hashing_embedder_is_deterministic_normalized_and_semantic_ish():
    e = HashingEmbedder(dim=128)
    a = e.embed(["governing law new york"])[0]
    assert a == e.embed(["governing law new york"])[0]                  # deterministic
    assert abs(math.sqrt(sum(x * x for x in a)) - 1.0) < 1e-6           # L2-normalized
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    base = e.embed(["monthly payment of fees"])[0]
    near = e.embed(["payment of fees each month"])[0]
    far = e.embed(["governing law jurisdiction"])[0]
    assert dot(base, near) > dot(base, far)                            # overlap ⇒ closer


def test_dense_index_retrieves_by_vector_similarity():
    idx = DenseIndex(HashingEmbedder(dim=128))
    idx.add(_DOCS)
    res = idx.search("fees invoice payment", k=1)
    assert res[0][0].chunk_id == "pay"


def test_hybrid_fuses_bm25_and_dense_and_returns_chunks():
    idx = build_index(_DOCS, embedder=HashingEmbedder(dim=128))
    res = idx.search("terminate the agreement on notice", k=2)
    assert res[0].chunk_id == "term"
    assert all(isinstance(c, Chunk) for c in res)


def test_hybrid_abac_filter_excludes_unpermitted_chunks():
    docs = [_c("fin", "fees of $5,000 payable monthly", tags=["finance"]),
            _c("gen", "general recitals of the parties", tags=["general"])]
    idx = build_index(docs, embedder=HashingEmbedder(dim=128))
    res = idx.search("fees payment", k=5, allowed_tags=["finance"])
    assert [c.chunk_id for c in res] == ["fin"]            # 'gen' filtered out by ABAC
