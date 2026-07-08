from contract_rag.chunk.models import Chunk
from contract_rag.index.dense import DenseIndex
from contract_rag.index.embed import HashingEmbedder
from contract_rag.index.hybrid import build_index
from contract_rag.index.pgvector import PgVectorStore, row_to_chunk, vec_literal
from contract_rag.index.rerank import LexicalReranker


def _c(cid, text, tags=None):
    return Chunk(chunk_id=cid, doc_id="d", text=text, block_ids=[cid], permission_tags=tags or [])


# ---- pgvector adapter (fake connection — no DB) ----

class _Cur:
    def __init__(self, rows):
        self.rows, self.sql = rows, []

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): self.sql.append((sql, params))
    def fetchall(self): return self.rows


class _Conn:
    def __init__(self, rows=()):
        self.cur = _Cur(list(rows))
        self.commits = 0

    def cursor(self): return self.cur
    def commit(self): self.commits += 1


def test_vec_literal_and_row_mapping():
    assert vec_literal([0.5, -1.0]) == "[0.50000000,-1.00000000]"
    c = row_to_chunk(("id", "d", "txt", ["b1"], "Head", "payment", ["finance"], 0.9))
    assert c.chunk_id == "id" and c.block_ids == ["b1"] and c.permission_tags == ["finance"]


def test_pgvector_store_add_inserts_and_query_maps_rows():
    row = ("c1", "d", "governing law text", ["b1"], "Law", "governing_law", ["legal"], 0.88)
    conn = _Conn(rows=[row])
    store = PgVectorStore(conn=conn)

    store.add([_c("c1", "governing law text")], [[0.1, 0.2, 0.3]])
    assert conn.commits >= 1
    assert any("INSERT INTO chunks" in sql for sql, _ in conn.cur.sql)
    assert any("vector(3)" in sql for sql, _ in conn.cur.sql)        # schema created at inferred dim

    out = store.query([0.1, 0.2, 0.3], k=5)
    assert any("ORDER BY embedding <=>" in sql for sql, _ in conn.cur.sql)
    assert out == [(out[0][0], 0.88)] and out[0][0].chunk_id == "c1"


def test_query_commits_to_release_read_transaction():
    # query() must not leave the connection idle-in-transaction: a lingering read
    # holds an ACCESS SHARE lock that blocks DDL (e.g. DROP/VACUUM) on the table.
    conn = _Conn(rows=[("c1", "d", "t", ["b1"], None, None, [], 0.5)])
    store = PgVectorStore(conn=conn)
    store.add([_c("c1", "t")], [[0.1, 0.2, 0.3]])
    before = conn.commits

    store.query([0.1, 0.2, 0.3], k=1)

    assert conn.commits == before + 1   # the read transaction is closed out


def test_dense_index_works_through_pluggable_store():
    idx = DenseIndex(HashingEmbedder(dim=64), store=PgVectorStore(conn=_Conn(rows=[
        ("c1", "d", "fees and payment", ["b1"], None, None, [], 0.7)])))
    idx.add([_c("c1", "fees and payment")])
    res = idx.search("payment", k=3)
    assert res[0][0].chunk_id == "c1"


# ---- reranker ----

def test_lexical_reranker_orders_by_query_overlap():
    chunks = [_c("a", "unrelated boilerplate text"), _c("b", "termination notice period clause")]
    out = LexicalReranker().rerank("termination notice", chunks)
    assert out[0].chunk_id == "b"


def test_lexical_reranker_scores_injected_index_extra_text():
    # "confidential" only appears in `a`'s injected index_extra (definition-injection
    # text), never in its display `text` or heading; `b` has no overlap at all. If the
    # reranker scored `f"{heading} {text}"` (the old behavior) it would rank `a` and `b`
    # identically (both zero overlap) and `a` could be demoted below `b` by search-order
    # tie-breaking. Scoring `index_text()` must surface `a` first.
    a = Chunk(chunk_id="a", doc_id="d", text="the parties shall cooperate in good faith",
              block_ids=["a"],
              index_extra='[DEFINITIONS: "Confidential Information" means secret data.]')
    b = Chunk(chunk_id="b", doc_id="d", text="notices shall be sent by certified mail",
              block_ids=["b"])
    out = LexicalReranker().rerank("confidential information", [b, a])
    assert out[0].chunk_id == "a"                  # not demoted below a chunk with zero overlap


def test_hybrid_search_applies_reranker_to_pick_top_k():
    docs = [_c("law", "governed by the laws of New York"),
            _c("pay", "payment of fees within thirty days"),
            _c("term", "either party may terminate on notice")]
    idx = build_index(docs, embedder=HashingEmbedder(dim=64))

    class _Pin:  # reranker that forces 'pay' to the top
        name = "pin"
        def rerank(self, query, chunks):
            return sorted(chunks, key=lambda c: c.chunk_id != "pay")

    res = idx.search("anything", k=1, reranker=_Pin())
    assert res[0].chunk_id == "pay"
