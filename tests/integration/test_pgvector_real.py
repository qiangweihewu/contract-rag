import os

import pytest

PGVECTOR_URL = os.environ.get("PGVECTOR_URL")


@pytest.mark.skipif(not PGVECTOR_URL, reason="set PGVECTOR_URL to a postgres+pgvector DSN")
def test_pgvector_roundtrip_against_real_db():
    from contract_rag.chunk.models import Chunk
    from contract_rag.index.pgvector import PgVectorStore

    store = PgVectorStore(dsn=PGVECTOR_URL, table="chunks_test")
    chunks = [
        Chunk(chunk_id="t1", doc_id="d", text="governed by the laws of New York", block_ids=["b1"]),
        Chunk(chunk_id="t2", doc_id="d", text="payment of fees within thirty days", block_ids=["b2"]),
    ]
    store.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    res = store.query([1.0, 0.0, 0.0], k=1)
    assert res and res[0][0].chunk_id == "t1"
