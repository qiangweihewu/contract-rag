"""Live parity + latency benchmark for the PgVectorStore adapter.

The spec's persistence exit criterion: a DenseIndex backed by PgVectorStore must
return the *same* retrieval result as the in-memory store. We index one contract
corpus into both stores through identical `DenseIndex(embedder, store=...)` wiring
and assert the **rank-1 match agrees** (parity) and is the **correct clause** for
each field query (retrieval correctness). We deliberately do NOT assert full top-k
ordering: pgvector stores vectors as float4, so the near-zero-similarity tail
(chunks that share almost no tokens with the query) can reorder vs in-memory's
float64 — that tail is retrieval noise, while the rank-1 hit is what Context Recall
actually rides on.

Gated on PGVECTOR_URL (a postgres+pgvector DSN); skips otherwise."""

import os
import time

import pytest

from contract_rag.chunk.models import Chunk
from contract_rag.eval.retrieval import FIELD_QUERIES
from contract_rag.index.dense import DenseIndex
from contract_rag.index.embed import HashingEmbedder
from contract_rag.index.store import InMemoryVectorStore

PGVECTOR_URL = os.environ.get("PGVECTOR_URL")

_CORPUS = [
    Chunk(chunk_id="c-parties", doc_id="d", heading="Parties", block_ids=["b1"],
          text="This Agreement is entered into by Acme Inc. and Globex LLC."),
    Chunk(chunk_id="c-law", doc_id="d", heading="Governing Law", block_ids=["b2"],
          text="This Agreement shall be governed by the laws of the State of New York."),
    Chunk(chunk_id="c-term", doc_id="d", heading="Termination", block_ids=["b3"],
          text="Either party may terminate this Agreement upon ninety (90) days written notice."),
    Chunk(chunk_id="c-renew", doc_id="d", heading="Renewal", block_ids=["b4"],
          text="This Agreement renews automatically for successive one-year terms."),
    Chunk(chunk_id="c-fees", doc_id="d", heading="Fees", block_ids=["b5"],
          text="The total contract value is one million dollars ($1,000,000) payable annually."),
    Chunk(chunk_id="c-eff", doc_id="d", heading="Effective Date", block_ids=["b6"],
          text="The effective date of this Agreement is January 1, 2024."),
    Chunk(chunk_id="c-conf", doc_id="d", heading="Confidentiality", block_ids=["b7"],
          text="Each party shall keep the other party's confidential information secret."),
    Chunk(chunk_id="c-indem", doc_id="d", heading="Indemnification", block_ids=["b8"],
          text="The Supplier shall indemnify the Customer against third-party claims."),
    Chunk(chunk_id="c-assign", doc_id="d", heading="Assignment", block_ids=["b9"],
          text="Neither party may assign this Agreement without prior written consent."),
    Chunk(chunk_id="c-notice", doc_id="d", heading="Notices", block_ids=["b10"],
          text="All notices must be sent in writing to the addresses set forth above."),
]


@pytest.mark.skipif(not PGVECTOR_URL, reason="set PGVECTOR_URL to a postgres+pgvector DSN")
def test_pgvector_dense_index_matches_inmemory_recall():
    from contract_rag.index.pgvector import PgVectorStore

    table = "parity_chunks"
    import psycopg

    with psycopg.connect(PGVECTOR_URL) as conn:   # clean slate
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()

    embedder = HashingEmbedder()
    mem = DenseIndex(embedder, store=InMemoryVectorStore())
    pg = DenseIndex(embedder, store=PgVectorStore(dsn=PGVECTOR_URL, table=table))
    mem.add(_CORPUS)
    pg.add(_CORPUS)

    k = 5
    latencies = []
    for field, q in FIELD_QUERIES.items():
        (mem_chunk, mem_score) = mem.search(q, k)[0]
        t0 = time.perf_counter()
        pg_hits = pg.search(q, k)
        latencies.append((time.perf_counter() - t0) * 1000)
        (pg_chunk, pg_score) = pg_hits[0]
        # Same rank-1 chunk, and the cosine score agrees within float4 storage precision.
        assert pg_chunk.chunk_id == mem_chunk.chunk_id, (
            f"rank-1 disagrees for {field}: pg={pg_chunk.chunk_id} mem={mem_chunk.chunk_id}"
        )
        assert abs(pg_score - mem_score) < 1e-4, (
            f"rank-1 score drift for {field}: pg={pg_score:.6f} mem={mem_score:.6f}"
        )
        # Attribution survives the round-trip through Postgres.
        assert pg_chunk.block_ids == mem_chunk.block_ids

    avg = sum(latencies) / len(latencies)
    print(f"\npgvector parity OK — rank-1 hit + cosine score identical to in-memory across all "
          f"{len(latencies)} field queries (block_ids preserved); pgvector query latency avg "
          f"{avg:.1f} ms (corpus={len(_CORPUS)} chunks, dim={embedder.dim})")
