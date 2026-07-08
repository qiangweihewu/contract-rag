"""pgvector-backed VectorStore — the spec's persistent index. Implements the same
add()/query() surface as InMemoryVectorStore, so DenseIndex/HybridIndex use it unchanged.

`psycopg` is imported lazily (optional `pgvector` extra); a connection can be injected for
testing without a database. Cosine distance via pgvector's `<=>`."""
from __future__ import annotations

from contract_rag.chunk.models import Chunk

_COLS = ("chunk_id", "doc_id", "text", "block_ids", "heading", "clause_type", "permission_tags")


def vec_literal(vector: list[float]) -> str:
    """pgvector text format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


def row_to_chunk(row: tuple) -> Chunk:
    return Chunk(chunk_id=row[0], doc_id=row[1], text=row[2], block_ids=list(row[3] or []),
                 heading=row[4], clause_type=row[5], permission_tags=list(row[6] or []))


class PgVectorStore:
    def __init__(self, dsn: str | None = None, conn=None, table: str = "chunks"):
        self._dsn = dsn
        self._conn = conn          # inject for tests; else lazily connected from dsn
        self.table = table
        self._dim: int | None = None

    def _connection(self):
        if self._conn is None:
            import psycopg

            self._conn = psycopg.connect(self._dsn)
        return self._conn

    def _ensure_schema(self, dim: int) -> None:
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                "chunk_id text PRIMARY KEY, doc_id text, text text, block_ids text[], "
                f"heading text, clause_type text, permission_tags text[], embedding vector({dim}))"
            )
        conn.commit()

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if self._dim is None:
            self._dim = len(vectors[0])
            self._ensure_schema(self._dim)
        conn = self._connection()
        with conn.cursor() as cur:
            for c, v in zip(chunks, vectors):
                cur.execute(
                    f"INSERT INTO {self.table} ({', '.join(_COLS)}, embedding) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                    (c.chunk_id, c.doc_id, c.text, c.block_ids, c.heading,
                     c.clause_type, c.permission_tags, vec_literal(v)),
                )
        conn.commit()

    def query(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        conn = self._connection()
        lit = vec_literal(vector)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(_COLS)}, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (lit, lit, k),
            )
            rows = cur.fetchall()
        conn.commit()   # close the read txn; don't sit idle-in-transaction holding locks
        return [(row_to_chunk(r), float(r[7])) for r in rows]
