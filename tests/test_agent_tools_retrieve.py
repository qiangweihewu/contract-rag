from contract_rag.agent.tools import RetrieveTool
from contract_rag.chunk.models import Chunk


def _chunk(cid, tags=None):
    return Chunk(chunk_id=cid, doc_id="d", text=f"text {cid}", block_ids=[cid],
                 permission_tags=tags or ["general"])


class _FakeIndex:
    def __init__(self, hits):
        self._hits = hits
        self.seen = None

    def search(self, query, k=5, allowed_tags=None, **kw):
        self.seen = {"query": query, "k": k, "allowed_tags": allowed_tags}
        return self._hits[:k]


class _FilteringFakeIndex:
    def __init__(self, hits):
        self._hits = hits
        self.seen = None

    def search(self, query, k=5, allowed_tags=None, **kw):
        self.seen = {"query": query, "k": k, "allowed_tags": allowed_tags}
        if allowed_tags is None:
            return self._hits[:k]
        allowed = set(allowed_tags)
        return [c for c in self._hits if set(c.permission_tags) & allowed][:k]


def test_retrieve_tool_calls_index_and_returns_chunks():
    index = _FakeIndex([_chunk("a"), _chunk("b")])
    out = RetrieveTool(index).run({"query": "governing law", "k": 2})
    assert index.seen["query"] == "governing law" and index.seen["k"] == 2
    assert [c["chunk_id"] for c in out["chunks"]] == ["a", "b"]


def test_retrieve_tool_uses_principal_when_given():
    from contract_rag.security.abac import Principal
    index = _FilteringFakeIndex([_chunk("a", ["general"]), _chunk("secret", ["restricted"])])
    out = RetrieveTool(index, principal=Principal(subject="u", roles=["viewer"])).run({"query": "q"})
    # search_as resolves viewer -> ["general"], so the fake index filters to "a"
    assert index.seen["allowed_tags"] == ["general"]
    assert [c["chunk_id"] for c in out["chunks"]] == ["a"]
