# tests/test_secure_search.py
import pytest

from contract_rag.chunk.models import Chunk
from contract_rag.security.abac import Principal
from contract_rag.security.search import search_as


def _chunk(cid, tags):
    return Chunk(chunk_id=cid, doc_id="d", text="t", block_ids=["b"], permission_tags=tags)


class _FakeIndex:
    """Records the allowed_tags it was called with and returns canned hits."""
    def __init__(self, hits, *, ignore_filter=False):
        self._hits = hits
        self._ignore_filter = ignore_filter
        self.seen_allowed = None

    def search(self, query, k=5, allowed_tags=None, **kwargs):
        self.seen_allowed = allowed_tags
        if self._ignore_filter:
            return self._hits[:k]
        allowed = set(allowed_tags or [])
        return [c for c in self._hits if set(c.permission_tags) & allowed][:k]


def test_search_as_passes_principal_tags_to_index():
    index = _FakeIndex([_chunk("a", ["general"]), _chunk("b", ["restricted"])])
    out = search_as(index, "q", Principal(subject="u", roles=["viewer"]), k=5)
    assert index.seen_allowed == ["general"]
    assert [c.chunk_id for c in out] == ["a"]


def test_search_as_raises_if_index_leaks_forbidden_chunk():
    # A buggy index that ignores the filter must still be caught by the guard.
    leaky = _FakeIndex([_chunk("secret", ["restricted"])], ignore_filter=True)
    with pytest.raises(PermissionError):
        search_as(leaky, "q", Principal(subject="u", roles=["viewer"]), k=5)
