from contract_rag.chunk.models import Chunk
from contract_rag.security.guard import Violation, audit_results, permitted


def _chunk(cid, tags):
    return Chunk(chunk_id=cid, doc_id="d", text="t", block_ids=["b"], permission_tags=tags)


def test_permitted_requires_tag_intersection():
    assert permitted(_chunk("c1", ["finance"]), ["finance", "general"]) is True
    assert permitted(_chunk("c2", ["restricted"]), ["general"]) is False
    assert permitted(_chunk("c3", []), ["general"]) is False


def test_audit_flags_only_forbidden_chunks():
    results = [_chunk("ok", ["general"]), _chunk("leak", ["restricted"])]
    violations = audit_results(results, allowed_tags=["general"])
    assert [v.chunk_id for v in violations] == ["leak"]
    assert isinstance(violations[0], Violation)
    assert violations[0].tags == ["restricted"]


def test_audit_clean_when_all_permitted():
    results = [_chunk("a", ["general"]), _chunk("b", ["finance"])]
    assert audit_results(results, allowed_tags=["general", "finance"]) == []
