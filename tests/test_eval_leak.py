from contract_rag.chunk.models import Chunk
from contract_rag.eval.leak import LeakCase, evaluate_leaks, format_leaks
from contract_rag.obs.counters import InMemoryCounterStore
from contract_rag.security.abac import Principal


def _chunk(cid, tags):
    return Chunk(chunk_id=cid, doc_id="d", text="t", block_ids=["b"], permission_tags=tags)


def test_clean_when_every_chunk_is_permitted():
    case = LeakCase(name="viewer-ok", principal=Principal(subject="u", roles=["viewer"]),
                    retrieved=[_chunk("a", ["general"])])
    res = evaluate_leaks([case])
    assert res["n_leaks"] == 0
    assert res["leak_rate"] == 0.0
    assert res["clean"] is True


def test_detects_and_counts_a_leak():
    case = LeakCase(name="viewer-leak", principal=Principal(subject="u", roles=["viewer"]),
                    retrieved=[_chunk("ok", ["general"]), _chunk("secret", ["restricted"])])
    counter = InMemoryCounterStore()
    res = evaluate_leaks([case], counter=counter)
    assert res["n_leaks"] == 1
    assert res["n_chunks_checked"] == 2
    assert res["leak_rate"] == 0.5
    assert res["clean"] is False
    assert [v.chunk_id for v in res["violations"]] == ["secret"]
    # routed through obs -> countable
    assert counter.value("permission_leaks") == 1
    assert counter.value("permission_checks") == 2


def test_legal_principal_sees_restricted_without_a_leak():
    case = LeakCase(name="legal-ok", principal=Principal(subject="u", roles=["legal"]),
                    retrieved=[_chunk("c", ["restricted"])])
    assert evaluate_leaks([case])["clean"] is True


def test_format_leaks_mentions_count():
    out = format_leaks(evaluate_leaks([]))
    assert "leak" in out.lower()
