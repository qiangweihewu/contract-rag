from contract_rag.eval.golden import GoldenDoc
from contract_rag.eval.sampling import golden_stratum_key, stratified_sample, stratify


def _docs(n, label):
    return [GoldenDoc(doc_id=f"{label}{i}", source_pdf=f"{label}{i}.pdf", facts={}) for i in range(n)]


def test_stratify_groups_by_key():
    items = ["apple", "avocado", "banana"]
    groups = stratify(items, key_fn=lambda s: s[0])
    assert groups["a"] == ["apple", "avocado"]
    assert groups["b"] == ["banana"]


def test_stratified_sample_is_proportional_and_deterministic():
    items = _docs(8, "x") + _docs(2, "y")  # 8 of group x, 2 of group y
    key = lambda d: d.doc_id[0]
    sample = stratified_sample(items, key_fn=key, n=5, seed=1)
    assert len(sample) == 5
    groups = stratify(sample, key)
    assert len(groups["x"]) == 4 and len(groups["y"]) == 1   # 80/20 preserved
    # deterministic: same seed -> same ids
    again = stratified_sample(items, key_fn=key, n=5, seed=1)
    assert [d.doc_id for d in sample] == [d.doc_id for d in again]


def test_golden_stratum_key_buckets_on_clause_complexity():
    sparse = GoldenDoc(doc_id="a", source_pdf="a.pdf", facts={"counterparty": "Acme"})
    rich = GoldenDoc(doc_id="b", source_pdf="b.pdf",
                     facts={"counterparty": "Acme", "effective_date": "2020", "governing_law": "NY",
                            "total_value": "$1", "termination_notice_days": "30", "auto_renewal": "yes"})
    assert golden_stratum_key(sparse) != golden_stratum_key(rich)
