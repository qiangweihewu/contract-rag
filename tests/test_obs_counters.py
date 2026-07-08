from contract_rag.obs.counters import InMemoryCounterStore


def test_incr_and_value():
    c = InMemoryCounterStore()
    c.incr("permission_leaks")
    c.incr("permission_leaks", by=2)
    c.incr("checks", by=5)
    assert c.value("permission_leaks") == 3
    assert c.value("checks") == 5


def test_unseen_key_is_zero():
    assert InMemoryCounterStore().value("never") == 0


def test_snapshot_is_a_plain_dict_copy():
    c = InMemoryCounterStore()
    c.incr("a", by=2)
    snap = c.snapshot()
    assert snap == {"a": 2}
    snap["a"] = 99  # snapshot must be a copy, not the live store
    assert c.value("a") == 2
