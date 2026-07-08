from __future__ import annotations

from contract_rag.benchmark.core import BenchmarkResult, run_nda_benchmark


def test_benchmark_runs_credential_free_on_committed_corpus():
    r = run_nda_benchmark(seed=0)
    assert isinstance(r, BenchmarkResult)
    assert r.corpus == "synthetic-nda"
    assert r.n_docs == 8
    assert len(r.per_doc) == 8


def test_cleaning_lift_is_positive():
    r = run_nda_benchmark(seed=0)
    # dirt hurts, cleaning recovers — on both quality and extraction
    assert r.quality_clean_mean >= r.quality_dirty_mean
    assert r.f1_clean >= r.f1_dirty
    assert r.quality_lift >= 0.0 and r.f1_lift >= 0.0
    # cleaning recovers extraction to a healthy floor
    assert r.f1_clean >= 0.60
    # rule extractor cites verbatim spans -> attribution holds after cleaning
    assert r.source_accuracy == 1.0


def test_seeded_stable():
    a = run_nda_benchmark(seed=0)
    b = run_nda_benchmark(seed=0)
    assert (a.quality_dirty_mean, a.quality_clean_mean, a.f1_dirty, a.f1_clean) == (
        b.quality_dirty_mean, b.quality_clean_mean, b.f1_dirty, b.f1_clean)
