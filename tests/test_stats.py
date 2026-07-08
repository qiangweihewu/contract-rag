"""Dep-free tests for eval/stats.py: bootstrap CI + paired permutation test, plus the
additive baseline/compare wiring. Rows are hand-built dicts matching metrics.row_for()'s
shape, keyed by the default contract vertical's field names — no golden set, no network."""
from __future__ import annotations

import pytest

from contract_rag.baseline import run_baseline
from contract_rag.compare import compare_parsers, format_comparison
from contract_rag.config import Settings
from contract_rag.eval.stats import (
    bootstrap_metric_ci,
    field_f1_of,
    paired_permutation_test,
    source_accuracy_of,
)
from contract_rag.extract.extractor import FakeExtractor
from contract_rag.extract.schema import ContractFacts, ExtractedClause
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

FIELDS = ContractFacts.FIELD_NAMES


def _row(all_correct: bool) -> dict:
    """A homogeneous row: every field either fully correct+attributed, or an omission
    (gold present, nothing predicted) — the two extremes used to drive the metric."""
    return {
        "scores": {n: all_correct for n in FIELDS},
        "source": {n: all_correct for n in FIELDS},
        "pred_nonempty": {n: all_correct for n in FIELDS},
        "gold_nonempty": dict.fromkeys(FIELDS, True),
    }


def _mixed_rows(n: int) -> list[dict]:
    return [_row(i % 2 == 0) for i in range(n)]


# --- bootstrap_metric_ci -----------------------------------------------------------

def test_bootstrap_ci_deterministic_same_seed():
    rows = _mixed_rows(10)
    ci1 = bootstrap_metric_ci(rows, field_f1_of, n_boot=200, seed=42)
    ci2 = bootstrap_metric_ci(rows, field_f1_of, n_boot=200, seed=42)
    assert ci1 == ci2


def test_bootstrap_ci_different_seed_differs():
    rows = _mixed_rows(10)
    ci1 = bootstrap_metric_ci(rows, field_f1_of, n_boot=200, seed=1)
    ci2 = bootstrap_metric_ci(rows, field_f1_of, n_boot=200, seed=2)
    assert (ci1["lo"], ci1["hi"]) != (ci2["lo"], ci2["hi"])


def test_bootstrap_ci_point_within_bounds():
    rows = _mixed_rows(10)
    ci = bootstrap_metric_ci(rows, field_f1_of, n_boot=200, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_bootstrap_ci_degenerate_all_identical_rows():
    rows = [_row(True) for _ in range(5)]
    ci = bootstrap_metric_ci(rows, field_f1_of, n_boot=100, seed=0)
    assert ci["lo"] == ci["hi"] == ci["point"] == 1.0


def test_bootstrap_ci_empty_rows():
    ci = bootstrap_metric_ci([], field_f1_of, n_boot=100, seed=0)
    assert ci == {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": 100, "confidence": 0.95}


def test_bootstrap_ci_shape_and_source_accuracy_metric():
    rows = _mixed_rows(6)
    ci = bootstrap_metric_ci(rows, source_accuracy_of, n_boot=50, seed=0)
    assert set(ci) == {"point", "lo", "hi", "n_boot", "confidence"}
    assert ci["n_boot"] == 50
    assert ci["confidence"] == 0.95


# --- paired_permutation_test -------------------------------------------------------

def test_paired_permutation_identical_rows_p_is_one():
    rows = _mixed_rows(10)
    result = paired_permutation_test(rows, list(rows), field_f1_of, n_perm=200, seed=0)
    assert result["observed_diff"] == 0.0
    assert result["p_value"] == 1.0


def test_paired_permutation_strongly_different_rows_significant():
    rows_a = [_row(True) for _ in range(20)]
    rows_b = [_row(False) for _ in range(20)]
    result = paired_permutation_test(rows_a, rows_b, field_f1_of, n_perm=2000, seed=0)
    assert result["observed_diff"] == pytest.approx(1.0)
    assert result["p_value"] < 0.01


def test_paired_permutation_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_permutation_test(_mixed_rows(3), _mixed_rows(4), field_f1_of)


def test_paired_permutation_shape():
    rows = _mixed_rows(4)
    result = paired_permutation_test(rows, list(rows), field_f1_of, n_perm=50, seed=0)
    assert set(result) == {"observed_diff", "p_value", "n_perm"}
    assert result["n_perm"] == 50


class _FakeVertical:
    """Minimal vertical for aggregate(): non-contract field_names, no field_risk."""

    field_names = ("secret_term", "expiry_date")


def _fake_vertical_row(all_correct: bool) -> dict:
    names = _FakeVertical.field_names
    return {
        "scores": {n: all_correct for n in names},
        "source": {n: all_correct for n in names},
        "pred_nonempty": {n: all_correct for n in names},
        "gold_nonempty": dict.fromkeys(names, True),
    }


def test_paired_permutation_non_default_vertical_rows():
    # Regression: rows shaped by a NON-default vertical must work when the metric fn
    # binds that vertical (compare.main() once passed unbound field_f1_of, which fell
    # back to the contract vertical and raised KeyError('counterparty') on NDA rows).
    v = _FakeVertical()
    rows_a = [_fake_vertical_row(True) for _ in range(10)]
    rows_b = [_fake_vertical_row(i % 2 == 0) for i in range(10)]
    metric = lambda rs: field_f1_of(rs, v)  # noqa: E731
    result = paired_permutation_test(rows_a, rows_b, metric, n_perm=200, seed=0)
    assert result["observed_diff"] == pytest.approx(1.0 - field_f1_of(rows_b, v))
    assert 0.0 < result["p_value"] <= 1.0


# --- baseline.py wiring: collect_rows -----------------------------------------------

def _stub_ir() -> DocumentIR:
    return DocumentIR(
        doc_id="msa-acme", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH,
                     text="entered into by Acme Inc.",
                     bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                     confidence=1.0, source_engine="docling"),
        ],
        metadata={},
    )


def _settings_with_golden(tmp_path):
    gdir = tmp_path / "golden_set"
    gdir.mkdir()
    (gdir / "msa-acme.json").write_text(
        '{"doc_id":"msa-acme","source_pdf":"msa-acme.pdf",'
        '"facts":{"counterparty":"Acme Inc.","effective_date":"","governing_law":""}}'
    )
    return Settings(golden_set_dir=gdir)


def _canned_facts():
    return ContractFacts(
        counterparty=ExtractedClause(value="Acme Inc.", source_block_id="#/b/1", confidence=0.9),
        effective_date=ExtractedClause(),
        governing_law=ExtractedClause(),
    )


def test_run_baseline_without_collect_rows_unchanged(tmp_path):
    settings = _settings_with_golden(tmp_path)
    agg = run_baseline(settings, extractor=FakeExtractor(_canned_facts()), parse_fn=lambda _p: _stub_ir())
    assert agg["per_field"]["counterparty"] == 1.0


def test_run_baseline_collect_rows_populated(tmp_path):
    settings = _settings_with_golden(tmp_path)
    collected: list = []
    agg = run_baseline(
        settings, extractor=FakeExtractor(_canned_facts()), parse_fn=lambda _p: _stub_ir(),
        collect_rows=collected,
    )
    assert len(collected) == 1
    assert collected[0]["scores"]["counterparty"] is True
    assert agg["n_docs"] == 1


# --- compare.py wiring: stats param + STATS_CI seam ---------------------------------

def _ir(text: str) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///x", file_hash="h", mime_type="application/pdf",
        blocks=[DocBlock(block_id="#/b/1", type=BlockType.PARAGRAPH, text=text,
                         bbox=BoundingBox(page=1, x0=0, y0=0, x1=1, y1=1),
                         confidence=1.0, source_engine="docling")],
        metadata={},
    )


def test_format_comparison_without_stats_byte_identical():
    docling_agg = {"n_docs": 1, "field_f1": 0.5, "source_accuracy": 0.5, "per_field": {"counterparty": 0.5}}
    router_agg = {"n_docs": 1, "field_f1": 0.6, "source_accuracy": 0.6, "per_field": {"counterparty": 0.6}}
    before = format_comparison(docling_agg, router_agg)
    after = format_comparison(docling_agg, router_agg, stats=None)
    assert before == after
    assert "p-value" not in before


def test_format_comparison_with_stats_appends_line():
    docling_agg = {"n_docs": 1, "field_f1": 0.5, "source_accuracy": 0.5, "per_field": {"counterparty": 0.5}}
    router_agg = {"n_docs": 1, "field_f1": 0.6, "source_accuracy": 0.6, "per_field": {"counterparty": 0.6}}
    stats = {"observed_diff": -0.1, "p_value": 0.234, "n_perm": 2000}
    out = format_comparison(docling_agg, router_agg, stats=stats)
    assert "p-value (paired permutation, field_f1): 0.234" in out
    assert "n_perm=2000" in out


def test_compare_parsers_collects_rows(tmp_path):
    settings = _settings_with_golden(tmp_path)
    extractor = FakeExtractor(_canned_facts())
    docling_rows: list = []
    router_rows: list = []
    docling_agg, router_agg = compare_parsers(
        settings, extractor,
        docling_fn=lambda _p: _ir("entered into by Acme Inc."),
        router_fn=lambda _p: _ir("entered into by Acme Inc."),
        collect_docling_rows=docling_rows,
        collect_router_rows=router_rows,
    )
    assert len(docling_rows) == 1
    assert len(router_rows) == 1
    assert docling_agg["n_docs"] == 1
    assert router_agg["n_docs"] == 1
