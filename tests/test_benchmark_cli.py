from __future__ import annotations

import json

from contract_rag.benchmark.__main__ import format_table, write_results
from contract_rag.benchmark.core import run_nda_benchmark


def test_format_table_reports_lift():
    r = run_nda_benchmark(seed=0)
    t = format_table(r)
    assert "field_f1" in t and "quality_score" in t
    assert "synthetic-nda" in t


def test_write_results_json(tmp_path):
    r = run_nda_benchmark(seed=0)
    out = write_results(r, tmp_path)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["n_docs"] == 8
    assert set(data) >= {"f1_dirty", "f1_clean", "quality_dirty_mean", "quality_clean_mean", "per_doc"}
