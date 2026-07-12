"""Ensemble(rule+openai) end-to-end measurement on the 40-doc CUAD golden set.

Runs the rule baseline first (cached IRs, fast) to collect paired per-doc rows,
then the ensemble backend with per-field routing to the measured winner
(counterparty/governing_law/termination_notice_days/auto_renewal -> openai;
effective_date/total_value -> rule), then reports:
  - both aggregates + per-field-on-labeled
  - ensemble field-F1 bootstrap CI
  - paired sign-flip permutation test rule vs ensemble (first real p-value)
  - fallback + reattribution counts from the ensemble
Rows are dumped to JSON so future pairings (e.g. vs a constrained GPU re-run)
don't need to re-spend API calls.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from contract_rag.baseline import format_report, run_baseline
from contract_rag.config import assert_backend_allowed, get_settings
from contract_rag.eval.ir_cache import ir_cache
from contract_rag.eval.stats import (
    bootstrap_metric_ci,
    field_f1_of,
    paired_permutation_test,
)
from contract_rag.extract.extractor import get_extractor
from contract_rag.parse.docling_parser import parse_with_docling
from contract_rag.verticals.registry import get_vertical_for

OUT = Path("measure_ensemble_rows.json")


class CountingExtractor:
    def __init__(self, inner):
        self._inner = inner
        self.n_docs = 0
        self.doc_seconds: list[float] = []
        self.fallback_totals: dict[str, int] = {}
        self.reattr_totals: dict[str, int] = {}

    def extract(self, ir):
        t0 = time.time()
        facts = self._inner.extract(ir)
        self.doc_seconds.append(time.time() - t0)
        self.n_docs += 1
        for key, n in (getattr(self._inner, "last_fallbacks", {}) or {}).items():
            self.fallback_totals[key] = self.fallback_totals.get(key, 0) + n
        for key, n in (getattr(self._inner, "last_reattributions", {}) or {}).items():
            self.reattr_totals[key] = self.reattr_totals.get(key, 0) + n
        print(f"  doc {self.n_docs}: {self.doc_seconds[-1]:.1f}s", flush=True)
        return facts


def run_one(backend: str) -> tuple[list[dict], dict, CountingExtractor]:
    os.environ["EXTRACT_BACKEND"] = backend
    settings = get_settings()
    assert_backend_allowed(settings)
    vertical = get_vertical_for(settings)
    extractor = CountingExtractor(get_extractor(settings, vertical))
    parse_fn = ir_cache(Path(".ir_cache"), parse_with_docling)
    rows: list[dict] = []
    agg = run_baseline(settings, extractor, parse_fn, vertical=vertical, collect_rows=rows)
    print(f"\n=== {backend} ===")
    print(format_report(agg))
    print("per-field on-labeled:")
    for name, acc in agg["per_field_on_labeled"].items():
        shown = "n/a (0 gold)" if acc is None else f"{acc:.3f}"
        print(f"  {name:<26} {shown}  (gold n={agg['support'][name]})")
    print("error taxonomy totals:", json.dumps(agg["error_taxonomy"]["totals"]))
    return rows, agg, extractor


def main() -> None:
    print("model:", os.environ.get("OPENAI_MODEL"), "| routing:",
          os.environ.get("ENSEMBLE_ROUTING") or "(DEFAULT_ROUTING)", flush=True)

    rows_rule, agg_rule, _ = run_one("rule")
    rows_ens, agg_ens, ens = run_one("ensemble")

    ci = bootstrap_metric_ci(rows_ens, field_f1_of)
    print(f"\nensemble field_f1 CI95: point={ci['point']:.3f} "
          f"[{ci['lo']:.3f}, {ci['hi']:.3f}]")
    perm = paired_permutation_test(rows_rule, rows_ens, field_f1_of)
    print(f"paired permutation rule-vs-ensemble (field_f1): "
          f"diff={perm['observed_diff']:.3f} p={perm['p_value']:.4f}")
    print(f"ensemble fallbacks: {ens.fallback_totals}")
    print(f"ensemble reattributions: {ens.reattr_totals}")
    secs = ens.doc_seconds
    if secs:
        print(f"latency/doc: mean {sum(secs)/len(secs):.1f}s "
              f"min {min(secs):.1f}s max {max(secs):.1f}s")

    OUT.write_text(json.dumps({
        "rows_rule": rows_rule, "rows_ensemble": rows_ens,
        "agg_rule": agg_rule, "agg_ensemble": agg_ens,
        "ci_ensemble_f1": ci, "perm_rule_vs_ensemble": perm,
        "fallbacks": ens.fallback_totals, "reattributions": ens.reattr_totals,
        "model": os.environ.get("OPENAI_MODEL"),
        "routing": os.environ.get("ENSEMBLE_ROUTING"),
    }, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print("DONE_MARKER")


if __name__ == "__main__":
    main()
