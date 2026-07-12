"""GPU-rig measurements against a live Ollama endpoint (qwen2.5:32b-instruct):

1. `constrained` backend, per-doc rows collected -> aggregate + CI + repair counts.
2. `ensemble` with DEFAULT_ROUTING (constrained child), rows collected
   -> aggregate + fallback/reattribution counts.
3. Statistics, pairing against the saved rows in measure_ensemble_rows.json
   (same 40-doc golden order — run_baseline iterates load_golden_set
   deterministically):
   - paired permutation rule vs constrained  (the pairing the stats layer waited on)
   - paired permutation rule vs ensemble(constrained)
   - paired permutation ensemble(openai) vs ensemble(constrained)
   - bootstrap CIs for both new runs

Env expected: CONSTRAINED_ENDPOINT=http://localhost:11434/v1,
CONSTRAINED_MODEL=qwen2.5:32b-instruct (EXTRACT_BACKEND is set per-run here).
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

SAVED = Path("measure_ensemble_rows.json")
OUT = Path("measure_gpu_rows.json")


class CountingExtractor:
    def __init__(self, inner):
        self._inner = inner
        self.n_docs = 0
        self.doc_seconds: list[float] = []
        self.counters: dict[str, dict[str, int]] = {}

    def _bump(self, bucket: str, d: dict) -> None:
        tot = self.counters.setdefault(bucket, {})
        for key, n in (d or {}).items():
            tot[key] = tot.get(key, 0) + n

    def extract(self, ir):
        t0 = time.time()
        facts = self._inner.extract(ir)
        self.doc_seconds.append(time.time() - t0)
        self.n_docs += 1
        self._bump("repairs", getattr(self._inner, "last_repairs", {}))
        self._bump("fallbacks", getattr(self._inner, "last_fallbacks", {}))
        self._bump("reattributions", getattr(self._inner, "last_reattributions", {}))
        print(f"  doc {self.n_docs}: {self.doc_seconds[-1]:.1f}s", flush=True)
        return facts


def run_one(backend: str):
    os.environ["EXTRACT_BACKEND"] = backend
    settings = get_settings()
    assert_backend_allowed(settings)
    vertical = get_vertical_for(settings)
    extractor = CountingExtractor(get_extractor(settings, vertical))
    parse_fn = ir_cache(Path(".ir_cache"), parse_with_docling)
    rows: list[dict] = []
    t0 = time.time()
    agg = run_baseline(settings, extractor, parse_fn, vertical=vertical, collect_rows=rows)
    print(f"\n=== {backend} ({time.time()-t0:.0f}s) ===")
    print(format_report(agg))
    print("per-field on-labeled:")
    for name, acc in agg["per_field_on_labeled"].items():
        shown = "n/a (0 gold)" if acc is None else f"{acc:.3f}"
        print(f"  {name:<26} {shown}  (gold n={agg['support'][name]})")
    print("error taxonomy totals:", json.dumps(agg["error_taxonomy"]["totals"]))
    print("counters:", json.dumps({k: v for k, v in extractor.counters.items() if v}))
    return rows, agg, extractor


def stat_pair(label: str, rows_a: list[dict], rows_b: list[dict]) -> dict:
    perm = paired_permutation_test(rows_a, rows_b, field_f1_of)
    print(f"{label}: diff={perm['observed_diff']:.3f} p={perm['p_value']:.4f}")
    return perm


def main() -> None:
    saved = json.loads(SAVED.read_text())
    rows_rule = saved["rows_rule"]
    rows_ens_openai = saved["rows_ensemble"]
    print("endpoint:", os.environ.get("CONSTRAINED_ENDPOINT"),
          "| model:", os.environ.get("CONSTRAINED_MODEL"), flush=True)

    rows_con, agg_con, ex_con = run_one("constrained")
    os.environ.pop("ENSEMBLE_ROUTING", None)  # DEFAULT_ROUTING -> constrained child
    rows_ens_con, agg_ens_con, ex_ens = run_one("ensemble")

    print()
    ci_con = bootstrap_metric_ci(rows_con, field_f1_of)
    print(f"constrained field_f1 CI95: {ci_con['point']:.3f} [{ci_con['lo']:.3f}, {ci_con['hi']:.3f}]")
    ci_ens = bootstrap_metric_ci(rows_ens_con, field_f1_of)
    print(f"ensemble(constrained) field_f1 CI95: {ci_ens['point']:.3f} [{ci_ens['lo']:.3f}, {ci_ens['hi']:.3f}]")
    perms = {
        "rule_vs_constrained": stat_pair("perm rule vs constrained", rows_rule, rows_con),
        "rule_vs_ensemble_constrained": stat_pair(
            "perm rule vs ensemble(constrained)", rows_rule, rows_ens_con),
        "ensemble_openai_vs_constrained_child": stat_pair(
            "perm ensemble(openai) vs ensemble(constrained)", rows_ens_openai, rows_ens_con),
    }
    for label, ex in (("constrained", ex_con), ("ensemble", ex_ens)):
        secs = ex.doc_seconds
        print(f"{label} latency/doc: mean {sum(secs)/len(secs):.1f}s "
              f"min {min(secs):.1f}s max {max(secs):.1f}s")

    OUT.write_text(json.dumps({
        "rows_constrained": rows_con, "agg_constrained": agg_con,
        "rows_ensemble_constrained": rows_ens_con, "agg_ensemble_constrained": agg_ens_con,
        "ci_constrained": ci_con, "ci_ensemble_constrained": ci_ens,
        "perms": perms,
        "counters_constrained": ex_con.counters, "counters_ensemble": ex_ens.counters,
        "model": os.environ.get("CONSTRAINED_MODEL"),
    }, indent=2, default=str))
    print(f"\nwrote {OUT}")
    print("DONE_MARKER")


if __name__ == "__main__":
    main()
