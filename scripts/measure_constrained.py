"""T3b: 40-doc CUAD measurement of the constrained backend against live Ollama.

Wraps the extractor to accumulate per-doc repair counts (last_repairs) alongside
the standard baseline aggregate. Run:
  EXTRACT_BACKEND=constrained CONSTRAINED_ENDPOINT=... CONSTRAINED_MODEL=... \
    uv run python scripts/measure_constrained.py
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from contract_rag.baseline import format_report, run_baseline
from contract_rag.config import assert_backend_allowed, get_settings
from contract_rag.eval.ir_cache import ir_cache
from contract_rag.extract.extractor import get_extractor
from contract_rag.parse.docling_parser import parse_with_docling
from contract_rag.verticals.registry import get_vertical_for


class RepairCountingExtractor:
    def __init__(self, inner):
        self._inner = inner
        self.repair_totals: Counter[str] = Counter()
        self.docs_with_repairs = 0
        self.n_docs = 0
        self.doc_seconds: list[float] = []

    def extract(self, ir):
        t0 = time.time()
        facts = self._inner.extract(ir)
        self.doc_seconds.append(time.time() - t0)
        self.n_docs += 1
        repairs = getattr(self._inner, "last_repairs", {})
        if repairs:
            self.docs_with_repairs += 1
            self.repair_totals.update(repairs)
        print(f"  doc {self.n_docs}: {self.doc_seconds[-1]:.1f}s repairs={dict(repairs)}",
              flush=True)
        return facts


def main() -> None:
    settings = get_settings()
    assert_backend_allowed(settings)
    vertical = get_vertical_for(settings)
    extractor = RepairCountingExtractor(get_extractor(settings, vertical))
    parse_fn = ir_cache(Path(".ir_cache"), parse_with_docling)
    agg = run_baseline(settings, extractor, parse_fn, vertical=vertical)
    out = {
        "agg": agg,
        "repairs_totals": dict(extractor.repair_totals),
        "docs_with_repairs": extractor.docs_with_repairs,
        "n_docs": extractor.n_docs,
        "doc_seconds": extractor.doc_seconds,
        "backend": settings.extract_backend,
        "model": settings.constrained_model or settings.local_model,
    }
    Path("measure_constrained_result.json").write_text(json.dumps(out, indent=2, default=str))
    print(format_report(agg))
    print("\nper-field on-labeled:")
    for name, acc in agg["per_field_on_labeled"].items():
        shown = "n/a (0 gold)" if acc is None else f"{acc:.3f}"
        print(f"  {name:<26} {shown}  (gold n={agg['support'][name]})")
    print("\nerror taxonomy totals:", json.dumps(agg["error_taxonomy"]["totals"]))
    print("risk tiers:")
    for tier, d in agg["risk_tiers"]["per_tier"].items():
        f1 = d["f1_on_labeled"]
        print(f"  {tier:6s} f1_on_labeled="
              f"{f1 if f1 is None else round(f1, 3)} tax={json.dumps(d['taxonomy'])}")
    print(f"\nrepairs: totals={dict(extractor.repair_totals)} "
          f"docs_with_repairs={extractor.docs_with_repairs}/{extractor.n_docs}")
    secs = extractor.doc_seconds
    print(f"latency: mean={sum(secs)/len(secs):.1f}s "
          f"min={min(secs):.1f}s max={max(secs):.1f}s total={sum(secs)/60:.1f}min")


if __name__ == "__main__":
    main()
