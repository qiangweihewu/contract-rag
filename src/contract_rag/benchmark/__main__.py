"""`python -m contract_rag.benchmark` — run the committed-corpus benchmark,
print the table, and write results.json (+ charts if matplotlib is present)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from contract_rag.benchmark.core import BenchmarkResult, run_nda_benchmark


def format_table(result: BenchmarkResult) -> str:
    return "\n".join([
        f"=== Cleaning benchmark ({result.corpus}, seed={result.seed}, n={result.n_docs}) ===",
        f"field_f1:      dirty={result.f1_dirty:.3f}  cleaned={result.f1_clean:.3f}  "
        f"lift={result.f1_lift:+.3f}",
        f"quality_score: dirty={result.quality_dirty_mean:.3f}  cleaned={result.quality_clean_mean:.3f}  "
        f"lift={result.quality_lift:+.3f}",
        f"source_accuracy (cleaned): {result.source_accuracy:.3f}",
    ])


def write_results(result: BenchmarkResult, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    return path


def main() -> None:
    out_dir = Path(os.environ.get("BENCHMARK_OUT", "benchmark_out"))
    result = run_nda_benchmark(seed=int(os.environ.get("BENCHMARK_SEED", "0")))
    print(format_table(result))
    print(f"wrote {write_results(result, out_dir)}")
    try:
        from contract_rag.benchmark.plots import write_charts
        for p in write_charts(result, out_dir / "charts"):
            print(f"wrote {p}")
    except ImportError:
        print("matplotlib not installed (extra `benchmark`) — skipped charts")


if __name__ == "__main__":
    main()
