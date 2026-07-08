"""`python -m contract_rag.site build` — run the benchmark, then build the site."""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

from contract_rag.benchmark.core import run_nda_benchmark
from contract_rag.site.builder import build_site


def build(content_dir, out_dir, *, base_url: str, seed: int = 0,
          now: str | None = None) -> list[Path]:
    result = run_nda_benchmark(seed=seed)
    # Write charts to benchmark_out/charts/ so build_site can copy them into out_dir/charts/
    # without a self-copy collision (which happens when charts_dir is inside out_dir).
    charts_dir: Path | None = Path("benchmark_out") / "charts"
    try:
        from contract_rag.benchmark.plots import write_charts
        write_charts(result, charts_dir)
    except ImportError:
        charts_dir = None  # charts optional; article still builds
    return build_site(content_dir, out_dir, base_url=base_url,
                      benchmark=result, charts_dir=charts_dir, now=now)


def main() -> None:
    base_url = os.environ.get("SITE_BASE_URL", "https://qiangweihewu.github.io/contract-rag")
    written = build(Path("content"), Path(os.environ.get("SITE_OUT", "site_out")),
                    base_url=base_url, seed=int(os.environ.get("BENCHMARK_SEED", "0")),
                    now=date.today().isoformat())
    print(f"built {len(written)} files to site_out/")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        main()
    else:
        print("usage: python -m contract_rag.site build")
