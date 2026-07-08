"""Matplotlib charts for the benchmark. matplotlib is imported lazily (extra
`benchmark`) so the core + unit suite run without it."""
from __future__ import annotations

from pathlib import Path

from contract_rag.benchmark.core import BenchmarkResult


def write_charts(result: BenchmarkResult, out_dir: Path) -> list[Path]:
    """Write before/after bar charts (quality + field-F1). Requires matplotlib."""
    import matplotlib
    matplotlib.use("Agg")  # headless, no display needed
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _bar(path: Path, title: str, dirty: float, clean: float) -> None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["dirty", "cleaned"], [dirty, clean], color=["#c0392b", "#27ae60"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        for i, v in enumerate([dirty, clean]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    _bar(out_dir / "quality.png", "Data-quality score", result.quality_dirty_mean, result.quality_clean_mean)
    _bar(out_dir / "field_f1.png", "Extraction field-F1", result.f1_dirty, result.f1_clean)
    return written
