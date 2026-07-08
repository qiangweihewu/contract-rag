# tests/test_content_drift.py
from __future__ import annotations

from pathlib import Path

import pytest

from contract_rag.site.builder import build_site
from contract_rag.benchmark.core import run_nda_benchmark

_CONTENT = Path(__file__).resolve().parents[1] / "content"


def test_articles_carry_the_required_invariants():
    for name in ("benchmark.en.md", "benchmark.zh.md"):
        text = (_CONTENT / name).read_text()
        assert "python -m contract_rag.benchmark" in text          # reproduce command
        assert "{{ f1_lift }}" in text or "{{ f1_clean }}" in text  # numbers from the benchmark
        if name.endswith(".en.md"):
            assert "simulated" in text.lower()                     # honesty caveat present (en)
        else:
            assert "模拟" in text                                   # honesty caveat present (zh)


def test_build_leaves_no_dangling_tokens(tmp_path):
    pytest.importorskip("markdown")
    written = build_site(_CONTENT, tmp_path, base_url="https://x.github.io/contract-rag",
                         benchmark=run_nda_benchmark(seed=0))
    for p in written:
        if p.suffix == ".html":
            assert "{{" not in p.read_text(), f"unsubstituted token in {p.name}"
