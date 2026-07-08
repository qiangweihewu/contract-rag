"""Gated on the cached FinCriticalED / Tobacco800 runs (the same caches the
`realscan` and `fincritical` harnesses populate). Exercises the ink-coverage
validation end-to-end against real paddle IRs + real omission/signature ground
truth; skips cleanly when a cache is absent. PIL/numpy/pypdfium2 required."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")
pytest.importorskip("pypdfium2")

_HOME = Path.home()
_FIN_CACHE = Path(os.environ.get(
    "FINCRITICAL_CACHE", str(_HOME / ".cache" / "contract-rag" / "fincriticaled-run")))
_FIN_DATA = Path(os.environ.get(
    "FINCRITICAL_DIR", str(_HOME / ".cache" / "contract-rag" / "fincriticaled")))
_RS_CACHE = Path(os.environ.get(
    "REALSCAN_CACHE", str(_HOME / ".cache" / "contract-rag" / "realscan")))
_GT = Path(os.environ.get("REALSCAN_GT_DIR", str(_HOME / ".cache" / "tobacco800" / "groundtruth")))
_IMG = Path(os.environ.get("TOBACCO_IMG_DIR", str(_HOME / ".cache" / "tobacco800" / "images")))


@pytest.mark.skipif(
    not ((_FIN_CACHE / "ir").exists() and (_FIN_DATA / "raw_input.csv").exists()),
    reason="FinCriticalED run cache not present",
)
def test_fincritical_coverage_signal_is_directionally_positive():
    from contract_rag.eval.coverage import run_fincritical_coverage

    s = run_fincritical_coverage(
        _FIN_DATA, _FIN_CACHE / "ir", _FIN_CACHE / "pdf",
        cap=40, dpi=150.0, dilate_px=3.0,
    )
    assert s.n_pages > 0
    # honest, weak-but-real: omitted-fact pages carry at least as much uncovered ink
    if s.mean_uncovered_omitted_pages is not None and s.mean_uncovered_clean_pages is not None:
        assert s.mean_uncovered_omitted_pages >= s.mean_uncovered_clean_pages


@pytest.mark.skipif(
    not ((_RS_CACHE / "ir" / "paddleocr").exists() and _GT.exists() and _IMG.exists()),
    reason="Tobacco800 realscan cache / GEDI groundtruth not present",
)
def test_tobacco_signature_zones_have_more_uncovered_ink():
    from contract_rag.eval.coverage import run_tobacco_coverage

    s = run_tobacco_coverage(
        _RS_CACHE / "ir" / "paddleocr", _RS_CACHE / "pdf", _GT, _IMG, cap=40,
    )
    assert s.n_pages > 0
    # the sharp test: uncovered ink concentrates inside signature/logo zones
    assert s.mean_density_in_zone > s.mean_density_elsewhere
    assert s.n_pages_in_higher / s.n_pages >= 0.6
