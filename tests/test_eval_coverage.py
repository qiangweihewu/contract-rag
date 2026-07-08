from __future__ import annotations

import pytest

from contract_rag.eval.coverage import (
    FinPage,
    ZonePage,
    _pearson,
    summarize_fin,
    summarize_gedi,
    zone_density,
)


def test_pearson_perfect_and_degenerate():
    assert _pearson([1, 2, 3], [2, 4, 6]) == 1.0
    assert _pearson([1, 2, 3], [6, 4, 2]) == -1.0
    assert _pearson([1, 1, 1], [1, 2, 3]) is None  # zero variance
    assert _pearson([1.0], [2.0]) is None           # too few points


def test_summarize_fin_splits_omitted_vs_clean():
    pages = [
        FinPage(page_id=0, uncovered_ink_ratio=0.10, ink_coverage=0.90, n_facts=5, n_omitted=2),
        FinPage(page_id=1, uncovered_ink_ratio=0.02, ink_coverage=0.98, n_facts=5, n_omitted=0),
        FinPage(page_id=2, uncovered_ink_ratio=0.03, ink_coverage=0.97, n_facts=4, n_omitted=0),
    ]
    s = summarize_fin(pages)
    assert s.n_pages == 3
    assert s.n_pages_with_omission == 1
    assert s.mean_uncovered_omitted_pages == 0.10
    assert s.mean_uncovered_clean_pages == 0.025  # (0.02 + 0.03) / 2
    # more uncovered ink co-occurs with more omission → positive correlation
    assert s.pointbiserial_uncovered_vs_has_omission > 0


def test_summarize_fin_empty_raises():
    with pytest.raises(ValueError):
        summarize_fin([])


def test_summarize_gedi_ratio_and_higher_count():
    pages = [
        ZonePage(name="a", density_in_zone=0.08, density_elsewhere=0.01, n_zone_px=100),
        ZonePage(name="b", density_in_zone=0.04, density_elsewhere=0.02, n_zone_px=80),
        ZonePage(name="c", density_in_zone=0.01, density_elsewhere=0.02, n_zone_px=90),
    ]
    s = summarize_gedi(pages)
    assert s.n_pages == 3
    assert s.mean_density_in_zone == pytest.approx((0.08 + 0.04 + 0.01) / 3, abs=1e-4)
    assert s.n_pages_in_higher == 2  # a and b, not c
    assert s.ratio_in_over_out > 1


def test_summarize_gedi_empty_raises():
    with pytest.raises(ValueError):
        summarize_gedi([])


def test_zone_density_pure_numpy():
    np = pytest.importorskip("numpy")
    uncovered = np.zeros((10, 10), dtype=bool)
    uncovered[0:5, 0:5] = True  # 25 uncovered px, all in the top-left quadrant
    zone = np.zeros((10, 10), dtype=bool)
    zone[0:5, 0:5] = True       # zone == the uncovered region
    din, dout, nz = zone_density(uncovered, zone)
    assert din == 1.0    # every zone pixel is uncovered ink
    assert dout == 0.0   # nothing uncovered outside
    assert nz == 25
