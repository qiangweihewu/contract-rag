from __future__ import annotations

import pytest

from contract_rag.eval.layout_coverage import (
    FinLayoutPage,
    ZoneRegionPage,
    _rect_overlap_frac,
    region_in_zone,
    summarize_fin_layout,
    summarize_gedi_layout,
)
from contract_rag.eval.gedi import Zone


def test_summarize_fin_layout_splits_omitted_vs_clean():
    pages = [
        FinLayoutPage(page_id=0, layout_omission_score=0.30, n_regions=5, n_uncovered=2, n_facts=5, n_omitted=2),
        FinLayoutPage(page_id=1, layout_omission_score=0.02, n_regions=5, n_uncovered=0, n_facts=5, n_omitted=0),
        FinLayoutPage(page_id=2, layout_omission_score=0.05, n_regions=4, n_uncovered=0, n_facts=4, n_omitted=0),
    ]
    s = summarize_fin_layout(pages)
    assert s.n_pages == 3
    assert s.n_pages_with_omission == 1
    assert s.mean_omission_omitted_pages == 0.30
    assert s.mean_omission_clean_pages == pytest.approx(0.035, abs=1e-6)
    assert s.pointbiserial_omission_vs_has_omission > 0


def test_summarize_fin_layout_empty_raises():
    with pytest.raises(ValueError):
        summarize_fin_layout([])


def test_summarize_gedi_layout_aggregates_rates_and_fill():
    pages = [
        ZoneRegionPage(name="a", n_zone_regions=2, n_zone_uncovered=2, n_other_regions=8,
                       n_other_uncovered=1, mean_fill_in_zone=0.1, mean_fill_elsewhere=0.9),
        ZoneRegionPage(name="b", n_zone_regions=1, n_zone_uncovered=0, n_other_regions=5,
                       n_other_uncovered=0, mean_fill_in_zone=0.9, mean_fill_elsewhere=0.85),
    ]
    s = summarize_gedi_layout(pages)
    assert s.n_pages == 2
    assert s.uncovered_rate_in_zone == pytest.approx(2 / 3, abs=1e-4)
    assert s.uncovered_rate_elsewhere == pytest.approx(1 / 13, abs=1e-4)
    assert s.n_pages_zone_fill_lower == 1  # only page "a" has lower in-zone fill
    assert s.mean_fill_in_zone == pytest.approx((0.1 + 0.9) / 2, abs=1e-4)


def test_summarize_gedi_layout_handles_missing_elsewhere_group():
    pages = [
        ZoneRegionPage(name="a", n_zone_regions=1, n_zone_uncovered=1, n_other_regions=0,
                       n_other_uncovered=0, mean_fill_in_zone=0.0, mean_fill_elsewhere=None),
    ]
    s = summarize_gedi_layout(pages)
    assert s.mean_fill_elsewhere is None
    assert s.uncovered_rate_elsewhere is None
    assert s.n_pages_zone_fill_lower == 0  # can't compare, elsewhere is None


def test_summarize_gedi_layout_empty_raises():
    with pytest.raises(ValueError):
        summarize_gedi_layout([])


def test_rect_overlap_frac_full_partial_none():
    zone = Zone(kind="DLSignature", x0=0, y0=0, x1=100, y1=100)
    assert _rect_overlap_frac((0, 0, 100, 100), zone) == 1.0
    assert _rect_overlap_frac((50, 0, 150, 100), zone) == pytest.approx(0.5, abs=1e-6)
    assert _rect_overlap_frac((200, 200, 300, 300), zone) == 0.0


def test_rect_overlap_frac_degenerate_box_is_zero():
    zone = Zone(kind="DLSignature", x0=0, y0=0, x1=100, y1=100)
    assert _rect_overlap_frac((10, 10, 10, 50), zone) == 0.0


def test_region_in_zone_threshold():
    zones = [Zone(kind="DLSignature", x0=0, y0=0, x1=100, y1=100)]
    # 50% overlap: passes a 0.3 threshold, fails a 0.6 threshold
    box = (50, 0, 150, 100)
    assert region_in_zone(box, zones, min_overlap=0.3) is True
    assert region_in_zone(box, zones, min_overlap=0.6) is False


def test_region_in_zone_no_zones_is_false():
    assert region_in_zone((0, 0, 10, 10), [], min_overlap=0.3) is False
