"""Validate the layout-model coverage signal (`clean/layout_coverage.py`) against
the same two ground-truth datasets `eval/coverage.py` used for the geometric
ink-coverage signal, so the two are directly comparable.

The geometric signal (whole-page uncovered-ink ratio) is a strong region-scale
omission detector (Tobacco800 GEDI signature/logo zones: 5.4x density) but weak at
single-fact granularity (FinCriticalED point-biserial 0.18, pearson 0.15) — a
dropped number is a few pixels among thousands, so it barely dents a whole-page
ratio. This harness asks whether scoring LAYOUT-DETECTOR regions against the OCR
blocks that fill them (rather than raw ink pixels against a whole page) does
better at that same single-fact granularity:

1. **FinCriticalED** (fact-level omission truth). For each cached paddle page IR,
   run the layout detector, compute doc-level `layout_omission_score`, and join
   against the number of expert gold facts that vanished from the OCR text (via
   `eval.fincritical`). Same statistics as the geometric run: point-biserial /
   pearson correlation with page-level omission, and mean omission on
   fact-omitting vs clean pages — directly comparable to the 0.18 baseline.

2. **Tobacco800 GEDI** (occlusion / signature truth). For each page with a
   DLSignature/DLLogo zone, split the detected layout regions into
   zone-overlapping vs elsewhere and compare mean OCR fill_ratio / uncovered rate
   between the two groups — the region-level analogue of the geometric run's
   in-zone-vs-elsewhere ink density.

Layout-model inference is comparatively expensive (a real detector call per page,
not a numpy mask), so results are disk-cached per (pdf, page) under
`LAYOUT_REGIONS_CACHE` (default `~/.cache/contract-rag/layout-regions`) — re-runs
after the first are fast and never re-invoke the model.

Env: LAYOUT_COVERAGE_DATASET (fincritical|tobacco|both, default both — each
auto-skips if its cache is absent), LAYOUT_COVERAGE_SET_SIZE (default 100),
LAYOUT_COVERAGE_FILL_THRESHOLD (region "covered" verdict, default 0.5),
LAYOUT_COVERAGE_DILATE (OCR-box dilation px at 300dpi, default 4), and the same
cache dirs `eval/coverage.py` uses: FINCRITICAL_CACHE / FINCRITICAL_DIR /
REALSCAN_CACHE / REALSCAN_GT_DIR / TOBACCO_IMG_DIR. LAYOUT_COVERAGE_OUT dumps JSON.

Run: uv run python -m contract_rag.eval.layout_coverage
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from contract_rag.clean.layout_coverage import (
    LayoutRegion,
    document_layout_coverage,
    page_layout_coverage,
)
from contract_rag.eval.coverage import _mean, _pearson
from contract_rag.ir import DocumentIR

_PADDLE_RENDER_DPI = 300.0


# ============================================================ FinCriticalED omission vs layout omission


class FinLayoutPage(BaseModel):
    page_id: int
    layout_omission_score: float
    n_regions: int
    n_uncovered: int
    n_facts: int
    n_omitted: int


class FinLayoutSummary(BaseModel):
    n_pages: int
    n_pages_with_omission: int
    mean_omission_omitted_pages: float | None    # pages with >=1 omitted fact
    mean_omission_clean_pages: float | None       # pages with 0 omitted facts
    pearson_omission_vs_omission_rate: float | None
    pointbiserial_omission_vs_has_omission: float | None
    pages: list[FinLayoutPage] = Field(default_factory=list)


def summarize_fin_layout(pages: Sequence[FinLayoutPage]) -> FinLayoutSummary:
    if not pages:
        raise ValueError("no FinCriticalED pages evaluated")
    with_om = [p.layout_omission_score for p in pages if p.n_omitted >= 1]
    clean = [p.layout_omission_score for p in pages if p.n_omitted == 0]
    omis = [p.layout_omission_score for p in pages]
    om_rate = [p.n_omitted / p.n_facts if p.n_facts else 0.0 for p in pages]
    has_om = [1.0 if p.n_omitted >= 1 else 0.0 for p in pages]
    return FinLayoutSummary(
        n_pages=len(pages),
        n_pages_with_omission=len(with_om),
        mean_omission_omitted_pages=_mean(with_om),
        mean_omission_clean_pages=_mean(clean),
        pearson_omission_vs_omission_rate=_pearson(omis, om_rate),
        pointbiserial_omission_vs_has_omission=_pearson(omis, has_om),
        pages=list(pages),
    )


# ============================================================ Tobacco800 signature-zone region split


def _rect_overlap_frac(box: tuple, zone) -> float:
    """Pure: fraction of `box`'s own area covered by `zone` (a `gedi.Zone`, or
    anything with x0/y0/x1/y1). No numpy — a handful of regions per page, plain
    rectangle math is enough."""
    x0, y0, x1, y1 = box
    bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
    barea = bw * bh
    if barea <= 0:
        return 0.0
    ix0, iy0 = max(x0, zone.x0), max(y0, zone.y0)
    ix1, iy1 = min(x1, zone.x1), min(y1, zone.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    return (iw * ih) / barea


def region_in_zone(box: tuple, zones: Sequence, min_overlap: float = 0.3) -> bool:
    """A region counts as "in the signature/logo zone" if at least `min_overlap`
    of its own area falls inside some (scaled) GEDI zone."""
    return any(_rect_overlap_frac(box, z) >= min_overlap for z in zones)


class ZoneRegionPage(BaseModel):
    name: str
    n_zone_regions: int
    n_zone_uncovered: int
    n_other_regions: int
    n_other_uncovered: int
    mean_fill_in_zone: float | None
    mean_fill_elsewhere: float | None


class GediLayoutSummary(BaseModel):
    n_pages: int
    mean_fill_in_zone: float | None
    mean_fill_elsewhere: float | None
    uncovered_rate_in_zone: float | None
    uncovered_rate_elsewhere: float | None
    n_pages_zone_fill_lower: int   # pages where in-zone mean fill < elsewhere
    pages: list[ZoneRegionPage] = Field(default_factory=list)


def summarize_gedi_layout(pages: Sequence[ZoneRegionPage]) -> GediLayoutSummary:
    if not pages:
        raise ValueError("no Tobacco800 pages with signature/logo zones evaluated")
    fill_in = [p.mean_fill_in_zone for p in pages if p.mean_fill_in_zone is not None]
    fill_out = [p.mean_fill_elsewhere for p in pages if p.mean_fill_elsewhere is not None]
    n_zone = sum(p.n_zone_regions for p in pages)
    n_zone_unc = sum(p.n_zone_uncovered for p in pages)
    n_other = sum(p.n_other_regions for p in pages)
    n_other_unc = sum(p.n_other_uncovered for p in pages)
    lower = sum(
        1 for p in pages
        if p.mean_fill_in_zone is not None and p.mean_fill_elsewhere is not None
        and p.mean_fill_in_zone < p.mean_fill_elsewhere
    )
    return GediLayoutSummary(
        n_pages=len(pages),
        mean_fill_in_zone=_mean(fill_in),
        mean_fill_elsewhere=_mean(fill_out),
        uncovered_rate_in_zone=round(n_zone_unc / n_zone, 4) if n_zone else None,
        uncovered_rate_elsewhere=round(n_other_unc / n_other, 4) if n_other else None,
        n_pages_zone_fill_lower=lower,
        pages=list(pages),
    )


# ============================================================ region cache (impure)


def _cache_key(pdf_path: Path, page_index: int) -> str:
    return f"{pdf_path.stem}_p{page_index}"


def cached_detect_page_regions(
    pdf_path: Path, page_index: int, cache_dir: Path | None, dpi: float = _PADDLE_RENDER_DPI,
) -> list[LayoutRegion]:
    """Disk-cached wrapper around `layout_coverage.detect_page_regions_from_pdf` —
    layout-model inference is the expensive step here, so a re-run over the same
    cached IRs shouldn't re-invoke it. `cache_dir=None` disables caching."""
    from contract_rag.clean.layout_coverage import detect_page_regions_from_pdf

    if cache_dir is None:
        return detect_page_regions_from_pdf(pdf_path, page_index, dpi=dpi)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(pdf_path, page_index)}.json"
    if cache_file.exists():
        raw = json.loads(cache_file.read_text())
        return [LayoutRegion.model_validate(r) for r in raw]
    regions = detect_page_regions_from_pdf(pdf_path, page_index, dpi=dpi)
    cache_file.write_text(json.dumps([r.model_dump() for r in regions]))
    return regions


# ============================================================ impure runners


def run_fincritical_layout_coverage(
    data_dir: Path,
    ir_dir: Path,
    pdf_dir: Path,
    *,
    cap: int = 100,
    fill_threshold: float = 0.5,
    dilate_px: float = 3.0,
    regions_cache_dir: Path | None = None,
    regions_fn: Callable[[Path, int], Sequence[LayoutRegion]] | None = None,
) -> FinLayoutSummary:
    """Join cached FinCriticalED paddle IRs with the gold omission labels, scored
    via layout-detector region fill instead of raw ink coverage."""
    from contract_rag.eval.fincritical import evaluate_page, load_samples, parse_gold_html

    regions_fn = regions_fn or (
        lambda p, i: cached_detect_page_regions(p, i, regions_cache_dir)
    )
    by_id = {s.page_id: s for s in load_samples(data_dir, cap=cap)}
    pages: list[FinLayoutPage] = []
    for pid, sample in sorted(by_id.items()):
        irp = Path(ir_dir) / f"fincritical_{pid}.ir.json"
        pdfp = Path(pdf_dir) / f"fincritical_{pid}.pdf"
        if not (irp.exists() and pdfp.exists()):
            continue
        ir = DocumentIR.model_validate_json(irp.read_text())
        facts = parse_gold_html(sample.gold_html)
        if not facts:
            continue
        outcomes = evaluate_page(ir, facts)
        dc = document_layout_coverage(
            ir, lambda i, p=pdfp: regions_fn(p, i),
            fill_threshold=fill_threshold, dilate_px=dilate_px,
        )
        pages.append(FinLayoutPage(
            page_id=pid,
            layout_omission_score=dc.layout_omission_score,
            n_regions=dc.n_regions,
            n_uncovered=dc.n_uncovered,
            n_facts=len(outcomes),
            n_omitted=sum(not o.in_document for o in outcomes),
        ))
    return summarize_fin_layout(pages)


def run_tobacco_layout_coverage(
    ir_dir: Path,
    pdf_dir: Path,
    gt_dir: Path,
    img_dir: Path,
    *,
    cap: int = 100,
    fill_threshold: float = 0.5,
    dilate_px: float = 4.0,
    min_zone_overlap: float = 0.3,
    regions_cache_dir: Path | None = None,
    regions_fn: Callable[[Path, int], Sequence[LayoutRegion]] | None = None,
) -> GediLayoutSummary:
    """Signature/logo-zone region-fill split on cached Tobacco800 paddle IRs.
    Regions are detected at paddle's 300dpi so IR boxes are identity-scaled and
    GEDI zones map by `zone_scale(image_dpi)`, exactly like `eval.coverage`."""
    from contract_rag.eval.gedi import parse_gedi, scale_zones, zone_scale
    from contract_rag.eval.scanio import image_dpi

    regions_fn = regions_fn or (
        lambda p, i: cached_detect_page_regions(p, i, regions_cache_dir)
    )
    pages: list[ZoneRegionPage] = []
    for xmlp in sorted(Path(gt_dir).glob("*.xml"))[:cap]:
        stem = xmlp.stem
        irp = Path(ir_dir) / f"{stem}.ir.json"
        pdfp = Path(pdf_dir) / f"{stem}.pdf"
        tif = Path(img_dir) / f"{stem}.tif"
        if not (irp.exists() and pdfp.exists() and tif.exists()):
            continue
        pz = parse_gedi(xmlp.read_text(errors="replace"))
        zones = [z for z in pz.zones if z.kind in ("DLSignature", "DLLogo")]
        if not zones:
            continue
        ir = DocumentIR.model_validate_json(irp.read_text())
        scaled = scale_zones(zones, zone_scale(image_dpi(tif)))

        regions = regions_fn(pdfp, 0)
        from contract_rag.clean.coverage import ir_page_boxes

        ocr_boxes = ir_page_boxes(ir, 1)  # 300dpi identity, matches regions_fn's dpi
        pc = page_layout_coverage(regions, ocr_boxes, page=1, fill_threshold=fill_threshold, dilate_px=dilate_px)
        if not pc.regions:
            continue

        in_zone = [r for r in pc.regions if region_in_zone(r.box, scaled, min_zone_overlap)]
        elsewhere = [r for r in pc.regions if not region_in_zone(r.box, scaled, min_zone_overlap)]
        if not in_zone:
            continue
        pages.append(ZoneRegionPage(
            name=stem,
            n_zone_regions=len(in_zone),
            n_zone_uncovered=sum(1 for r in in_zone if not r.covered),
            n_other_regions=len(elsewhere),
            n_other_uncovered=sum(1 for r in elsewhere if not r.covered),
            mean_fill_in_zone=_mean([r.fill_ratio for r in in_zone]),
            mean_fill_elsewhere=_mean([r.fill_ratio for r in elsewhere]) if elsewhere else None,
        ))
    return summarize_gedi_layout(pages)


# ============================================================ reporting


def format_fin_layout(s: FinLayoutSummary) -> str:
    return "\n".join([
        "=== FinCriticalED: layout-region omission vs fact omission ===",
        f"pages={s.n_pages}  with>=1 omission={s.n_pages_with_omission}",
        f"mean layout_omission_score  omitted-pages={s.mean_omission_omitted_pages}"
        f"  clean-pages={s.mean_omission_clean_pages}",
        f"pearson(layout_omission, page omission rate)={s.pearson_omission_vs_omission_rate}",
        f"pointbiserial(layout_omission, has-omission)={s.pointbiserial_omission_vs_has_omission}",
        "(compare to the geometric ink-coverage baseline: pointbiserial 0.18, pearson 0.15)",
    ])


def format_gedi_layout(s: GediLayoutSummary) -> str:
    return "\n".join([
        "=== Tobacco800 GEDI: layout-region OCR fill in signature/logo zones ===",
        f"pages_with_zones={s.n_pages}",
        f"mean region fill_ratio  in-zone={s.mean_fill_in_zone}  elsewhere={s.mean_fill_elsewhere}",
        f"uncovered-region rate  in-zone={s.uncovered_rate_in_zone}  elsewhere={s.uncovered_rate_elsewhere}",
        f"in-zone fill lower than elsewhere on {s.n_pages_zone_fill_lower}/{s.n_pages} pages",
    ])


# ============================================================ CLI


def main() -> None:
    import os

    which = os.environ.get("LAYOUT_COVERAGE_DATASET", "both")
    cap = int(os.environ.get("LAYOUT_COVERAGE_SET_SIZE", "100"))
    fill_threshold = float(os.environ.get("LAYOUT_COVERAGE_FILL_THRESHOLD", "0.5"))
    dilate = float(os.environ.get("LAYOUT_COVERAGE_DILATE", "4"))
    regions_cache = Path(os.environ.get(
        "LAYOUT_REGIONS_CACHE", str(Path.home() / ".cache" / "contract-rag" / "layout-regions"),
    ))
    payload: dict = {}

    if which in ("fincritical", "both"):
        fin_cache = Path(os.environ.get(
            "FINCRITICAL_CACHE",
            str(Path.home() / ".cache" / "contract-rag" / "fincriticaled-run"),
        ))
        fin_data = Path(os.environ.get(
            "FINCRITICAL_DIR",
            str(Path.home() / ".cache" / "contract-rag" / "fincriticaled"),
        ))
        if (fin_cache / "ir").exists() and (fin_data / "raw_input.csv").exists():
            fs = run_fincritical_layout_coverage(
                fin_data, fin_cache / "ir", fin_cache / "pdf",
                cap=cap, fill_threshold=fill_threshold, dilate_px=dilate,
                regions_cache_dir=regions_cache / "fincritical",
            )
            print(format_fin_layout(fs))
            payload["fincritical"] = fs.model_dump()
        else:
            print(f"[skip] FinCriticalED cache not found under {fin_cache} / {fin_data}")

    if which in ("tobacco", "both"):
        rs_cache = Path(os.environ.get(
            "REALSCAN_CACHE", str(Path.home() / ".cache" / "contract-rag" / "realscan")
        ))
        gt = Path(os.environ.get(
            "REALSCAN_GT_DIR", str(Path.home() / ".cache" / "tobacco800" / "groundtruth")
        ))
        img = Path(os.environ.get(
            "TOBACCO_IMG_DIR", str(Path.home() / ".cache" / "tobacco800" / "images")
        ))
        ir_dir = rs_cache / "ir" / "paddleocr"
        if ir_dir.exists() and gt.exists() and img.exists():
            gs = run_tobacco_layout_coverage(
                ir_dir, rs_cache / "pdf", gt, img,
                cap=cap, fill_threshold=fill_threshold, dilate_px=dilate,
                regions_cache_dir=regions_cache / "tobacco",
            )
            print("\n" + format_gedi_layout(gs))
            payload["tobacco"] = gs.model_dump()
        else:
            print(f"[skip] Tobacco800 cache/groundtruth not found under {rs_cache} / {gt} / {img}")

    out = os.environ.get("LAYOUT_COVERAGE_OUT")
    if out and payload:
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
