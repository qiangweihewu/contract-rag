"""Validate the geometric ink-coverage signal (`clean/coverage.py`) against two
datasets that carry omission / occlusion ground truth.

The signal's premise: OCR omissions are invisible to `quality.compute_quality_score`
(no block → no garble/empty/confidence penalty), but they DO leave visible ink the
OCR boxes fail to cover. This harness measures whether uncovered ink actually
tracks real omissions:

1. **FinCriticalED** (fact-level omission truth). For each cached paddle page IR,
   compute doc-level `uncovered_ink_ratio` and, via `eval.fincritical`, the number
   of expert gold facts that vanished from the OCR text. Question: do pages with
   >=1 omitted fact carry more uncovered ink than pages with none? (a page-level
   correlation).

2. **Tobacco800 GEDI** (occlusion / signature truth — the sharper test, because we
   know *where* the missed ink is). For each page with a DLSignature/DLLogo zone,
   compare the density of uncovered ink INSIDE the annotated signature/logo zone to
   ELSEWHERE. If the geometric signal is real, the ink OCR under-reads should
   concentrate in exactly those zones.

Reuses the cached IRs/PDFs/gold from the `realscan` and `fincritical` runs and the
`eval.gedi` / `eval.fincritical` primitives — no re-parse. Pure aggregation
(correlation, density stats) is separated from the impure cache/render runners so
unit tests use hand-built inputs.

Env: COVERAGE_DATASET (fincritical|tobacco|both, default both — each auto-skips if
its cache is absent), COVERAGE_SET_SIZE (default 100), COVERAGE_DPI (mask render
dpi, default 150 — boxes are scaled from paddle's 300), COVERAGE_DILATE (box
dilation px at the render dpi, default 3), COVERAGE_BORDER (border-ignore fraction,
default 0.0), and the cache dirs FINCRITICAL_CACHE / FINCRITICAL_DIR / REALSCAN_CACHE
/ REALSCAN_GT_DIR / TOBACCO_IMG_DIR. COVERAGE_OUT dumps JSON.

Run: uv run python -m contract_rag.eval.coverage
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from contract_rag.clean.coverage import (
    ir_page_boxes,
    render_page_gray,
    uncovered_ink_mask,
)
from contract_rag.ir import DocumentIR

_PADDLE_RENDER_DPI = 300.0


# ============================================================ pure correlation


def _mean(xs: Sequence[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson r without numpy (the aggregation stays dependency-free)."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None
    return round(sxy / (sxx * syy) ** 0.5, 4)


# ------------------------------------------------- FinCriticalED omission vs coverage

class FinPage(BaseModel):
    page_id: int
    uncovered_ink_ratio: float
    ink_coverage: float
    n_facts: int
    n_omitted: int


class FinSummary(BaseModel):
    n_pages: int
    n_pages_with_omission: int
    mean_uncovered_omitted_pages: float | None   # pages with >=1 omitted fact
    mean_uncovered_clean_pages: float | None      # pages with 0 omitted facts
    pearson_uncovered_vs_omission_rate: float | None
    pointbiserial_uncovered_vs_has_omission: float | None
    pages: list[FinPage] = Field(default_factory=list)


def summarize_fin(pages: Sequence[FinPage]) -> FinSummary:
    if not pages:
        raise ValueError("no FinCriticalED pages evaluated")
    with_om = [p.uncovered_ink_ratio for p in pages if p.n_omitted >= 1]
    clean = [p.uncovered_ink_ratio for p in pages if p.n_omitted == 0]
    uncov = [p.uncovered_ink_ratio for p in pages]
    om_rate = [p.n_omitted / p.n_facts if p.n_facts else 0.0 for p in pages]
    has_om = [1.0 if p.n_omitted >= 1 else 0.0 for p in pages]
    return FinSummary(
        n_pages=len(pages),
        n_pages_with_omission=len(with_om),
        mean_uncovered_omitted_pages=_mean(with_om),
        mean_uncovered_clean_pages=_mean(clean),
        pearson_uncovered_vs_omission_rate=_pearson(uncov, om_rate),
        pointbiserial_uncovered_vs_has_omission=_pearson(uncov, has_om),
        pages=list(pages),
    )


# ------------------------------------------------- Tobacco800 signature-zone density

class ZonePage(BaseModel):
    name: str
    density_in_zone: float     # fraction of in-zone pixels that are uncovered ink
    density_elsewhere: float
    n_zone_px: int


class GediSummary(BaseModel):
    n_pages: int
    mean_density_in_zone: float | None
    mean_density_elsewhere: float | None
    ratio_in_over_out: float | None
    n_pages_in_higher: int      # pages where in-zone density exceeds elsewhere
    pages: list[ZonePage] = Field(default_factory=list)


def zone_density(uncovered, zone_mask) -> tuple[float, float, int]:
    """Pure: (uncovered-ink density inside the zone mask, density elsewhere,
    zone-pixel count). `uncovered` / `zone_mask` are equal-shape boolean arrays."""
    n_zone = int(zone_mask.sum())
    n_out = int((~zone_mask).sum())
    din = float(uncovered[zone_mask].mean()) if n_zone else 0.0
    dout = float(uncovered[~zone_mask].mean()) if n_out else 0.0
    return round(din, 4), round(dout, 4), n_zone


def summarize_gedi(pages: Sequence[ZonePage]) -> GediSummary:
    if not pages:
        raise ValueError("no Tobacco800 pages with signature/logo zones evaluated")
    ins = [p.density_in_zone for p in pages]
    outs = [p.density_elsewhere for p in pages]
    mi, mo = _mean(ins), _mean(outs)
    return GediSummary(
        n_pages=len(pages),
        mean_density_in_zone=mi,
        mean_density_elsewhere=mo,
        ratio_in_over_out=round(mi / mo, 2) if mi and mo else None,
        n_pages_in_higher=sum(1 for p in pages if p.density_in_zone > p.density_elsewhere),
        pages=list(pages),
    )


# ============================================================ impure runners


def _zone_mask(shape, zones, w: int, h: int):
    """Rasterize scaled GEDI zones (rendered-px coords) into a boolean mask."""
    import numpy as np

    zmask = np.zeros(shape, dtype=bool)
    for z in zones:
        x0, y0 = max(0, int(z.x0)), max(0, int(z.y0))
        x1, y1 = min(w, int(z.x1)), min(h, int(z.y1))
        if x1 > x0 and y1 > y0:
            zmask[y0:y1, x0:x1] = True
    return zmask


def run_fincritical_coverage(
    data_dir: Path,
    ir_dir: Path,
    pdf_dir: Path,
    *,
    cap: int = 100,
    dpi: float = 150.0,
    dilate_px: float = 3.0,
    border_ignore_frac: float = 0.0,
    render_fn: Callable[[Path, int], "object"] | None = None,
) -> FinSummary:
    """Join cached FinCriticalED paddle IRs with the gold omission labels."""
    from contract_rag.clean.coverage import document_coverage
    from contract_rag.eval.fincritical import (
        evaluate_page,
        load_samples,
        parse_gold_html,
    )

    render_fn = render_fn or (lambda p, i: render_page_gray(p, i, dpi=dpi))
    by_id = {s.page_id: s for s in load_samples(data_dir, cap=cap)}
    pages: list[FinPage] = []
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
        dc = document_coverage(
            ir, lambda i, p=pdfp: render_fn(p, i),
            dpi=dpi, box_dpi=_PADDLE_RENDER_DPI,
            dilate_px=dilate_px, border_ignore_frac=border_ignore_frac,
        )
        pages.append(FinPage(
            page_id=pid,
            uncovered_ink_ratio=dc.uncovered_ink_ratio,
            ink_coverage=dc.ink_coverage,
            n_facts=len(outcomes),
            n_omitted=sum(not o.in_document for o in outcomes),
        ))
    return summarize_fin(pages)


def run_tobacco_coverage(
    ir_dir: Path,
    pdf_dir: Path,
    gt_dir: Path,
    img_dir: Path,
    *,
    cap: int = 100,
    dilate_px: float = 4.0,
    border_ignore_frac: float = 0.0,
    render_fn: Callable[[Path, int], "object"] | None = None,
) -> GediSummary:
    """Signature/logo-zone uncovered-ink density on cached Tobacco800 paddle IRs.
    Renders the mask at paddle's 300 dpi so IR boxes are identity-scaled and GEDI
    zones map by `zone_scale(image_dpi)`."""
    from contract_rag.eval.gedi import parse_gedi, scale_zones, zone_scale
    from contract_rag.eval.scanio import image_dpi

    render_fn = render_fn or (lambda p, i: render_page_gray(p, i, dpi=_PADDLE_RENDER_DPI))
    pages: list[ZonePage] = []
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
        gray = render_fn(pdfp, 0)
        boxes = ir_page_boxes(ir, 1)  # 300 dpi identity
        uncovered = uncovered_ink_mask(
            gray, boxes, dilate_px=dilate_px, border_ignore_frac=border_ignore_frac
        )
        h, w = uncovered.shape
        zmask = _zone_mask(uncovered.shape, scale_zones(zones, zone_scale(image_dpi(tif))), w, h)
        if not zmask.any() or zmask.all():
            continue
        din, dout, nz = zone_density(uncovered, zmask)
        pages.append(ZonePage(name=stem, density_in_zone=din, density_elsewhere=dout, n_zone_px=nz))
    return summarize_gedi(pages)


# ============================================================ reporting


def format_fin(s: FinSummary) -> str:
    return "\n".join([
        "=== FinCriticalED: ink-coverage vs fact omission ===",
        f"pages={s.n_pages}  with>=1 omission={s.n_pages_with_omission}",
        f"mean uncovered_ink_ratio  omitted-pages={s.mean_uncovered_omitted_pages}"
        f"  clean-pages={s.mean_uncovered_clean_pages}",
        f"pearson(uncovered, page omission rate)={s.pearson_uncovered_vs_omission_rate}",
        f"pointbiserial(uncovered, has-omission)={s.pointbiserial_uncovered_vs_has_omission}",
    ])


def format_gedi(s: GediSummary) -> str:
    return "\n".join([
        "=== Tobacco800 GEDI: uncovered-ink density in signature/logo zones ===",
        f"pages_with_zones={s.n_pages}",
        f"mean uncovered-ink density  in-zone={s.mean_density_in_zone}"
        f"  elsewhere={s.mean_density_elsewhere}  ratio={s.ratio_in_over_out}x",
        f"in-zone density higher on {s.n_pages_in_higher}/{s.n_pages} pages",
    ])


# ============================================================ CLI


def main() -> None:
    import json
    import os

    which = os.environ.get("COVERAGE_DATASET", "both")
    cap = int(os.environ.get("COVERAGE_SET_SIZE", "100"))
    dpi = float(os.environ.get("COVERAGE_DPI", "150"))
    dilate = float(os.environ.get("COVERAGE_DILATE", "3"))
    border = float(os.environ.get("COVERAGE_BORDER", "0.0"))
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
            fs = run_fincritical_coverage(
                fin_data, fin_cache / "ir", fin_cache / "pdf",
                cap=cap, dpi=dpi, dilate_px=dilate, border_ignore_frac=border,
            )
            print(format_fin(fs))
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
            gs = run_tobacco_coverage(
                ir_dir, rs_cache / "pdf", gt, img,
                cap=cap, border_ignore_frac=border,
            )
            print("\n" + format_gedi(gs))
            payload["tobacco"] = gs.model_dump()
        else:
            print(f"[skip] Tobacco800 cache/groundtruth not found under {rs_cache} / {gt} / {img}")

    out = os.environ.get("COVERAGE_OUT")
    if out and payload:
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
