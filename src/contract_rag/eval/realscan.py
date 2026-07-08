"""Real-scan quality-score calibration (Tobacco800 / any dir of scanned images or PDFs).

The synthetic `dirtify` suite can't push clean digital contracts below ~0.5 quality
(the table-integrity + confidence weight reservation) — the 0.4→0.9 story is really
about scanned/OCR inputs. This harness measures that for real: for every document in
`REALSCAN_DIR` it runs probe → route → parse → `compute_quality_score()` on the raw
IR, then `clean_ir()` → score again, and prints per-doc + aggregate quality, the
needs_review rate, and the probe text-coverage distribution (evidence the docs route
away from docling). Images (TIFF/PNG/JPG) are converted to single-page PDFs first —
the pipeline ingests PDFs.

Optionally (`REALSCAN_GT_DIR`), Tobacco800's GEDI signature/logo groundtruth is used
for the occlusion experiment: do OCR blocks overlapping a signature/stamp region score
lower confidence / higher garble than blocks elsewhere?

Pure logic (listing, conversion math, scoring, GEDI parsing, overlap stats) is
separated from the `__main__` shell; unit tests use hand-built IRs and fake adapters.

Env: REALSCAN_DIR (required), REALSCAN_SET_SIZE (default 100), REALSCAN_GT_DIR
(optional GEDI XML dir), REALSCAN_OUT (optional JSON dump path),
REALSCAN_CACHE (default ~/.cache/contract-rag/realscan).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from contract_rag.clean.pipeline import clean_ir
from contract_rag.clean.quality import QualityReport, compute_quality_score, is_garbled
from contract_rag.config import Settings
from contract_rag.eval.gedi import (
    PageZones,
    Zone,
    block_overlaps,
    parse_gedi,
    scale_zones,
    zone_scale,
)
from contract_rag.eval.scanio import IMAGE_SUFFIXES, ensure_pdf, image_dpi, image_to_pdf
from contract_rag.ir import DocBlock, DocumentIR
from contract_rag.parse.probe import DocProfile, probe_document
from contract_rag.parse.router import route

# re-exported for back-compat; canonical homes are eval.scanio / eval.gedi
_ = (ensure_pdf, image_to_pdf, PageZones, Zone, block_overlaps, parse_gedi, scale_zones, zone_scale)

_RENDER_DPI = 300  # paddle_parser renders PDF pages at this dpi


# ---------------------------------------------------------------- input listing

def list_input_docs(realscan_dir: Path, cap: int = 100) -> list[Path]:
    """Deterministic (name-sorted) list of scans, capped at `cap`."""
    dir_ = Path(realscan_dir)
    docs = sorted(
        (p for p in dir_.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES | {".pdf"}),
        key=lambda p: p.name,
    )
    return docs[: max(0, cap)]


# ---------------------------------------------------------------- per-doc scoring

class DocQuality(BaseModel):
    name: str
    engine: str
    page_count: int
    text_coverage: float
    raw: QualityReport
    cleaned: QualityReport


def evaluate_doc(
    pdf_path: Path,
    settings: Settings,
    probe_fn: Callable[[Path], DocProfile] | None = None,
    adapters: dict[str, Callable[[Path, Settings], DocumentIR]] | None = None,
    clean_fn: Callable[[DocumentIR], DocumentIR] = clean_ir,
    score_fn: Callable[[DocumentIR], QualityReport] = compute_quality_score,
    name: str | None = None,
) -> DocQuality:
    """probe → route → parse → score raw → clean → score cleaned. Same seams as
    `router.parse`, but the routing decision and coverage are recorded, not hidden."""
    probe_fn = probe_fn or probe_document
    if adapters is None:
        from contract_rag.parse.router import _default_adapters

        adapters = _default_adapters()
    profile = probe_fn(pdf_path)
    engine = route(profile, settings)
    ir = adapters[engine](pdf_path, settings)
    raw = score_fn(ir)
    cleaned = score_fn(clean_fn(ir))
    return DocQuality(
        name=name or Path(pdf_path).stem,
        engine=engine,
        page_count=profile.page_count,
        text_coverage=profile.text_coverage,
        raw=raw,
        cleaned=cleaned,
    )


# ---------------------------------------------------------------- aggregation

def percentile(vals: list[float], q: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    if not vals:
        raise ValueError("percentile of empty list")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]


class SideStats(BaseModel):
    """Aggregate of one side (raw or cleaned) over all docs."""

    mean_quality: float
    median_quality: float
    p10_quality: float
    needs_review_rate: float
    mean_garble: float
    mean_table_integrity: float
    mean_empty: float
    mean_confidence: float


class Summary(BaseModel):
    n_docs: int
    total_pages: int
    engines: dict[str, int] = Field(default_factory=dict)
    coverage_mean: float
    coverage_max: float
    raw: SideStats
    cleaned: SideStats


def _side_stats(reports: list[QualityReport]) -> SideStats:
    n = len(reports)
    qs = [r.quality_score for r in reports]
    return SideStats(
        mean_quality=round(sum(qs) / n, 3),
        median_quality=round(percentile(qs, 0.5), 3),
        p10_quality=round(percentile(qs, 0.1), 3),
        needs_review_rate=round(sum(r.needs_review for r in reports) / n, 3),
        mean_garble=round(sum(r.garble_ratio for r in reports) / n, 3),
        mean_table_integrity=round(sum(r.table_integrity for r in reports) / n, 3),
        mean_empty=round(sum(r.empty_ratio for r in reports) / n, 3),
        mean_confidence=round(sum(r.mean_confidence for r in reports) / n, 3),
    )


def summarize(results: list[DocQuality]) -> Summary:
    if not results:
        raise ValueError("no documents evaluated")
    engines: dict[str, int] = {}
    for r in results:
        engines[r.engine] = engines.get(r.engine, 0) + 1
    covs = [r.text_coverage for r in results]
    return Summary(
        n_docs=len(results),
        total_pages=sum(r.page_count for r in results),
        engines=engines,
        coverage_mean=round(sum(covs) / len(covs), 3),
        coverage_max=round(max(covs), 3),
        raw=_side_stats([r.raw for r in results]),
        cleaned=_side_stats([r.cleaned for r in results]),
    )


def format_report(results: list[DocQuality], summary: Summary) -> str:
    lines = [
        f"{'doc':<28} {'engine':<10} {'cov':>5} {'raw':>6} {'clean':>6} {'review':>6}",
    ]
    for r in results:
        lines.append(
            f"{r.name:<28} {r.engine:<10} {r.text_coverage:>5.2f}"
            f" {r.raw.quality_score:>6.3f} {r.cleaned.quality_score:>6.3f}"
            f" {'yes' if r.raw.needs_review else 'no':>6}"
        )
    s = summary
    lines += [
        "",
        f"docs={s.n_docs} pages={s.total_pages} engines={s.engines}"
        f" probe-coverage mean={s.coverage_mean} max={s.coverage_max}",
        f"{'':<10} {'mean':>6} {'median':>6} {'p10':>6} {'review%':>8}"
        f" {'garble':>7} {'table':>6} {'empty':>6} {'conf':>6}",
    ]
    for label, side in (("raw", s.raw), ("cleaned", s.cleaned)):
        lines.append(
            f"{label:<10} {side.mean_quality:>6.3f} {side.median_quality:>6.3f}"
            f" {side.p10_quality:>6.3f} {side.needs_review_rate:>8.1%}"
            f" {side.mean_garble:>7.3f} {side.mean_table_integrity:>6.3f}"
            f" {side.mean_empty:>6.3f} {side.mean_confidence:>6.3f}"
        )
    return "\n".join(lines)


# ------------------------------------------------- GEDI occlusion experiment
# (Zone / PageZones / parse_gedi / zone_scale / scale_zones / block_overlaps now
#  live in eval.gedi and are imported above; re-exported for back-compat.)


class OverlapStats(BaseModel):
    n_pages: int
    n_overlap: int
    n_other: int
    mean_conf_overlap: float | None
    mean_conf_other: float | None
    garble_rate_overlap: float | None
    garble_rate_other: float | None


def overlap_stats(pairs: list[tuple[list[DocBlock], list[Zone]]]) -> OverlapStats:
    """Pooled over (blocks, zones-in-block-coords) pairs: do blocks under a
    signature/logo region carry lower OCR confidence / more garble?"""
    conf_o: list[float] = []
    conf_x: list[float] = []
    garb_o: list[bool] = []
    garb_x: list[bool] = []
    for blocks, zones in pairs:
        for b in blocks:
            if b.bbox is None:
                continue
            if block_overlaps(b, zones):
                conf_o.append(b.confidence)
                garb_o.append(is_garbled(b.text))
            else:
                conf_x.append(b.confidence)
                garb_x.append(is_garbled(b.text))

    def _mean(vals) -> float | None:
        return round(sum(vals) / len(vals), 3) if vals else None

    return OverlapStats(
        n_pages=len(pairs),
        n_overlap=len(conf_o),
        n_other=len(conf_x),
        mean_conf_overlap=_mean(conf_o),
        mean_conf_other=_mean(conf_x),
        garble_rate_overlap=_mean(garb_o),
        garble_rate_other=_mean(garb_x),
    )


# ---------------------------------------------------------------- impure runner

def run_realscan(
    realscan_dir: Path,
    settings: Settings,
    cache_dir: Path,
    cap: int = 100,
    gt_dir: Path | None = None,
) -> tuple[list[DocQuality], Summary, OverlapStats | None]:
    """The real thing: convert images, parse via the router (IR-cached per engine so
    re-runs are fast), score raw + cleaned, and optionally run the GEDI occlusion
    experiment on the raw IRs."""
    from contract_rag.eval.ir_cache import ir_cache
    from contract_rag.parse.router import _default_adapters

    cache_dir = Path(cache_dir)
    real_adapters = _default_adapters()
    adapters = {
        eng: (lambda e, fn: lambda p, s: ir_cache(cache_dir / "ir" / e, lambda pp: fn(pp, s))(p))(
            eng, fn
        )
        for eng, fn in real_adapters.items()
    }

    docs = list_input_docs(realscan_dir, cap)
    if not docs:
        raise SystemExit(f"no scans (pdf/{'/'.join(sorted(IMAGE_SUFFIXES))}) in {realscan_dir}")

    results: list[DocQuality] = []
    overlap_pairs: list[tuple[list[DocBlock], list[Zone]]] = []
    for doc in docs:
        pdf = ensure_pdf(doc, cache_dir / "pdf")
        dq = evaluate_doc(pdf, settings, adapters=adapters, name=doc.stem)
        results.append(dq)
        if gt_dir is not None and doc.suffix.lower() in IMAGE_SUFFIXES:
            xml = Path(gt_dir) / f"{doc.stem}.xml"
            if xml.exists():
                pz = parse_gedi(xml.read_text(errors="replace"))
                if pz.zones:
                    ir = adapters[dq.engine](pdf, settings)  # cached
                    s = zone_scale(image_dpi(doc))
                    overlap_pairs.append((ir.blocks, scale_zones(pz.zones, s)))

    summary = summarize(results)
    overlap = overlap_stats(overlap_pairs) if overlap_pairs else None
    return results, summary, overlap


def main() -> None:
    import json
    import os

    from contract_rag.config import get_settings

    realscan_dir = os.environ.get("REALSCAN_DIR")
    if not realscan_dir:
        raise SystemExit("set REALSCAN_DIR to a directory of scanned images/PDFs")
    cap = int(os.environ.get("REALSCAN_SET_SIZE", "100"))
    gt = os.environ.get("REALSCAN_GT_DIR")
    cache = Path(
        os.environ.get(
            "REALSCAN_CACHE", str(Path.home() / ".cache" / "contract-rag" / "realscan")
        )
    )
    results, summary, overlap = run_realscan(
        Path(realscan_dir), get_settings(), cache, cap=cap, gt_dir=Path(gt) if gt else None
    )
    print(format_report(results, summary))
    if overlap is not None:
        o = overlap
        print(
            f"\nocclusion (GEDI signature/logo zones, {o.n_pages} pages):"
            f"\n  blocks overlapping a zone: n={o.n_overlap}"
            f" mean_conf={o.mean_conf_overlap} garble_rate={o.garble_rate_overlap}"
            f"\n  blocks elsewhere:          n={o.n_other}"
            f" mean_conf={o.mean_conf_other} garble_rate={o.garble_rate_other}"
        )
    out = os.environ.get("REALSCAN_OUT")
    if out:
        payload = {
            "results": [r.model_dump() for r in results],
            "summary": summary.model_dump(),
            "overlap": overlap.model_dump() if overlap else None,
        }
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
