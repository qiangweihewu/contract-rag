"""Image-level controlled degradation — the missing "before" for the 0.4→0.9 demo.

The `realscan` run refuted the naive story ("real scans score low quality"): modern
PaddleOCR reads even ugly 1940s-90s scans at mean quality 0.987, because the quality
formula scores *what OCR emitted*, not *what it missed*. And `eval/dirtify.py` only
corrupts at the IR level (mojibake, hyphenation, dupes) — it can't push a clean digital
contract below ~0.5 (the table-integrity + confidence weight reservation). So the demo's
headline recovery had no real basis: any real scan, and any dirtified clean PDF, scores
too high.

This module is the image-layer analogue of `dirtify`: **seeded, deterministic, physical**
degradation applied to a page image *before* OCR — low DPI (downscale→upscale), skew,
JPEG recompression, fax-style binarization, gaussian / salt-pepper noise, faint
bleed-through. Composed at named intensities ("light"/"medium"/"fax"). Every operator is a
pure function of (image, params[, rng]); PIL/numpy are lazy-imported so the unit suite runs
without them (mirroring the rest of the codebase).

The measurement harness (`run_degrade`) renders a clean *digital* CUAD contract to page
images, degrades them, wraps them back into a PDF (no text layer → the router sends it to
paddleocr, exactly as a real scan), and scores three columns per doc — **original**
(clean digital docling parse) / **degraded** (render→degrade→OCR) / **cleaned**
(`clean_ir` of the degraded IR) — each via `compute_quality_score()` and, where a golden
set exists, extraction field-F1. This finally gives the "clean 0.9 → degraded ~0.4 →
recovered" story a real, reproducible basis.

Honest caveat: like `dirtify`, this dirt is SIMULATED — a controlled, seeded stress test,
not dirt found in the wild. It is a calibration instrument, not a claim that customer scans
look exactly like this.

Env: DEGRADE_LEVEL (light|medium|fax, default medium), DEGRADE_SEED (default 0),
DEGRADE_SET_SIZE (docs, default 6), DEGRADE_MAX_PAGES (0 = all, default 0),
DEGRADE_RENDER_DPI (default 150 — the clean page is rasterized at this dpi before
degradation; paddle later re-renders the degraded PDF at 300), DEGRADE_CACHE
(pdf/IR cache dir), DEGRADE_OUT (optional JSON dump), EXTRACT_BACKEND (rule default).

Run: uv run python -m contract_rag.eval.degrade
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from contract_rag.clean.pipeline import clean_ir
from contract_rag.clean.quality import QualityReport, compute_quality_score
from contract_rag.config import Settings
from contract_rag.ir import DocumentIR

_RENDER_DPI = 150  # dpi the clean page is rasterized at before degradation


# ============================================================ pure image operators
# Each is a pure function of (image, params[, rng]); deterministic ops take no rng, the
# two stochastic ops (gaussian / salt-pepper) take a numpy RandomState so a fixed seed
# reproduces the exact output. All return a PIL "L" (grayscale) image — scans are
# effectively grayscale and it keeps the noise math simple.


def to_gray(img):
    return img if img.mode == "L" else img.convert("L")


def downscale_upscale(img, factor: float):
    """Simulate a low-DPI capture: shrink by `factor` (0<factor<1) then blow back up to
    the original size, permanently losing high-frequency detail (blurry glyph edges)."""
    from PIL import Image

    g = to_gray(img)
    if not (0 < factor < 1):
        return g
    w, h = g.size
    small = g.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def skew(img, degrees: float):
    """Page placed askew on the scanner bed: small rotation, white fill, no expand."""
    from PIL import Image

    g = to_gray(img)
    if degrees == 0:
        return g
    return g.rotate(degrees, resample=Image.BILINEAR, expand=False, fillcolor=255)


def jpeg_recompress(img, quality: int):
    """Lossy storage: round-trip through JPEG at low quality (blocking + ringing)."""
    import io

    from PIL import Image

    g = to_gray(img)
    buf = io.BytesIO()
    g.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as re:
        return re.convert("L")


def fax_binarize(img, threshold: int | None = None):
    """Fax-style 1-bit thresholding. `threshold=None` uses Otsu (histogram split);
    returned as an "L" image (0/255) so downstream ops keep one mode."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(to_gray(img), dtype=np.uint8)
    thr = _otsu_threshold(arr) if threshold is None else int(threshold)
    out = np.where(arr >= thr, 255, 0).astype("uint8")
    return Image.fromarray(out, mode="L")


def _otsu_threshold(arr) -> int:
    """Otsu's method — canonical implementation now lives in `clean.coverage`
    (shared with the ink-coverage signal); re-exported here for back-compat."""
    from contract_rag.clean.coverage import otsu_threshold

    return otsu_threshold(arr)


def gaussian_noise(img, sigma: float, rng):
    """Additive sensor noise ~ N(0, sigma). rng is a numpy RandomState (seeded)."""
    import numpy as np
    from PIL import Image

    g = to_gray(img)
    if sigma <= 0:
        return g
    arr = np.asarray(g, dtype=np.float32)
    noise = rng.normal(0.0, float(sigma), arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype("uint8"), mode="L")


def salt_pepper(img, amount: float, rng):
    """Speckle: a fraction `amount` of pixels flipped to pure black or white."""
    import numpy as np
    from PIL import Image

    g = to_gray(img)
    if amount <= 0:
        return g
    arr = np.asarray(g, dtype=np.uint8).copy()
    mask = rng.random(arr.shape)
    arr[mask < amount / 2] = 0
    arr[mask > 1 - amount / 2] = 255
    return Image.fromarray(arr, mode="L")


def bleed_through(img, alpha: float):
    """Faint mirror image of the page's own ink showing through the paper from the back.
    Deterministic: horizontally-flipped copy, lightened by (1-alpha) toward white, then
    `darker`-composited so it only adds faint gray where the page is otherwise blank."""
    from PIL import Image, ImageChops

    g = to_gray(img)
    if alpha <= 0:
        return g
    mirror = g.transpose(Image.FLIP_LEFT_RIGHT)
    a = float(alpha)
    faint = mirror.point(lambda px: int(255 - (255 - px) * a))
    return ImageChops.darker(g, faint)


# ============================================================ intensity presets

class DegradeParams(BaseModel):
    """One named intensity. Operators run in physical-pipeline order:
    bleed-through → skew → downscale → gaussian → salt-pepper → binarize → jpeg."""

    bleed_alpha: float = 0.0
    skew_deg: float = 0.0
    downscale: float | None = None
    gaussian_sigma: float = 0.0
    salt_pepper: float = 0.0
    binarize: bool = False
    binarize_threshold: int | None = None
    jpeg_quality: int | None = None


# Calibrated on rendered CUAD digital pages (see Implementation status in CLAUDE.md):
# "light" barely dents OCR; "medium" is a realistic office scan; "fax" is a degraded
# low-DPI fax/photocopy — the level that actually drives quality toward the 0.4 target.
LEVELS: dict[str, DegradeParams] = {
    "light": DegradeParams(
        skew_deg=0.4, downscale=0.7, gaussian_sigma=6.0, jpeg_quality=55
    ),
    "medium": DegradeParams(
        bleed_alpha=0.10, skew_deg=1.0, downscale=0.45, gaussian_sigma=14.0,
        salt_pepper=0.01, jpeg_quality=32,
    ),
    "fax": DegradeParams(
        bleed_alpha=0.16, skew_deg=1.6, downscale=0.32, gaussian_sigma=22.0,
        salt_pepper=0.03, binarize=True, jpeg_quality=22,
    ),
    # "shred" is a deliberately brutal stress level used to probe the quality FLOOR
    # (how low can image degradation drive the score at all) — very low DPI so glyphs
    # merge, heavy noise, no binarize (thresholding would *clean* the page). See the
    # Implementation-status calibration: even this can't reach 0.4 on digital contracts,
    # because OCR emits confident-but-wrong text, not garble/empty.
    "shred": DegradeParams(
        bleed_alpha=0.22, skew_deg=2.6, downscale=0.16, gaussian_sigma=40.0,
        salt_pepper=0.08, jpeg_quality=10,
    ),
}


def degrade_image(img, seed: int = 0, level: str = "medium", params: DegradeParams | None = None):
    """Compose the operators at a named intensity. Seeded + deterministic: one numpy
    RandomState(seed) threads through the two stochastic ops in a fixed order, so the
    same (image, seed, level) always yields byte-identical output. Pure — never mutates
    the input image."""
    import numpy as np

    p = params or LEVELS[level]
    rng = np.random.RandomState(seed)
    out = to_gray(img)
    out = bleed_through(out, p.bleed_alpha)
    out = skew(out, p.skew_deg)
    if p.downscale is not None:
        out = downscale_upscale(out, p.downscale)
    out = gaussian_noise(out, p.gaussian_sigma, rng)
    out = salt_pepper(out, p.salt_pepper, rng)
    if p.binarize:
        out = fax_binarize(out, p.binarize_threshold)
    if p.jpeg_quality is not None:
        out = jpeg_recompress(out, p.jpeg_quality)
    return out


# ============================================================ PDF render / degrade seam

def render_pdf_pages(pdf_path: Path, dpi: int = _RENDER_DPI, max_pages: int = 0) -> list:
    """Rasterize a (clean, digital) PDF to grayscale page images via pypdfium2."""
    import pypdfium2 as pdfium

    images = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i, page in enumerate(pdf):
            if max_pages and i >= max_pages:
                page.close()
                break
            try:
                images.append(page.render(scale=dpi / 72).to_pil().convert("L"))
            finally:
                page.close()
    finally:
        pdf.close()
    if not images:
        raise ValueError(f"no pages to render in {pdf_path}")
    return images


def images_to_pdf(images: list, out_pdf: Path, dpi: int = _RENDER_DPI) -> Path:
    """Multi-page PDF at `dpi` (page points = px * 72/dpi), so a later paddle render at
    300 dpi upscales exactly as a real low-DPI scan would."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    rgb = [im.convert("RGB") for im in images]
    rgb[0].save(
        out_pdf, "PDF", resolution=float(dpi), save_all=True, append_images=rgb[1:]
    )
    return out_pdf


def degrade_pdf(
    src_pdf: Path,
    out_pdf: Path,
    seed: int = 0,
    level: str = "medium",
    dpi: int = _RENDER_DPI,
    max_pages: int = 0,
) -> Path:
    """Clean digital PDF → per-page degraded images → degraded (text-layerless) PDF.
    Each page is degraded with a page-offset seed so no two pages share a noise field."""
    pages = render_pdf_pages(src_pdf, dpi=dpi, max_pages=max_pages)
    degraded = [degrade_image(im, seed=seed + i, level=level) for i, im in enumerate(pages)]
    return images_to_pdf(degraded, out_pdf, dpi=dpi)


# ============================================================ measurement harness

class ColumnQuality(BaseModel):
    quality: QualityReport
    n_blocks: int
    field_f1: float | None = None
    source_accuracy: float | None = None


class DocResult(BaseModel):
    name: str
    page_count: int
    original: ColumnQuality
    degraded: ColumnQuality
    cleaned: ColumnQuality


def _column(ir: DocumentIR, f1: float | None, src: float | None) -> ColumnQuality:
    return ColumnQuality(
        quality=compute_quality_score(ir),
        n_blocks=len(ir.blocks),
        field_f1=f1,
        source_accuracy=src,
    )


def evaluate_doc(
    name: str,
    original_ir: DocumentIR,
    degraded_ir: DocumentIR,
    *,
    page_count: int,
    gold=None,
    extractor=None,
    vertical=None,
    clean_fn: Callable[[DocumentIR], DocumentIR] = clean_ir,
) -> DocResult:
    """Pure scoring of the three columns for one doc. If a gold doc + extractor are
    supplied, each column also gets an F1/source-accuracy pair (single-doc aggregate)."""
    cleaned_ir = clean_fn(degraded_ir)

    def _f1(ir: DocumentIR) -> tuple[float | None, float | None]:
        if gold is None or extractor is None:
            return None, None
        from contract_rag.eval.metrics import aggregate, row_for

        row = row_for(extractor.extract(ir), gold, ir, vertical)
        agg = aggregate([row], vertical)
        return round(agg["field_f1"], 3), round(agg["source_accuracy"], 3)

    return DocResult(
        name=name,
        page_count=page_count,
        original=_column(original_ir, *_f1(original_ir)),
        degraded=_column(degraded_ir, *_f1(degraded_ir)),
        cleaned=_column(cleaned_ir, *_f1(cleaned_ir)),
    )


class ColumnSummary(BaseModel):
    mean_quality: float
    mean_garble: float
    mean_confidence: float
    needs_review_rate: float
    field_f1: float | None = None
    source_accuracy: float | None = None


class Summary(BaseModel):
    n_docs: int
    level: str
    seed: int
    render_dpi: int
    original: ColumnSummary
    degraded: ColumnSummary
    cleaned: ColumnSummary


def _column_summary(cols: list[ColumnQuality], f1: float | None, src: float | None) -> ColumnSummary:
    n = len(cols)
    return ColumnSummary(
        mean_quality=round(sum(c.quality.quality_score for c in cols) / n, 3),
        mean_garble=round(sum(c.quality.garble_ratio for c in cols) / n, 3),
        mean_confidence=round(sum(c.quality.mean_confidence for c in cols) / n, 3),
        needs_review_rate=round(sum(c.quality.needs_review for c in cols) / n, 3),
        field_f1=f1,
        source_accuracy=src,
    )


def _macro_f1(cols: list[ColumnQuality]) -> tuple[float | None, float | None]:
    f1s = [c.field_f1 for c in cols if c.field_f1 is not None]
    srcs = [c.source_accuracy for c in cols if c.source_accuracy is not None]
    f1 = round(sum(f1s) / len(f1s), 3) if f1s else None
    src = round(sum(srcs) / len(srcs), 3) if srcs else None
    return f1, src


def summarize(results: list[DocResult], level: str, seed: int, render_dpi: int) -> Summary:
    if not results:
        raise ValueError("no documents evaluated")
    orig = [r.original for r in results]
    degr = [r.degraded for r in results]
    clnd = [r.cleaned for r in results]
    return Summary(
        n_docs=len(results),
        level=level,
        seed=seed,
        render_dpi=render_dpi,
        original=_column_summary(orig, *_macro_f1(orig)),
        degraded=_column_summary(degr, *_macro_f1(degr)),
        cleaned=_column_summary(clnd, *_macro_f1(clnd)),
    )


def format_report(results: list[DocResult], summary: Summary) -> str:
    def _f(x: float | None, w: int = 5, p: int = 2) -> str:
        return f"{x:.{p}f}" if x is not None else "—"

    lines = [
        f"=== image-level degradation (level={summary.level} seed={summary.seed}"
        f" render_dpi={summary.render_dpi}) ===",
        f"{'doc':<30} {'pg':>3} {'orig':>6} {'degr':>6} {'clean':>6}"
        f" {'oF1':>5} {'dF1':>5} {'cF1':>5}",
    ]
    for r in results:
        lines.append(
            f"{r.name[:30]:<30} {r.page_count:>3}"
            f" {r.original.quality.quality_score:>6.3f}"
            f" {r.degraded.quality.quality_score:>6.3f}"
            f" {r.cleaned.quality.quality_score:>6.3f}"
            f" {_f(r.original.field_f1):>5} {_f(r.degraded.field_f1):>5}"
            f" {_f(r.cleaned.field_f1):>5}"
        )
    lines += [
        "",
        f"docs={summary.n_docs}",
        f"{'column':<10} {'quality':>8} {'garble':>7} {'conf':>6} {'review%':>8}"
        f" {'field_f1':>9} {'src_acc':>8}",
    ]
    for label, side in (
        ("original", summary.original),
        ("degraded", summary.degraded),
        ("cleaned", summary.cleaned),
    ):
        lines.append(
            f"{label:<10} {side.mean_quality:>8.3f} {side.mean_garble:>7.3f}"
            f" {side.mean_confidence:>6.3f} {side.needs_review_rate:>8.1%}"
            f" {_f(side.field_f1, p=3):>9} {_f(side.source_accuracy, p=3):>8}"
        )
    return "\n".join(lines)


# ============================================================ impure runner

def _page_count(pdf_path: Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def truncate_ir_to_pages(ir: DocumentIR, max_pages: int) -> DocumentIR:
    """Keep only blocks on pages 1..max_pages (1-indexed bbox.page); no-bbox blocks are
    kept (they're doc-level). Pure — used to page-match the full-doc `original` IR to the
    degraded IR when the harness only degraded the first `max_pages`, so the field-F1
    comparison isn't confounded by page truncation (a fact on page 40 the degraded side
    never saw would otherwise read as a degradation loss)."""
    if max_pages <= 0:
        return ir
    kept = [b for b in ir.blocks if b.bbox is None or b.bbox.page <= max_pages]
    return ir.model_copy(update={"blocks": kept})


def run_degrade(
    settings: Settings,
    cache_dir: Path,
    *,
    level: str = "medium",
    seed: int = 0,
    cap: int = 6,
    max_pages: int = 0,
    render_dpi: int = _RENDER_DPI,
    with_facts: bool = True,
) -> tuple[list[DocResult], Summary]:
    """Render clean digital CUAD PDFs, degrade, OCR, and score three columns per doc.

    Originals are docling-parsed (IR-cached under `.ir_cache`). Degraded PDFs and their
    paddle IRs are cached by (stem, level, seed) so re-runs are fast. Extraction F1 is
    computed per column when a golden set + a runnable extractor are available."""
    from contract_rag.eval.golden import load_golden_set
    from contract_rag.eval.ir_cache import ir_cache
    from contract_rag.parse.docling_parser import parse_with_docling
    from contract_rag.parse.router import parse as router_parse

    cache_dir = Path(cache_dir)
    golden = load_golden_set(settings.golden_set_dir)  # always: it maps doc → source_pdf
    if not golden:
        raise SystemExit(
            f"no golden set in {settings.golden_set_dir}; build one with "
            "`python -m contract_rag.eval.cuad` (needs CUAD_DIR)."
        )
    golden = golden[: max(0, cap)]

    extractor = vertical = None
    if with_facts:
        try:
            from contract_rag.extract.extractor import get_extractor
            from contract_rag.verticals.registry import get_vertical_for

            vertical = get_vertical_for(settings)
            extractor = get_extractor(settings, vertical)
        except Exception:  # missing creds/deps → quality-only run
            extractor = vertical = None

    docling_cache = ir_cache(Path(".ir_cache"), parse_with_docling)
    degraded_cache = ir_cache(
        cache_dir / "ir" / f"{level}_s{seed}", lambda p: router_parse(p, settings)
    )
    degraded_pdf_dir = cache_dir / "pdf" / f"{level}_s{seed}"

    results: list[DocResult] = []
    for g in golden:
        src_pdf = settings.data_dir / g.source_pdf
        if not src_pdf.exists():
            continue
        original_ir = docling_cache(src_pdf)
        if max_pages:  # page-match the full-doc original so F1 isn't truncation-confounded
            original_ir = truncate_ir_to_pages(original_ir, max_pages)
        out_pdf = degraded_pdf_dir / f"{Path(g.source_pdf).stem}.pdf"
        if not out_pdf.exists():
            degrade_pdf(src_pdf, out_pdf, seed=seed, level=level,
                        dpi=render_dpi, max_pages=max_pages)
        degraded_ir = degraded_cache(out_pdf)
        results.append(
            evaluate_doc(
                Path(g.source_pdf).stem,
                original_ir,
                degraded_ir,
                page_count=_page_count(src_pdf),
                gold=g if extractor else None,
                extractor=extractor,
                vertical=vertical,
            )
        )
    if not results:
        raise SystemExit("no documents evaluated (no source PDFs found in data_dir)")
    return results, summarize(results, level, seed, render_dpi)


def main() -> None:
    import json
    import os

    from contract_rag.config import get_settings

    settings = get_settings()
    level = os.environ.get("DEGRADE_LEVEL", "medium")
    if level not in LEVELS:
        raise SystemExit(f"DEGRADE_LEVEL must be one of {sorted(LEVELS)}; got {level!r}")
    seed = int(os.environ.get("DEGRADE_SEED", "0"))
    cap = int(os.environ.get("DEGRADE_SET_SIZE", "6"))
    max_pages = int(os.environ.get("DEGRADE_MAX_PAGES", "0"))
    render_dpi = int(os.environ.get("DEGRADE_RENDER_DPI", str(_RENDER_DPI)))
    cache = Path(
        os.environ.get("DEGRADE_CACHE", str(Path.home() / ".cache" / "contract-rag" / "degrade"))
    )
    results, summary = run_degrade(
        settings, cache, level=level, seed=seed, cap=cap,
        max_pages=max_pages, render_dpi=render_dpi,
    )
    print(format_report(results, summary))
    out = os.environ.get("DEGRADE_OUT")
    if out:
        payload = {
            "summary": summary.model_dump(),
            "results": [r.model_dump() for r in results],
        }
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
