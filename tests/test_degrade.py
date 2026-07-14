from __future__ import annotations

import pytest

from contract_rag.clean.quality import QualityReport
from contract_rag.eval.degrade import (
    ColumnQuality,
    DegradeParams,
    DocResult,
    LEVELS,
    evaluate_doc,
    format_report,
    summarize,
)
from contract_rag.eval.degrade import truncate_ir_to_pages
from contract_rag.ir import BlockType, BoundingBox, DocBlock, DocumentIR

# ---------------------------------------------------------------------------
# Pure image operators need PIL + numpy; skip cleanly when they're absent so the
# base unit suite still runs (mirroring the rest of the codebase's lazy imports).
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from contract_rag.eval.degrade import (  # noqa: E402
    bleed_through,
    degrade_image,
    downscale_upscale,
    fax_binarize,
    gaussian_noise,
    jpeg_recompress,
    salt_pepper,
    skew,
)


def _page(seed: int = 0) -> Image.Image:
    """A synthetic 'document' page: white background with black text-like strokes,
    so degradation has real ink to blur/threshold/mirror."""
    rng = np.random.RandomState(seed)
    arr = np.full((200, 160), 255, dtype=np.uint8)
    for _ in range(40):  # scatter black bars (fake glyphs)
        r = rng.randint(0, 190)
        c = rng.randint(0, 140)
        arr[r : r + 4, c : c + 12] = 0
    return Image.fromarray(arr, mode="L")


def _arr(img) -> "np.ndarray":
    return np.asarray(img.convert("L"), dtype=np.int16)


# ------------------------------------------------------------- determinism / seeding

def test_degrade_image_is_deterministic_for_a_seed():
    page = _page()
    a = degrade_image(page, seed=7, level="medium")
    b = degrade_image(page, seed=7, level="medium")
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_degrade_image_differs_by_seed():
    page = _page()
    a = degrade_image(page, seed=1, level="medium")
    b = degrade_image(page, seed=2, level="medium")
    # different noise field → not byte-identical
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_degrade_image_does_not_mutate_input():
    page = _page()
    before = np.asarray(page).copy()
    degrade_image(page, seed=0, level="fax")
    assert np.array_equal(np.asarray(page), before)


def test_degrade_image_changes_the_page():
    page = _page()
    out = degrade_image(page, seed=0, level="medium")
    assert out.size == page.size
    assert not np.array_equal(np.asarray(out.convert("L")), np.asarray(page))


# ------------------------------------------------------------- operator directions

def test_downscale_upscale_blurs_edges():
    page = _page()
    out = downscale_upscale(page, 0.3)
    assert out.size == page.size
    # blur softens the sharp 0/255 glyph edges → fewer pure-black pixels
    assert (_arr(out) == 0).sum() < (_arr(page) == 0).sum()


def test_downscale_noop_outside_unit_interval():
    page = _page()
    assert np.array_equal(np.asarray(downscale_upscale(page, 1.0)), np.asarray(page.convert("L")))


def test_skew_rotates_and_fills_white():
    page = _page()
    out = skew(page, 5.0)
    assert out.size == page.size
    assert not np.array_equal(np.asarray(out), np.asarray(page))
    # zero degrees is a no-op
    assert np.array_equal(np.asarray(skew(page, 0.0)), np.asarray(page.convert("L")))


def test_jpeg_recompress_introduces_intermediate_grays():
    page = _page()  # pure 0/255 bilevel
    out = jpeg_recompress(page, 20)
    vals = set(np.unique(np.asarray(out)).tolist())
    assert vals - {0, 255}  # ringing/blocking creates gray values between 0 and 255


def test_fax_binarize_is_bilevel():
    page = _page()
    noisy = gaussian_noise(page, 20.0, np.random.RandomState(0))
    out = fax_binarize(noisy)  # Otsu
    assert set(np.unique(np.asarray(out)).tolist()) <= {0, 255}


def test_fax_binarize_explicit_threshold():
    arr = np.array([[10, 200], [128, 129]], dtype=np.uint8)
    out = np.asarray(fax_binarize(Image.fromarray(arr, "L"), threshold=128))
    assert out.tolist() == [[0, 255], [255, 255]]


def test_gaussian_noise_seeded_and_adds_variation():
    page = _page()
    r1 = gaussian_noise(page, 15.0, np.random.RandomState(3))
    r2 = gaussian_noise(page, 15.0, np.random.RandomState(3))
    assert np.array_equal(np.asarray(r1), np.asarray(r2))  # seeded → reproducible
    # noise raises the std of a flat white region
    flat = Image.fromarray(np.full((50, 50), 255, np.uint8), "L")
    assert np.asarray(gaussian_noise(flat, 15.0, np.random.RandomState(0))).std() > 0
    assert np.asarray(gaussian_noise(page, 0.0, np.random.RandomState(0))).std() == pytest.approx(
        np.asarray(page.convert("L")).std()
    )


def test_salt_pepper_flips_pixels_to_extremes():
    flat = Image.fromarray(np.full((100, 100), 128, np.uint8), "L")
    out = np.asarray(salt_pepper(flat, 0.2, np.random.RandomState(0)))
    assert (out == 0).sum() > 0 and (out == 255).sum() > 0
    # zero amount is a no-op
    assert np.array_equal(np.asarray(salt_pepper(flat, 0.0, np.random.RandomState(0))), np.asarray(flat))


def test_bleed_through_only_darkens():
    page = _page()
    out = np.asarray(bleed_through(page, 0.3), dtype=np.int16)
    orig = _arr(page)
    assert (out <= orig).all()  # darker-composite never brightens
    assert (out < orig).any()   # the mirrored ghost darkens some blank pixels
    assert np.array_equal(np.asarray(bleed_through(page, 0.0)), np.asarray(page.convert("L")))


def test_levels_increase_severity():
    """Heavier levels degrade a flat-ish page more (lower mean similarity)."""
    page = _page()
    base = _arr(page)

    def _mean_abs_diff(level: str) -> float:
        return float(np.abs(_arr(degrade_image(page, seed=0, level=level)) - base).mean())

    light, medium, fax, shred = (
        _mean_abs_diff(x) for x in ("light", "medium", "fax", "shred")
    )
    assert light < medium < fax < shred


def test_all_levels_registered():
    assert set(LEVELS) == {"light", "medium", "fax", "shred"}
    assert isinstance(LEVELS["medium"], DegradeParams)


# ------------------------------------------------------------- harness aggregation
# These use hand-built IRs — no PIL/numpy path, pure scoring/aggregation.

def _ir(texts_confs: list[tuple[str, float]]) -> DocumentIR:
    return DocumentIR(
        doc_id="d", source_uri="file:///d", file_hash="h", mime_type="application/pdf",
        blocks=[
            DocBlock(block_id=f"#/b/{i}", type=BlockType.PARAGRAPH, text=t, bbox=None,
                     confidence=c, source_engine="test")
            for i, (t, c) in enumerate(texts_confs)
        ],
    )


def test_evaluate_doc_scores_three_columns_without_facts():
    original = _ir([("Clean governing law New York", 1.0), ("Body text here", 1.0)])
    # degraded: garbled + low confidence (what OCR of a bad scan looks like)
    degraded = _ir([("Ã¢â€ garbled scan", 0.4), ("   ", 0.3)])
    res = evaluate_doc("doc", original, degraded, page_count=2)
    assert res.original.quality.quality_score > res.degraded.quality.quality_score
    assert res.original.field_f1 is None  # no gold/extractor → quality-only
    assert res.degraded.n_blocks == 2


def test_evaluate_doc_cleaning_recovers_garble():
    original = _ir([("Clean text about the deal", 1.0)])
    # real utf-8→latin-1 mojibake, which fix_unicode repairs
    moji = "Agreement café clause".encode("utf-8").decode("latin-1")
    degraded = _ir([(moji, 0.6)])
    res = evaluate_doc("doc", original, degraded, page_count=1)
    assert res.cleaned.quality.quality_score >= res.degraded.quality.quality_score


def _dq(q: float) -> ColumnQuality:
    return ColumnQuality(
        quality=QualityReport(
            quality_score=q, garble_ratio=1 - q, empty_ratio=0.0, table_integrity=1.0,
            mean_confidence=q, needs_review=q < 0.75,
        ),
        n_blocks=3,
        field_f1=q,
        source_accuracy=1.0,
    )


def _doc(name: str, o: float, d: float, c: float) -> DocResult:
    return DocResult(name=name, page_count=1, original=_dq(o), degraded=_dq(d), cleaned=_dq(c))


def test_summarize_macro_averages_and_review_rate():
    results = [_doc("a", 0.9, 0.4, 0.8), _doc("b", 0.95, 0.5, 0.85)]
    s = summarize(results, level="medium", seed=0, render_dpi=150)
    assert s.n_docs == 2 and s.level == "medium"
    assert s.original.mean_quality == pytest.approx(0.925, abs=1e-3)
    assert s.degraded.mean_quality == pytest.approx(0.45, abs=1e-3)
    assert s.degraded.needs_review_rate == 1.0  # both < 0.75
    assert s.original.needs_review_rate == 0.0
    assert s.degraded.field_f1 == pytest.approx(0.45, abs=1e-3)  # macro over per-doc F1


def test_truncate_ir_to_pages_keeps_early_pages_and_nobbox():
    def blk(bid: str, page: int | None) -> DocBlock:
        return DocBlock(
            block_id=bid, type=BlockType.PARAGRAPH, text=bid,
            bbox=BoundingBox(page=page, x0=0, y0=0, x1=1, y1=1) if page else None,
            confidence=1.0, source_engine="test",
        )

    ir = DocumentIR(
        doc_id="d", source_uri="file:///d", file_hash="h", mime_type="application/pdf",
        blocks=[blk("p1", 1), blk("nobox", None), blk("p3", 3), blk("p5", 5)],
    )
    kept = {b.block_id for b in truncate_ir_to_pages(ir, 3).blocks}
    assert kept == {"p1", "nobox", "p3"}  # page 5 dropped, no-bbox kept
    # max_pages<=0 is a no-op (returns all)
    assert len(truncate_ir_to_pages(ir, 0).blocks) == 4
    # pure: original untouched
    assert len(ir.blocks) == 4


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([], level="medium", seed=0, render_dpi=150)


def test_format_report_has_columns_and_docs():
    results = [_doc("mydoc", 0.9, 0.4, 0.8)]
    out = format_report(results, summarize(results, level="fax", seed=1, render_dpi=150))
    assert "mydoc" in out
    assert "original" in out and "degraded" in out and "cleaned" in out
    assert "level=fax" in out


# ------------------------------------------------------------- invented_token_ratio

from contract_rag.eval.degrade import invented_token_ratio


def test_invented_token_ratio_formatting_only_is_zero():
    # case, thousands separators, currency symbols are canonicalized away
    assert invented_token_ratio("Total $1,200 USD", "total 1200 usd") == 0.0


def test_invented_token_ratio_counts_misread_digits():
    # OCR says 13000 where the original said 12000: 1 invented token of 2
    assert invented_token_ratio("total 13000", "total 12000") == 0.5


def test_invented_token_ratio_empty_ocr_is_zero():
    assert invented_token_ratio("", "anything at all") == 0.0


def test_evaluate_doc_populates_invented_ratio_for_degraded_and_cleaned():
    original = _ir([("alpha beta", 1.0)])
    noisy = _ir([("alpha zorp", 1.0)])
    r = evaluate_doc("d", original, noisy, page_count=1)
    assert r.original.invented_ratio is None
    assert r.degraded.invented_ratio == 0.5
    assert r.cleaned.invented_ratio is not None
