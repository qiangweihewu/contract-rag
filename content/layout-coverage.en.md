+++
title = "We built the recommended fix for single-fact OCR omission. It didn't work — and the reason is worth knowing."
description = "Layout-model coverage — scoring the regions a layout detector finds against the OCR blocks that fill them — measured on the same two ground-truth datasets as our geometric baseline: an {{ layout_gedi_ratio }}× occlusion signal on annotated signature/logo zones (the sharpest we've measured), but point-biserial {{ layout_fin_pointbiserial }} on single-fact omission, below the {{ cov_fact_pointbiserial }} geometric baseline it was built to beat."
lang = "en"
slug = "layout-coverage"
date = "2026-07-12"
target_queries = [
  "layout detection OCR omission",
  "detect missing text OCR layout model",
  "OCR coverage signal document layout analysis",
  "layout model vs OCR blocks missing regions",
]
[[faq]]
q = "Can a layout-detection model find text that OCR silently dropped?"
a = "Only at region scale. Scoring layout-detector regions against the OCR blocks that fill them is an excellent occlusion detector — uncovered-region rate {{ layout_gedi_uncov_in }} inside annotated signature/logo zones vs {{ layout_gedi_uncov_out }} elsewhere ({{ layout_gedi_ratio }}×) — but it does not reliably flag a single dropped number: point-biserial {{ layout_fin_pointbiserial }} against expert fact-level ground truth, below the {{ cov_fact_pointbiserial }} of a far cheaper geometric ink-coverage check."
[[faq]]
q = "Why doesn't region-level coverage detect a single missing number?"
a = "Because a dropped number usually sits inside a region OCR still partially filled. The layout detector draws a text region; OCR reads most of it and silently drops one value; the region's fill ratio stays above any reasonable threshold, so the region counts as covered. Finer geometry cannot fix this — the failure is that partial fill looks like full fill at every geometric granularity."
[[faq]]
q = "What actually detects single-fact OCR omission?"
a = "Nothing coverage-shaped, in our measurements. Both the geometric ink signal and the layout-model signal are region-scale occlusion detectors, not fact-level omission detectors. The honest remaining candidates are value-level checks: numeric cross-verification (totals that must sum, dates that must parse), a second OCR pass over low-fill regions, or comparing two independent engines — verifying the values, not the geometry."
[[howto]]
step = "Install: git clone the repo, uv sync --extra dev, then add the OCR/layout stack: uv pip install paddleocr paddlepaddle"
[[howto]]
step = "Build the OCR caches the harness scores against: uv run python -m contract_rag.eval.fincritical and uv run python -m contract_rag.eval.realscan (gated/external datasets, never committed)"
[[howto]]
step = "Run the layout-coverage validation: uv run python -m contract_rag.eval.layout_coverage (layout inference is disk-cached per page, so re-runs are fast)"
+++

# We built the recommended fix for single-fact OCR omission. It didn't work — and the reason is worth knowing.

**TL;DR:** In [our OCR-omission article](/ocr-omission.html) we showed that {{ fin_omission_rate }} of expert-labeled facts vanish from OCR output while the document quality score reads {{ fin_quality_score }}, and that a geometric ink-coverage check catches region-scale loss but not single missing numbers (point-biserial {{ cov_fact_pointbiserial }}). We named the obvious finer instrument: **layout-model coverage** — score the regions a layout detector finds against the OCR blocks that fill them. We built it and measured it on the same two ground-truth datasets. Result: it is the **sharpest occlusion detector we have measured** ({{ layout_gedi_ratio }}× separation on annotated signature/logo zones, vs the geometric signal's {{ cov_zone_ratio }}×) — and it **fails the single-fact test it was built for**: point-biserial **{{ layout_fin_pointbiserial }}**, *below* the geometric baseline's {{ cov_fact_pointbiserial }}. Every number is a point-in-time measurement (**{{ layout_measured_date }}**), committed in `content/layout_results.toml` and reproducible with the commands at the end.

## The hypothesis

The geometric signal's weakness had a clean explanation: a dropped financial number is a few pixels among thousands, so it barely dents a whole-page uncovered-ink ratio. The natural fix is granularity. Instead of asking "is the ink on this *page* accounted for?", ask "is each *region* a layout detector finds actually filled by OCR blocks?" A region-sized question should notice a region-sized hole.

So we built it: PaddleOCR's `LayoutDetection` (PP-DocLayout_plus-L) proposes layout regions per page; each region is scored by the fraction of its area covered by OCR block bounding boxes (`fill_ratio`); a region below the fill threshold counts as uncovered, and the document's `layout_omission_score` is the fraction of regions left unfilled. Inference is disk-cached per page, and the signal is additive — two new optional fields on the quality report, with the quality score itself byte-for-byte unchanged.

## Result 1: the sharpest occlusion signal we've measured

On the {{ layout_gedi_pages }} real scanned pages (Tobacco800) with expert-annotated signature/logo zones:

| | inside annotated zones | elsewhere |
|---|---|---|
| mean region fill ratio | **{{ layout_gedi_fill_in }}** | {{ layout_gedi_fill_out }} |
| uncovered-region rate | **{{ layout_gedi_uncov_in }}** | {{ layout_gedi_uncov_out }} |

That uncovered-region separation is **{{ layout_gedi_ratio }}×** — roughly double the geometric ink signal's {{ cov_zone_ratio }}×, and the in-zone fill is lower on {{ layout_gedi_pages_lower }} pages ({{ layout_gedi_pages_pct }}). For the question that matters in contract-archive migration — [was this document ever physically signed](/signature-detection.html), is a stamp being silently dropped — layout regions are the right granularity to route human review to, and clearly better than anything we had before.

If that were the headline, this would be a victory post. It isn't, because the occlusion case wasn't the open problem.

## Result 2: the fact-level gap does not close

The open problem was single-fact omission — the {{ fin_omission_rate }} of expert-labeled facts that vanish silently. Same measurement as the geometric run, on the same {{ layout_fin_n_pages }} degraded SEC pages ({{ layout_fin_pages_omitted }} carry at least one omitted gold fact):

| signal | point-biserial vs has-omission | verdict |
|---|---|---|
| geometric ink coverage (baseline) | {{ cov_fact_pointbiserial }} | weak |
| layout-model coverage (the "fix") | **{{ layout_fin_pointbiserial }}** | **weaker** |

The mean separation actually *widens* — pages with an omitted fact average layout-omission {{ layout_fin_mean_omitted }} vs {{ layout_fin_mean_clean }} on clean pages (~{{ layout_fin_mean_ratio }}×, vs the geometric ~{{ cov_fact_ratio }}×) — but the per-page correlation is noise-dominated (pearson {{ layout_fin_pearson }}). A wider mean gap with a worse correlation means the signal fires on the wrong pages too often to route on.

## Why finer geometry cannot fix this

The refutation is more useful than the number, because it sharpens the mechanism. A dropped number does not usually live in a region the layout detector flags as empty. It lives in a region OCR **partially** filled: the detector draws a text block, OCR reads most of it, and one value inside it is silently gone. The region's fill ratio stays above any reasonable threshold, so the region counts as covered.

That kills the whole family of fixes, not just this one. Partial fill looks like full fill at *every* geometric granularity — page, region, or line — because the failure is not spatial. The missing information is a value, and only something that reasons about values can notice its absence: numeric cross-verification (totals that must sum, percentages that must add up), a targeted second OCR pass over low-fill regions, or cross-checking two independent engines. Coverage signals — geometric or layout-model — are region-scale occlusion detectors, full stop.

## What this means for your pipeline

- **Use layout-model coverage for occlusion routing** if you already run a layout model: at {{ layout_gedi_ratio }}× separation it is the best signature/stamp-loss router we have measured. If you don't, the geometric ink check gets you {{ cov_zone_ratio }}× with no model dependency.
- **Do not buy "layout-aware omission detection" as a fact-level guarantee.** We measured the obvious version against expert fact-level ground truth and it performed *below* a pure-geometry baseline. Ask any vendor claiming otherwise for their point-biserial against fact-level truth, not a demo.
- **Treat single-fact omission as a value-verification problem**, not a coverage problem. That is where our own roadmap goes next.

## Honest limitations

- **One layout model, one detector config.** PP-DocLayout_plus-L with default thresholds; a different detector could shift the numbers, though the partial-fill mechanism is detector-independent.
- **FinCriticalED is SEC financial pages, not commercial contracts** — the transfer caveat from the original article applies unchanged.
- **The fill threshold (0.5) was not tuned.** Tuning it against the same ground truth would be overfitting the eval; we report the untuned default.
- **Point-in-time numbers ({{ layout_measured_date }})**, committed as data and injected at build time, because the datasets are download-gated and never committed.

## Reproduce it yourself

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
uv pip install paddleocr paddlepaddle
# Build the OCR caches first (gated/external datasets — see the OCR-omission article):
uv run python -m contract_rag.eval.fincritical
uv run python -m contract_rag.eval.realscan
# Then the layout-coverage validation (layout inference disk-cached per page):
uv run python -m contract_rag.eval.layout_coverage
```

If your numbers differ materially, open an issue — negative results only stay useful while they stay true.
