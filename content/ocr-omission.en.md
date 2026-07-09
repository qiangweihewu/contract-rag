+++
title = "OCR confidence can't detect omissions: what a near-perfect quality score hides"
description = "Measured on {{ fin_n_pages }} real degraded SEC filings: {{ fin_omission_rate }} of {{ fin_n_facts }} expert-labeled facts appear nowhere in the OCR output, while the document-level quality score reads {{ fin_quality_score }} — and no OCR confidence threshold can flag the loss, because an omitted fact produces no block at all."
lang = "en"
slug = "ocr-omission"
target_queries = [
  "why does OCR miss text",
  "detect missing text in scanned documents",
  "OCR confidence score reliability",
  "silent OCR omission",
]
[[faq]]
q = "Why does OCR miss text without reporting any error?"
a = "Because OCR quality signals score what the engine emitted, not what it skipped. On {{ fin_n_pages }} real degraded SEC pages, {{ fin_omission_rate }} of {{ fin_n_facts }} expert-labeled facts appeared nowhere in the OCR output while the document-level quality score read {{ fin_quality_score }}. A region the OCR engine never segmented produces no text, no garble, and no low-confidence block — every downstream signal is silent."
[[faq]]
q = "Can OCR confidence scores detect missing text?"
a = "No — structurally. An omitted fact produces no low-confidence block; it produces no block at all, so no confidence threshold can ever surface it. In our calibration, every gold fact's paired confidence landed in the top two bins (fact survival {{ fin_bin95_survival }} and {{ fin_bin99_survival }}), a flat reliability curve: the minimum confidence achieving ≥{{ fin_survival_target }} fact survival is unreachable."
[[faq]]
q = "How do you detect missing text in scanned documents?"
a = "With a coverage signal, not a confidence threshold: compare what is visibly on the page against what OCR produced. A geometric ink-coverage check (fraction of dark pixels falling inside no OCR block) localizes region-scale loss well — {{ cov_zone_ratio }}× higher uncovered-ink density inside occluded signature/stamp zones — but is weak for single missing numbers; that finer case needs layout-model coverage (regions a layout detector finds vs OCR blocks that fill them)."
[[howto]]
step = "Install: git clone the repo, uv sync --extra dev (paddleocr extra needed for the OCR path)"
[[howto]]
step = "Run the omission + confidence calibration: uv run python -m contract_rag.eval.fincritical (auto-downloads the gated HuggingFace dataset TheFinAI/FinCriticalED after a one-time accept; dataset files are never committed)"
[[howto]]
step = "Run the ink-coverage validation: uv run python -m contract_rag.eval.coverage (reuses the fincritical/realscan OCR caches)"
+++

# OCR confidence can't detect omissions: what a near-perfect quality score hides

**TL;DR:** We measured OCR omission against expert fact-level ground truth on **{{ fin_n_pages }}** real degraded SEC EDGAR pages (the **FinCriticalED** dataset). **{{ fin_omission_rate }} of {{ fin_n_facts }} expert-labeled facts appear nowhere in the OCR output** — silently dropped — while the document-level quality score reads **{{ fin_quality_score }}**. Worse: **no OCR confidence threshold can flag the loss**, because an omitted fact doesn't produce a low-confidence block — it produces *no block at all*. The fix is a coverage signal, not a confidence threshold; we built a geometric one and report where it works ({{ cov_zone_ratio }}× signal on occluded signature zones) and where it doesn't (single missing numbers). Every number here is a point-in-time measurement (**{{ fincritical_measured_date }}**), committed in `content/fincritical_results.toml` and reproducible with the commands at the end.

## The headline tension

One number pair, from the same {{ fin_n_pages }} pages:

| what the pipeline reports | what the ground truth shows |
|---|---|
| document quality score **{{ fin_quality_score }}** | **{{ fin_omission_rate }}** of expert-labeled facts missing from the output entirely |

The quality score is near-perfect exactly when **one in {{ fin_omission_one_in }} critical facts has vanished**. That is not a bug in one formula — it is a structural blind spot shared by essentially every OCR quality signal in production use: garble rate, empty-block rate, and confidence all score *what the engine emitted*. A fact the engine never emitted touches none of them.

We first hit this blind spot on 100 real scanned documents (Tobacco800, 1940s–90s typewritten and faxed pages), where modern OCR read even ugly scans at mean quality **{{ realscan_quality_mean }}** with zero pages flagged for review. FinCriticalED let us put a number on what that score misses, because it carries something rare: expert annotations of every critical fact on the page.

## The measurement

FinCriticalED (HuggingFace `TheFinAI/FinCriticalED`, Apache-2.0) is a set of real degraded SEC EDGAR financial pages with expert-annotated fact tags: numbers, dates, monetary units, reporting entities, financial concepts. We rendered each page, ran it through paddleocr via our parse router, and checked each gold fact for presence anywhere in the parsed output.

Canonicalization is applied identically to gold and OCR text, tolerant on formatting and strict on meaning: thousands separators and currency symbols are dropped (so `1,200,000` vs `1200000` never counts as an error), but decimal points, minus signs, and `%` are **kept** — a shifted decimal or a dropped sign *is* a critical error and must never be normalized away. Matching is token-boundary substring, so `3.5` never matches inside `13.5`.

Result: **{{ fin_omission_rate }} of {{ fin_n_facts }} gold facts appear nowhere in the OCR output**, concentrated in plain numbers ({{ fin_omission_number }}) and reporting entities ({{ fin_omission_entity }}) — the two kinds a financial-document pipeline can least afford to lose.

## Confidence cannot flag it — structurally

The obvious counter-move is a confidence threshold: route low-confidence output to human review. We calibrated exactly that — each gold fact paired with the OCR confidence of its containing block (or, when no block contains it, the block where its surrounding context landed, as a proxy for local OCR quality):

| confidence bin | fact survival | n |
|---|---|---|
| [0.95, 0.99) | {{ fin_bin95_survival }} | {{ fin_bin95_n }} |
| [0.99, 1.0) | {{ fin_bin99_survival }} | {{ fin_bin99_n }} |

Every fact's paired confidence lands in the top two bins, and survival is nearly identical in both. The reliability curve is flat: **the minimum confidence achieving ≥{{ fin_survival_target }} fact survival does not exist** on this data. You cannot buy omission-safety with a confidence floor at any price.

The reason is structural, and it generalizes beyond this dataset: an omitted fact produces **no low-confidence block — it produces no block at all**. Confidence is a property of emitted text. Omission is the absence of emitted text. A threshold on the former is, by construction, blind to the latter. (Confidence still carries *some* signal for other failure modes: on scanned pages, blocks overlapping occluded regions like signatures and stamps average {{ realscan_conf_occluded }} vs {{ realscan_conf_elsewhere }} elsewhere — real, but far too weak to route on at block granularity.)

## What works instead: a coverage signal — partially

If confidence can't see omission, what can? Something that compares *what is visibly on the page* against *what OCR produced*. We built the cheapest version that could work — geometric ink coverage, no new model dependency: render the page, Otsu-threshold it to an "ink" mask (dark pixels = visible content), and measure the fraction of ink that falls inside **no** OCR block's bounding box. That `uncovered_ink_ratio` sees exactly what confidence cannot: visible content that produced no block.

Validated against both datasets with omission/occlusion ground truth, it splits cleanly into a win and a limitation:

- **Region-scale loss: a clear win.** On pages with annotated signature/logo zones, uncovered-ink density is **{{ cov_zone_ratio }}× higher inside the zones than elsewhere** ({{ cov_zone_density_in }} vs {{ cov_zone_density_out }}), and higher on {{ cov_zone_pages }} pages ({{ cov_zone_pages_pct }}). The signal correctly localizes the ink OCR under-reads at signatures and stamps — the case that matters for "was this contract ever physically signed?"
- **Single-fact loss: too weak to rely on.** Pages with ≥1 omitted fact do carry more uncovered ink ({{ cov_fact_uncovered_omitted }} vs {{ cov_fact_uncovered_clean }}, ~{{ cov_fact_ratio }}×), but the page-level correlation is weak (point-biserial {{ cov_fact_pointbiserial }}). A dropped number is a few pixels among thousands, and OCR usually still emitted *something* nearby that covers the region.

That is an honest partial answer. Region-scale omission — occluded signatures, stamps, logos — is detectable today with geometry alone. Individual-fact omission needs a finer instrument: layout-model coverage, scoring the regions a layout detector finds against the OCR blocks that fill them. We publish the gap rather than papering over it, because the gap is the actual state of the art.

## What this means for your pipeline

- **Do not use a document-level quality score as an omission guarantee.** Ours read {{ fin_quality_score }} while {{ fin_omission_rate }} of facts were gone; yours will too, because the blind spot is in what such scores measure, not in any particular formula.
- **Do not budget human review by confidence threshold if omission is your risk.** The reliability curve is flat where it matters; the threshold you are looking for is unreachable.
- **Add a coverage check for region-scale loss now** — it is cheap (pure geometry, no model) and catches occluded signatures/stamps at {{ cov_zone_ratio }}× signal.
- **Treat single-fact omission as unsolved** at document granularity without layout-model coverage — and demand omission numbers, not quality scores, from any OCR vendor claim.

## Honest limitations

- **FinCriticalED is SEC financial pages, not commercial contracts.** The omission rate and confidence-flatness transfer as *OCR properties*; they are not contract-field accuracy numbers.
- **Known canonicalization limits, by design:** accounting-style negatives (`(5.2)` vs `-5.2`) count as different, and a value OCR-split mid-word across two lines counts as lost.
- **The confidence calibration is paddleocr-specific in its exact values** — though the structural argument (no block ⇒ no confidence to threshold) is engine-independent.
- **These are point-in-time numbers ({{ fincritical_measured_date }})**, committed as data and injected at build time, because the datasets are download-gated and never committed to the repo.

## Reproduce it yourself

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
# One-time: accept the dataset terms on the HuggingFace page (gated auto-approval), hf auth login
uv run python -m contract_rag.eval.fincritical   # omission rate + confidence reliability table
uv run python -m contract_rag.eval.coverage      # ink-coverage validation on both ground-truth sets
```

The OCR parses are cached, so re-runs are fast. If your numbers differ materially, open an issue — that is the point of measuring in public.
