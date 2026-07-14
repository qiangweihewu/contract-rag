+++
title = "The traditional OCR engine is the worst hallucinator — and the quality score can't see any of it"
description = "We put two vision-language OCR models (dots.ocr, DeepSeek-OCR) up against PaddleOCR on expert fact-level omission and a new hallucination metric. Vision-OCR wins on almost every axis — {{ vo_fin_dots_number_omitted }}/{{ vo_fin_dots_number_n }} omitted numbers, invented-ratio fail-safe on garbage — but the pre-registered adoption bar still says FAIL ({{ vo_fin_dots_omission }} vs a {{ vo_rubric_bar }} bar), and a same-day dual-engine crosscheck confirms the fix isn't as simple as running two OCR engines either ({{ vo_cc_recall }} recall)."
lang = "en"
slug = "vision-ocr"
date = "2026-07-14"
target_queries = [
  "vision language model OCR vs traditional OCR benchmark",
  "dots.ocr accuracy contracts",
  "OCR hallucination detection",
  "DeepSeek-OCR omission",
]
[[faq]]
q = "Does a vision-language OCR model omit fewer facts than PaddleOCR?"
a = "Yes, on our measurement: dots.ocr's overall FinCriticalED omission rate ({{ vo_fin_dots_omission }}) beats PaddleOCR's ({{ vo_fin_paddle_omission }}), and on the number facts that matter most to a financial or contract fact-extraction pipeline, dots.ocr dropped {{ vo_fin_dots_number_omitted }} of {{ vo_fin_dots_number_n }} versus PaddleOCR's {{ vo_fin_paddle_number }} omission rate. DeepSeek-OCR, by contrast, was worse than the traditional engine on every omission axis ({{ vo_fin_dsocr_omission }} overall, {{ vo_fin_dsocr_number }} of numbers dropped)."
[[faq]]
q = "If a vision-OCR model wins on nearly every axis, why does PaddleOCR stay the default?"
a = "Because the adoption decision was pre-registered before the numbers existed, specifically to prevent us from rationalizing a win after the fact. The bar required FinCriticalED omission at or below half of PaddleOCR's rate — {{ vo_rubric_bar }} — and dots.ocr's {{ vo_fin_dots_omission }} misses it. The rest of the picture (field-F1, fail-safe degradation, near-zero number omission) makes dots.ocr the recommended opt-in `VLM_ENDPOINT` candidate, just not the default."
[[faq]]
q = "Can a second OCR engine catch what the first one silently drops?"
a = "Not reliably, on our same-day measurement. Cross-checking PaddleOCR's output against dots.ocr's for critical digit-bearing tokens missing from the primary engine gives overall fact-level flag-recall of only {{ vo_cc_recall }} against the pre-registered {{ vo_cc_verdict }} bar — because most omitted facts in our ground truth are pure-text entity names, which a digit-token design can never see by construction. On the digit-bearing facts it was actually built to protect, it does much better: {{ vo_cc_number_caught }} omitted numbers caught, {{ vo_cc_digit_recall }} of all digit-bearing omissions, at a {{ vo_cc_false_alarm }} false-alarm rate."
[[howto]]
step = "Install: git clone the repo, uv sync --extra dev, then the OCR/VLM stack per scripts/measure_vision_ocr.py's runbook (vLLM serving dots.ocr / DeepSeek-OCR on a rented GPU)"
[[howto]]
step = "Build the OCR caches the harness scores against: uv run python -m contract_rag.eval.fincritical and uv run python -m contract_rag.eval.degrade (gated/external datasets, never committed)"
[[howto]]
step = "Run the vision-OCR measurement: uv run python scripts/measure_vision_ocr.py --model dots (then --model dsocr)"
[[howto]]
step = "Run the dual-engine crosscheck (offline, cached IRs only): uv run python -m contract_rag.eval.crosscheck"
+++

# The traditional OCR engine is the worst hallucinator — and the quality score can't see any of it

**TL;DR:** Every negative result we've published shares one root cause: PaddleOCR silently *omits* content, and no downstream signal — not confidence, not geometric ink-coverage, not layout-region coverage — reliably catches it. This time we asked a different question: does a modern vision-language OCR model, which reads a page as a whole instead of stitching detection + recognition, actually fix the omission blind spot? We measured two candidates (dots.ocr and DeepSeek-OCR) against PaddleOCR on the same FinCriticalED fact-level ground truth and the same degrade ladder, plus a brand-new hallucination metric for the failure direction a vision model is more prone to. Result: the vision-OCR model wins on nearly every axis we measured — and still fails the pre-registered adoption bar we fixed before running it. All numbers below are point-in-time measurements (**{{ vo_measured_date }}**, {{ vo_gpu }}), committed in `content/vision_ocr_results.toml` and reproducible with the commands at the end.

## Why we ran it

The pattern across every prior article in this series is the same shape: the document-level quality score reads near-perfect (**{{ fin_quality_score }}** on FinCriticalED) while **{{ fin_omission_rate }}** of expert-labeled facts vanish from the OCR output entirely. OCR confidence can't flag it (an omitted fact produces no block, not a low-confidence one). Geometric ink-coverage and layout-region coverage both catch region-scale occlusion (signatures, stamps) but not a single dropped number. Every fix we've tried treats the *symptom* — PaddleOCR's detection-then-recognition pipeline stitching regions together and dropping some.

A vision-language OCR model reads the whole page in one pass — layout, recognition, and reading order together — so it has no detection-stitching seam to drop text at. Its failure mode should be the opposite: not silent omission but confident *invention*. Both directions are measurable against ground truth we already had cached, so the outcome was going to be publishable either way: either the VLM materially fixes the omission blind spot, or it's a fifth well-measured negative result.

## Result 1: FinCriticalED omission — the vision model wins, especially on numbers

Same {{ fin_n_pages }} gold pages, {{ fin_n_facts }} expert facts, same omission scoring as the original fincritical run — now run through both vision-OCR candidates:

| engine | overall omission | number-fact omission |
|---|---|---|
| PaddleOCR (baseline) | {{ vo_fin_paddle_omission }} | {{ vo_fin_paddle_number }} |
| **dots.ocr** | **{{ vo_fin_dots_omission }}** | **{{ vo_fin_dots_number_omitted }}/{{ vo_fin_dots_number_n }}** |
| DeepSeek-OCR | {{ vo_fin_dsocr_omission }} | {{ vo_fin_dsocr_number }} |

dots.ocr beats the traditional engine on both columns, and the number-fact result is the headline: it dropped none of the {{ vo_fin_dots_number_n }} number facts — the ones a contract or financial fact-extraction pipeline depends on most. That strength is specific to numbers, not to digit-bearing facts as a whole: dots.ocr still dropped **{{ vo_fin_dots_temporal }}** of date/temporal facts, a real residual the number-only result doesn't cover. DeepSeek-OCR moves the wrong way — worse than PaddleOCR overall and dramatically worse on numbers, which we attribute to its optical-compression approach being hostile to exactly the facts that matter, compounded by a hard context-length ceiling that is itself a real serving limitation. And the blind spot that started this whole series is engine-independent: document-level quality reads **{{ vo_fin_quality_all }}** across all three engines, so the quality score cannot tell you which one is actually dropping facts.

## Result 2: the degrade ladder — and a new metric for the *other* failure direction

FinCriticalED can only measure omission (a gold fact absent from the output). The degrade ladder — clean digital CUAD pages rendered, degraded, and re-OCR'd — gives ground truth in the opposite direction too, because the original clean text is the reference. We added `invented_token_ratio`: the fraction of OCR output tokens that don't appear anywhere in the original page text, canonicalized so pure formatting (case, thousands separators, currency symbols) never counts, while a misread digit or sign does.

Field-F1 on the same {{ vo_deg_n_docs }}-doc, first-{{ vo_deg_n_pages }}-page slice used in the earlier degrade run:

| level | PaddleOCR F1 | dots.ocr F1 |
|---|---|---|
| light | {{ vo_deg_f1_light_paddle }} | {{ vo_deg_f1_light_dots }} |
| medium | {{ vo_deg_f1_medium_paddle }} | {{ vo_deg_f1_medium_dots }} |

dots.ocr wins at both levels. The invented-token metric is where the real reframe happens — it separates *confident junk* from *safe failure*: at `fax`, PaddleOCR invents **{{ vo_inv_fax_paddle }}** of its output tokens while its own quality score still reads **{{ vo_quality_fax_paddle }}**; dots.ocr invents only **{{ vo_inv_fax_dots }}** at the same level. At `shred`, PaddleOCR invents **{{ vo_inv_shred_paddle }}**, dots.ocr invents **{{ vo_inv_shred_dots }}** (it returns essentially nothing rather than confabulating), and DeepSeek-OCR invents **{{ vo_inv_shred_dsocr }}** — heavy hallucination while its own quality score reads a blind **{{ vo_quality_shred_dsocr }}**. **PaddleOCR, the "traditional" engine, is the worst hallucinator we measured, and the quality formula cannot see any of it** — this is the same blind spot as the omission story, just in the other direction.

**Caveat on the metric itself:** on sparse-cover documents — where even the original clean digital text on those {{ vo_deg_n_pages }} pages is nearly empty — the invented-token ratio is inflated identically for every engine, because a short reference text makes almost any output token count as "not in the reference." The `light`-level absolutes above sit in a range dominated by this reference-bias effect; the per-doc values on dense-text pages are close to zero. Treat the *deltas between engines* as meaningful and the *absolute numbers* as reference-bias-sensitive.

One more operational finding shaped the harness itself: run uncapped, both vision-OCR models **repetition-loop** on badly degraded pages — generating **{{ vo_loop_tokens }}** junk tokens and taking **{{ vo_loop_minutes }}** minutes per page before we added a hard generation cap. A production VLM-OCR deployment needs that cap; it is not optional.

## The pre-registered rubric: missed

Before any of the numbers above existed, we fixed an adoption bar: the vision model becomes the scanned-route default only if its FinCriticalED omission rate is at or below **half** of PaddleOCR's — {{ vo_rubric_bar }}. dots.ocr's measured omission rate is {{ vo_fin_dots_omission }}, which is above that bar. **Verdict: missed.** PaddleOCR stays the default scanned-route engine; the VLM route stays opt-in. Latency backs this up too — {{ vo_latency_vlm }} on a GPU versus PaddleOCR's {{ vo_latency_paddle }} on CPU is a real cost, not a rounding error, for a bar that wasn't even cleared. But the honest picture is not "don't use it": on field-F1, fail-safe degradation behavior, and near-zero number-fact omission, dots.ocr is the strongest candidate we've measured for an opt-in `VLM_ENDPOINT`. DeepSeek-OCR is not recommended for this vertical on any axis.

## Result 3: can a second engine catch what the first one drops?

The natural next question, given dots.ocr's number-fact strength: use it as a *verifier* rather than a replacement. Run PaddleOCR as primary, diff its output against dots.ocr's for digit-bearing "critical tokens" (numbers, amounts, dates, percentages) that are missing from the primary — flag the page for human review when they don't match. We pre-registered a bar for this too, before running it: {{ vo_cc_bar }}.

Measured offline against the same cached FinCriticalED IRs: overall flag-recall **{{ vo_cc_recall }}**, false-alarm rate **{{ vo_cc_false_alarm }}**. **Verdict: {{ vo_cc_verdict }}** on the overall bar ({{ vo_cc_bar }}) — well short of the recall side of it. The honest reason is a design limit, not a bug: {{ vo_cc_entity_caught }} entity-name omissions were caught, because the digit-token cross-check can never see a pure-text entity name by construction — entity-name omissions are the single largest category, out of {{ vo_cc_n_omissions }} total omissions in the dataset. But on the digit-bearing facts the check exists to protect, it works: it caught **{{ vo_cc_number_caught }}** omitted number facts and **{{ vo_cc_digit_recall }}** of all digit-bearing omissions, at that same {{ vo_cc_false_alarm }} false-alarm rate. **The dual-engine crosscheck is a usable digit-fact safety net, not a general omission detector** — the same "region-scale vs fact-level, geometric-only signals can't see values" lesson from the coverage work, applied to a second OCR engine instead of a geometric ratio.

## Honest limitations

- **FinCriticalED is SEC financial pages, not commercial contracts.** The omission-rate results (Result 1) and the crosscheck recall/false-alarm numbers (Result 3) are both measured on FinCriticalED; they transfer as *OCR properties*, not as contract-field accuracy numbers. Only the degrade-ladder results (Result 2) run on contract pages (CUAD).
- **DeepSeek-OCR-2 → v1 substitution.** The design targeted DeepSeek-OCR 2; at run time its `DeepseekOCR2ForCausalLM` architecture had no vLLM build compatible with the rig's CUDA 12.8 driver, so we substituted DeepSeek-OCR v1. The v1 numbers above are not evidence about OCR-2's specific architecture.
- **Degradation is simulated.** The degrade ladder is a controlled, seeded stress test (downscale/upscale, skew, JPEG recompression, binarize, noise) — not dirt collected in the wild. It's a calibration instrument, like `dirtify`.
- **Invented-ratio absolutes carry reference bias.** Sparse-cover documents inflate the ratio for every engine identically (see above) — compare engines to each other at a given level, not the raw numbers across levels.
- **Repetition loops are real and uncapped generation is not production-safe.** {{ vo_loop_tokens }} tokens / {{ vo_loop_minutes }} minutes per page happened without a generation cap; any VLM-OCR deployment needs one from day one.
- **The crosscheck bar is a formal FAIL on the metric it was pre-registered against**, even though the digit-fact split is a real positive result — we report both, in that order, deliberately.
- **One GPU rig, one run per model.** No repeated-run variance estimate for the vision-OCR numbers (unlike some of our LLM-extraction measurements, which do carry bootstrap CIs).
- **Point-in-time numbers ({{ vo_measured_date }})**, committed as data and injected at build time, because the GPU rig is rented-and-terminated and the datasets are download-gated and never committed.

## Reproduce it yourself

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
# Build the OCR caches first (gated/external datasets — see the OCR-omission article):
uv run python -m contract_rag.eval.fincritical
uv run python -m contract_rag.eval.degrade
# Vision-OCR measurement needs a served VLM endpoint (see scripts/measure_vision_ocr.py's RUNBOOK):
uv run python scripts/measure_vision_ocr.py --model dots
uv run python scripts/measure_vision_ocr.py --model dsocr
# Dual-engine crosscheck is fully offline once both IR caches exist:
uv run python -m contract_rag.eval.crosscheck
```

If your numbers differ materially, open an issue — negative results only stay useful while they stay true.
