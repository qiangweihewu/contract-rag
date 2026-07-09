+++
title = "We benchmarked the hyped new OCR engine — and kept the old one"
description = "FrankenOCR (a pure-Rust, CPU-only 3B-VLM OCR wrapper) vs PaddleOCR on our own harness: extraction parity (field-F1 {{ fr_f1_light }} vs {{ fr_paddle_f1_light }}, n={{ fr_degrade_n }} = noise), {{ fr_slowdown }} slower ({{ fr_sec_per_page }}/page vs {{ fr_paddle_sec_per_page }}), and a failure mode worse than garble: confident structured hallucination on unreadable pages that our quality score reads as {{ fr_quality_shred }}."
lang = "en"
slug = "franken-ocr"
target_queries = [
  "FrankenOCR vs PaddleOCR",
  "FrankenOCR benchmark",
  "VLM OCR hallucination",
  "how to evaluate an OCR engine",
]
[[faq]]
q = "Is FrankenOCR better than PaddleOCR for document pipelines?"
a = "Not on our benchmark. On degraded contract pages the two reach extraction parity (field-F1 {{ fr_f1_light }} vs {{ fr_paddle_f1_light }}, n={{ fr_degrade_n }} — a noise-level difference), but FrankenOCR runs {{ fr_slowdown }} slower on CPU ({{ fr_sec_per_page }}/page vs {{ fr_paddle_sec_per_page }}) and, on unreadable pages, hallucinates confident structure instead of emitting visible garble. We kept it as an opt-in experiment behind an env var and stayed on PaddleOCR."
[[faq]]
q = "Do VLM-based OCR engines hallucinate?"
a = "Yes, and the failure mode is worse than classic OCR garble: on a deliberately unreadable page, the VLM emitted empty table skeletons and a {{ fr_hallucination_size }} repeating list loop that took {{ fr_hallucination_decode_time }} to decode — fluent, well-formed, and entirely invented. Our document quality score read that output as {{ fr_quality_shred }} while PaddleOCR's visible garble on the same page scored {{ fr_paddle_quality_shred }}. Fluent hallucination defeats quality signals that garble triggers."
[[faq]]
q = "How should I evaluate a new OCR engine for my pipeline?"
a = "Wrap it behind the same document representation as your incumbent so everything downstream is identical, then benchmark on your own corpus including worst-case pages — measuring end-task accuracy (extraction F1, not just text similarity), speed per page on your hardware, and the failure mode on unreadable input. Demo screenshots show the best case; the failure mode is what you operate."
[[howto]]
step = "Install the engine under test: FrankenOCR ({{ fr_version }}, MIT, github.com/Dicklesworthstone/franken_ocr) — a {{ fr_binary_size }} binary that downloads {{ fr_weights_size }} of model weights on first run"
[[howto]]
step = "The adapter is committed in this repo: src/contract_rag/parse/franken_parser.py, opt-in via FRANKEN_BIN=path/to/focr (with it unset, the parse router is byte-identical to before)"
[[howto]]
step = "Run the benchmark: FRANKEN_BIN=path/to/focr uv run python scripts/benchmark_franken.py (realscan + degrade arms; needs the Tobacco800 scans and a CUAD golden set — datasets are never committed; OCR outputs are IR-cached)"
+++

# We benchmarked the hyped new OCR engine — and kept the old one

**TL;DR:** FrankenOCR — a pure-Rust, CPU-only wrapper around a 3B-parameter OCR vision-language model, and the subject of a wave of market hype — went onto our own benchmark harness against PaddleOCR, our incumbent scanned-document engine. Result: **extraction parity** (field-F1 **{{ fr_f1_light }}** vs **{{ fr_paddle_f1_light }}** on degraded contracts, n={{ fr_degrade_n }} — noise), **{{ fr_slowdown }} slower** ({{ fr_sec_per_page }}/page vs {{ fr_paddle_sec_per_page }} on the same CPU), and a failure mode we consider *worse* than classic OCR garble: **confident structured hallucination on unreadable pages** — which our quality score reads as a perfect {{ fr_quality_shred }}. The adapter stays in the repo as an opt-in experiment; the default engine does not change. Every number is a point-in-time measurement (**{{ fr_measured_date }}**), committed in `content/franken_results.toml`, harness included.

## Why we tested it at all

FrankenOCR has genuinely attractive properties, and pretending otherwise would be its own kind of dishonesty. It ships as a single **{{ fr_binary_size }}** binary with zero Python dependencies (the weights, {{ fr_weights_size }}, download on first run) — a real operations story for locked-down environments. It emits layout classification labels (`header` / `title` / `text` / `page_number` / `image`) that our coverage experiments could use. And the hype cycle around it was loud enough that "we didn't look" would have been negligence.

So we did what we do with every engine: wrote an adapter that converges to the same Document IR every other parser emits — one subprocess call per document, markdown output rebuilt into blocks, opt-in behind a `FRANKEN_BIN` env var, **byte-identical router behavior when unset** — and put it on the same harness as the incumbent. Same documents, same extractor, same metrics.

## The numbers

Two arms, run {{ fr_measured_date }}:

| arm | FrankenOCR | PaddleOCR |
|---|---|---|
| speed ({{ fr_realscan_n }} real Tobacco800 scans, CPU) | **{{ fr_sec_per_page }}/page** | {{ fr_paddle_sec_per_page }}/page |
| extraction, degrade-light ({{ fr_degrade_n }} CUAD contracts) | field-F1 {{ fr_f1_light }} | field-F1 {{ fr_paddle_f1_light }} |
| extraction, degrade-shred (unreadable) | {{ fr_f1_shred }} | {{ fr_f1_shred }} |

The extraction difference at n={{ fr_degrade_n }} is noise: read it as **parity**. The speed difference is not noise: **{{ fr_slowdown }}** on the same hardware, which at archive-migration scale is the difference between an overnight job and a quarter.

One number in the raw output looked like a FrankenOCR win and is worth debunking ourselves before anyone quotes it: source-attribution accuracy **{{ fr_srcacc_light }} vs {{ fr_paddle_srcacc_light }}**. That gap is an **artifact of block granularity**, not accuracy — FrankenOCR emits ~{{ fr_blocks_per_doc }} page-sized markdown blocks per document vs PaddleOCR's {{ fr_paddle_blocks_per_doc }} line-level blocks, and citing "somewhere in this page" is simply an easier target than citing the right line. A metric that looks better because the evidence got coarser is not better.

## The failure mode that decided it

On `shred` — our deliberately unreadable degradation level, where any engine *should* fail — PaddleOCR fails honestly: it emits visible garble, and our quality score duly craters to **{{ fr_paddle_quality_shred }}**, flagging the document for review.

FrankenOCR fails differently. It emitted **fluent, well-formed, entirely invented structure**: empty `<table>` skeletons, and in one case a **{{ fr_hallucination_size }}** repeating "2. 3. 4. …" list loop that took **{{ fr_hallucination_decode_time }}** to decode. Our quality score read that output as **{{ fr_quality_shred }}** — perfect.

This is the [quality-signal blind spot we measured before](/ocr-omission.html), in a sharper form. Quality signals score *what the engine emitted*: garble triggers them; silence evades them; and fluent hallucination actively **defeats** them — it manufactures exactly the well-formed evidence the signals reward. For a pipeline whose product is *sourced, verifiable facts from legal documents*, an engine that invents structure on unreadable input is disqualifying at the default position, whatever its throughput or packaging. Garble gets caught. Hallucination gets trusted.

## Verdict, and the rule behind it

FrankenOCR stays in the repo as an **opt-in experiment** — the adapter and benchmark harness are committed, and the layout labels may yet earn their keep in coverage experiments. PaddleOCR stays the default scanned-document engine.

The transferable part is not the verdict but the procedure, which costs about a day once per candidate:

1. **Adapt to one internal representation** so the comparison isolates the engine (everything downstream identical).
2. **Benchmark on your corpus** — including the worst pages you have, not the demo's best.
3. **Measure the end task** (extraction F1), the **speed on your hardware**, and the **failure mode on unreadable input** — the last one is where the engines truly diverged.

## Honest limitations

- **Small n**: {{ fr_degrade_n }} contracts on the accuracy arm, {{ fr_realscan_n }} scans on the speed arm. Fine for a keep/switch decision at this effect size ({{ fr_slowdown }} speed, disqualifying failure mode); not a leaderboard.
- **CPU-only timing**, per FrankenOCR's own positioning. A GPU deployment would change the speed math — not the hallucination finding.
- **One version** ({{ fr_version }}, {{ fr_measured_date }}). VLM wrappers iterate fast; the harness is committed precisely so this is one command to re-run.
- **Two of our four degradation levels** (light, shred) were run — the extremes bracket the middle, but medium/fax numbers don't exist yet.

## Reproduce it yourself

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
# FrankenOCR: {{ fr_binary_size }} binary, MIT — github.com/Dicklesworthstone/franken_ocr ({{ fr_weights_size }} weights on first run)
FRANKEN_BIN=path/to/focr uv run python scripts/benchmark_franken.py
```

Needs the Tobacco800 scans and a CUAD golden set (never committed — see the repo README); OCR outputs are IR-cached so re-runs are fast. If a newer FrankenOCR version changes these numbers, run it and tell us — the harness exists so that this argument can be had with data.
