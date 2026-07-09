+++
title = "Was this contract ever physically signed? Detecting unsigned documents in scanned archives"
description = "A signature-presence detector for scanned documents, scored against expert zone annotations on {{ sig_n_pages }} real Tobacco800 pages: precision {{ sig_precision }}, F1 {{ sig_f1 }} — and, the actually useful part, it flags {{ sig_unsigned_flagged }} of {{ sig_unsigned_total }} unsigned documents where the always-signed assumption flags zero."
lang = "en"
slug = "signature-detection"
target_queries = [
  "detect signature in scanned document",
  "check if a contract was signed",
  "signature detection OCR",
  "unsigned contract detection archive",
]
[[faq]]
q = "How do you detect whether a scanned contract was physically signed?"
a = "Not by reading the signature — OCR can't — but from three block-level signals in the OCR output: a closing salutation (present in {{ sig_salutation_signed }} of signed documents vs {{ sig_salutation_unsigned }} of unsigned ones), explicit signature cues like /s/ or By:, and a signature block — a typed personal-name line in the lower page with a low-confidence garble token directly above it, which is what an ink signature looks like to an OCR engine. Combined, on {{ sig_n_pages }} real scanned pages: precision {{ sig_precision }}, F1 {{ sig_f1 }}."
[[faq]]
q = "Can OCR detect handwritten signatures?"
a = "Not directly — an OCR engine has no signature concept. But its failure mode on an ink signature is itself a usable signal: the engine emits a garbled low-confidence token (e.g. \"{{ sig_squiggle_text }}\" at confidence {{ sig_squiggle_conf }}) directly above the cleanly-read typed name (\"{{ sig_typed_name }}\" at {{ sig_typed_name_conf }}). That squiggle-over-name geometry recovers signed memos and forms that carry no closing salutation, at zero measured false-positive cost on our eval set."
[[faq]]
q = "How accurate is signature detection on scanned documents?"
a = "On {{ sig_n_pages }} real Tobacco800 pages ({{ sig_n_signed }} signed / {{ sig_n_unsigned }} unsigned, expert zone annotations as ground truth): precision {{ sig_precision }}, recall {{ sig_recall }}, F1 {{ sig_f1 }}. The headline is the unsigned side: {{ sig_unsigned_flagged }} of {{ sig_unsigned_total }} unsigned documents flagged with {{ sig_false_positives }} false positive, where the trivial always-signed baseline finds zero. Caveats: the heuristic is tuned to typewritten-letter conventions and recall is capped near {{ sig_recall }} because about 1 in 4 signed documents is a form or memo without a salutation."
[[howto]]
step = "Get the data: Tobacco800 page TIFFs plus the GEDI XML zone annotations (signature/logo zones; available via the Illinois Complex Document Image Processing collection — dataset files are never committed to this repo)"
[[howto]]
step = "Install: git clone the repo, uv sync --extra dev, plus paddleocr for the scanned-document path"
[[howto]]
step = "Run the eval: SIGNATURE_DIR=path/to/tobacco800/tiffs SIGNATURE_GT_DIR=path/to/gedi/xml uv run python -m contract_rag.eval.signature (prints precision/recall/F1 vs the always-signed baseline; OCR parses are IR-cached)"
+++

# Was this contract ever physically signed? Detecting unsigned documents in scanned archives

**TL;DR:** We built a signature-presence detector for scanned documents and scored it against expert zone annotations on **{{ sig_n_pages }}** real Tobacco800 pages ({{ sig_n_signed }} signed / {{ sig_n_unsigned }} unsigned): **precision {{ sig_precision }}, recall {{ sig_recall }}, F1 {{ sig_f1 }}**. The trivial baseline — assume every archived contract is signed — scores F1 {{ sig_baseline_f1 }} and finds **zero** unsigned documents; the detector flags **{{ sig_unsigned_flagged }} of {{ sig_unsigned_total }}** with {{ sig_false_positives }} false positive. The best signal is not what you'd guess: OCR *cannot read* an ink signature, and that failure is exactly what we detect. Every number is a point-in-time measurement (**{{ sig_measured_date }}**), committed in `content/signature_results.toml` and reproducible with the commands at the end.

## What a signature looks like to an OCR engine

Here is a real signature from the eval set, as paddleocr saw it — two consecutive blocks near the bottom of a 1970s typewritten letter:

| block text | OCR confidence |
|---|---|
| `{{ sig_squiggle_text }}` | {{ sig_squiggle_conf }} |
| `{{ sig_typed_name }}` | {{ sig_typed_name_conf }} |

The engine read the typed name perfectly and choked on the ink squiggle above it — emitting a garbled, low-confidence token. It has no idea a signature exists. But that *pattern* — a low-confidence garble token sitting directly above a cleanly-read personal-name line, in the lower part of the page — is what a signature reliably does to OCR output. We don't detect signatures; we detect their wreckage.

## Why anyone needs this

The question comes from contract lifecycle management: an enterprise migrating a legacy archive — thousands of scanned historical contracts — needs to know **which of them were never actually signed**. An unsigned contract in an archive is a legal and audit problem, and the default assumption ("it's in the archive, so it's executed") is silently wrong for a substantial fraction: **{{ sig_n_unsigned }} of {{ sig_n_pages }}** pages in our real-archive eval set carry no signature at all. No document-level quality score will tell you this — we measured that blind spot separately in [our OCR-omission article](/ocr-omission.html): a missing signature, like any omission, produces no block for a quality signal to score.

## Three signals, and one deliberately rejected

`detect_signature(ir)` combines three block-level signals, each designed from real Tobacco800 OCR output, combined as a probabilistic OR into a P(signed) confidence:

1. **Closing salutation** — a block matching "Sincerely / Regards / Very truly yours / …". By far the strongest: present in **{{ sig_salutation_signed }}** of signed documents vs **{{ sig_salutation_unsigned }}** of unsigned ones, with near-perfect precision. A letter that signs off was signed.
2. **Explicit signature cues** — `/s/`, `By:`, "duly authorized", "authorized signature".
3. **The signature block** — the squiggle-over-name geometry above: a typed personal-name line in the lower page with a low-confidence OCR token directly on top of it. This recovers signed memos and forms that carry *no* salutation, at zero measured false-positive cost on this set.

One tempting signal was deliberately **rejected**: per-block OCR confidence as a general occlusion detector. On real scans, blocks overlapping signature/stamp zones do average lower confidence ({{ realscan_conf_occluded }} vs {{ realscan_conf_elsewhere }}) — but unsigned faxes and telexes are just as noisy as signed letters, so at block granularity the signal barely discriminates and leaning on it hurt precision. Confidence only becomes useful when *anchored to geometry* (signal 3): low confidence directly above a name line means something; low confidence in general means it's a fax.

## Results against expert ground truth

Ground truth is the GEDI zone annotations for Tobacco800: a page counts as signed **iff** its annotation carries a signature zone. Tobacco800 annotates signatures comprehensively, so a zero-zone page is a *genuine* negative, not an unannotated one. On {{ sig_n_pages }} pages:

| | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| always-signed baseline | {{ sig_baseline_precision }} | 1.000 | {{ sig_baseline_f1 }} | — |
| signature detector | **{{ sig_precision }}** | {{ sig_recall }} | **{{ sig_f1 }}** ({{ sig_f1_delta }}) | {{ sig_accuracy }} |

The F1 delta looks modest because the baseline gets recall for free on a mostly-signed corpus. The column that matters is the one F1 doesn't show: the baseline identifies **zero** unsigned documents by construction; the detector flags **{{ sig_unsigned_flagged }} of {{ sig_unsigned_total }}**, with {{ sig_false_positives }} false positive. For the archive-migration question — "show me everything that might never have been executed" — that is the entire value, and the always-signed default provides none of it.

Each prediction carries `evidence_block_ids`, so a reviewer sees *which blocks* fired — the same source-attribution discipline as the rest of the pipeline: a claim you can't trace to evidence is a claim you can't audit.

## Honest limitations

- **The heuristic is tuned to Tobacco800's typewritten-letter conventions** — the salutation list, the name-line regex, the squiggle geometry. A different corpus (modern DocuSign PDFs, non-English archives) needs re-tuning; this is a corpus-calibrated heuristic, not a universal model.
- **Recall is capped at {{ sig_recall }}** because roughly 1 in 4 signed documents is a form or memo with no salutation and a signature the name-block geometry misses. A trained classifier, or zone-level signals at finer granularity, would lift it.
- **Block granularity**: the evidence is the cited OCR block, not the signature's pixel region.
- **These are point-in-time numbers ({{ sig_measured_date }})** on a {{ sig_n_pages }}-page set; the datasets come from external archives and are never committed to the repo.

## Reproduce it yourself

```bash
git clone <repo> && cd contract-rag && uv sync --extra dev
# Tobacco800 TIFFs + GEDI XML zone annotations: from the Illinois CDIP collection (never committed here)
SIGNATURE_DIR=path/to/tobacco800/tiffs SIGNATURE_GT_DIR=path/to/gedi/xml \
  uv run python -m contract_rag.eval.signature
```

The OCR parses are IR-cached, so re-runs are fast. The detector itself is pure logic over the parsed document — unit-tested with hand-built IRs, no OCR or network needed. If your numbers differ materially, open an issue — that is the point of measuring in public.
