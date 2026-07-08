+++
title = "Why your contract RAG returns garbage — and a reproducible fix"
description = "A one-command, credential-free before/after benchmark: cleaning dirty contract PDFs lifts data-quality and extraction accuracy on a committed synthetic set."
lang = "en"
slug = "benchmark"
target_queries = [
  "why is my RAG returning garbage",
  "how to clean PDFs for RAG extraction",
  "RAG not production ready",
]
[[faq]]
q = "Why does a dirty PDF break RAG retrieval?"
a = "Mojibake, mid-word hyphenation, repeated page headers, and near-duplicate blocks poison both chunk retrieval and structured extraction — the model retrieves noise and cites the wrong span."
[[faq]]
q = "How much does cleaning actually help?"
a = "On a committed synthetic set of contracts, cleaning lifts the data-quality score and extraction field-F1 significantly — reproducible in one command with no API key."
[[howto]]
step = "Clone the repo and run: python -m contract_rag.benchmark"
+++

# Why your contract RAG returns garbage — and a reproducible fix

**TL;DR:** Dirty contract PDFs — mojibake, hyphenation, repeated headers, near-duplicates — quietly wreck RAG. On a committed synthetic set of {{ n_docs }} contracts, cleaning raises the data-quality score from **{{ quality_dirty }}** to **{{ quality_clean }}** ({{ quality_lift }}) and extraction field-F1 from **{{ f1_dirty }}** to **{{ f1_clean }}** ({{ f1_lift }}) — and you can reproduce every number with one command, no API key.

## What "garbage in" looks like

Real enterprise documents arrive with utf-8/latin-1 mojibake, words split across line breaks, page headers repeated on every page, and near-duplicate blocks. Retrieval returns noise; extraction cites the wrong block.

![Data-quality score, dirty vs cleaned](charts/quality.png)

## The fix: clean to a typed Document IR, then extract with attribution

Every parser converges to one Document IR. An ordered, pure clean pipeline fixes unicode, de-hyphenates, strips repeated headers, de-dupes, and normalizes whitespace. An explainable quality score flags what still needs review. Extraction then returns sourced, confidence-scored facts — every value a verbatim span of its cited block.

![Extraction field-F1, dirty vs cleaned](charts/field_f1.png)

## Reproduce it yourself

```bash
git clone <repo> && cd contract-rag
uv sync --extra dev --extra benchmark
python -m contract_rag.benchmark
```

You will get the same field-F1 ({{ f1_dirty }} → {{ f1_clean }}) and quality ({{ quality_dirty }} → {{ quality_clean }}) numbers shown above.

## Honest caveat

The dirt here is **simulated** by a seeded corruption suite on synthetic contracts, so this proves the pipeline's *recovery behavior end-to-end* — not real-world OCR accuracy. On real labeled contracts (CUAD, download-gated) the same cleaning lifts field-F1 from 0.33 to 0.70.

## Get a free diagnosis

If your RAG returns garbage on real contracts, we will run this before/after on one of your de-identified documents and show you the numbers. Open an issue or get in touch.
