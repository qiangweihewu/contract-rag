+++
title = "Kleister-NDA, measured: honest extraction numbers on a public benchmark"
description = "Point-in-time results on the public Kleister-NDA benchmark (real SEC EDGAR NDAs): rule-based field-F1 {{ kleister_f1_initial }} → {{ kleister_f1_improved }} with source-accuracy {{ kleister_source_acc }}, and why server-side schema-constrained decoding eliminated 30% structured-output failures."
lang = "en"
slug = "kleister-nda"
date = "2026-07-07"
target_queries = [
  "NDA extraction benchmark",
  "Kleister NDA results",
  "structured extraction from contracts accuracy",
]
[[faq]]
q = "How accurate is structured extraction from real NDA contracts?"
a = "On the public Kleister-NDA benchmark (a deterministic 40-doc set of real SEC EDGAR NDAs), a deterministic rule-based extractor reaches field-F1 {{ kleister_f1_improved }} with source-attribution accuracy {{ kleister_source_acc }} — measured 2026-07-06 and reproducible from any Kleister checkout with one command."
[[faq]]
q = "Why do synthetic extraction demos overstate accuracy?"
a = "Because rules and prompts get tuned on the demo documents themselves. The same extractor that scored {{ synthetic_nda_f1 }} on a synthetic NDA set scored {{ kleister_f1_initial }} on real Kleister-NDA documents before targeted fixes. The synthetic number proves pipeline correctness; the public benchmark measures real-document extraction."
[[faq]]
q = "Does schema-constrained decoding fix LLM structured-output failures?"
a = "At the transport level, yes: on an identical rig (Ollama, qwen2.5:32b-instruct), server-side response_format=json_schema cut schema-validation failures from {{ tools_schema_failures }} (30%) to {{ constrained_schema_failures }}, lifting field-F1 from {{ mlx_f1 }} to {{ constrained_f1 }}. It cannot fix wrong-span citations — the right value cited from the wrong block."
[[howto]]
step = "Clone the dataset: git clone https://github.com/applicaai/kleister-nda (no explicit license is stated, so dataset files are never committed to this repo)"
[[howto]]
step = "Build the deterministic 40-doc set: KLEISTER_DIR=path/to/kleister-nda uv run python -m contract_rag.verticals.nda.kleister"
[[howto]]
step = "Run the eval: KLEISTER_DIR=path/to/kleister-nda uv run python -m contract_rag.verticals.nda.kleister --eval (prints field-F1 and source-accuracy)"
+++

# Kleister-NDA, measured: honest extraction numbers on a public benchmark

**TL;DR:** On the public **Kleister-NDA** benchmark (arXiv 2105.05796 — real SEC EDGAR NDAs, a deterministic {{ kleister_n_docs }}-doc set), our credential-free rule extractor scores field-F1 **{{ kleister_f1_initial }}** out of the box and **{{ kleister_f1_improved }}** after one round of targeted rule improvements, with source-attribution accuracy **{{ kleister_source_acc }}** in both rounds — every extracted value is a verbatim span of the block it cites. Separately, switching structured LLM output from client-side TOOLS-mode function calling to server-side schema-constrained decoding cut schema-validation failures from **{{ tools_schema_failures }} (30%)** to **{{ constrained_schema_failures }}** on an identical rig. Every number in this article is a point-in-time measurement (**{{ kleister_measured_date }}**), committed to the repo in `content/kleister_results.toml` and injected at build time — **not** recomputed on each site build — and reproducible with the commands at the end.

## We attacked our own synthetic number first

Our NDA vertical first shipped with an author-written synthetic golden set, scoring field-F1 **{{ synthetic_nda_f1 }}**. That number is real but proves the wrong thing: it demonstrates *pipeline correctness* — extraction, attribution, and metrics all work end-to-end for a new domain — not real-world accuracy, because the rules were tuned on the very documents they were scored against.

So we attacked it with a third-party public benchmark. Kleister-NDA is a set of real NDAs filed with the SEC, labeled by the benchmark's authors, not by us. The same extractor that scored {{ synthetic_nda_f1 }} on synthetic documents scored **{{ kleister_f1_initial }}** on real ones. That gap — {{ synthetic_nda_f1 }} vs {{ kleister_f1_initial }} — is the honest picture, and it is exactly why extraction demos on vendor-authored data should not be trusted, including ours.

## Per-field results, two rounds

Kleister-NDA labels four fields. Field-F1 on labeled docs ("initial" = rules as tuned on the synthetic set; "improved" = after targeted fixes; the {{ kleister_n_docs }}-doc set is built deterministically from train + dev-0 — the hidden test-A gold is never used):

| field | labeled docs | initial F1 | improved F1 |
|---|---|---|---|
| party | 40 | {{ kleister_party_initial }} | **{{ kleister_party_improved }}** |
| effective_date | 29 | {{ kleister_date_initial }} | **{{ kleister_date_improved }}** |
| term | 14 | {{ kleister_term_initial }} | **{{ kleister_term_improved }}** |
| governing_law | 40 | {{ kleister_law }} | {{ kleister_law }} (unchanged) |

Blended field-F1: **{{ kleister_f1_initial }} → {{ kleister_f1_improved }}**. Source-attribution accuracy: **{{ kleister_source_acc }}** in both rounds.

## What the improvements actually were

Three targeted fixes, each aimed at a measured failure mode, each canonicalized on *both* sides (the same helper canonicalizes gold and extraction, so formatting differences never count as errors) and unit-tested:

- **party {{ kleister_party_initial }} → {{ kleister_party_improved }}.** The synthetic-set heuristic looked for explicit "Disclosing Party" / "Receiving Party" role labels — which almost never appear in real SEC NDAs. Fix: a preamble fallback that parses the "by and between …" opening clause, with a documented first-named-is-disclosing convention (explicit labels still win when present).
- **effective_date {{ kleister_date_initial }} → {{ kleister_date_improved }}.** Real filings use legalese date forms the original regex missed — "the 6th day of January, 2012", ordinals, day-month-year order, standalone letterhead dates. Fix: those forms, plus a cue-proximity finder so a date near "effective" / "entered into" beats one far away.
- **term {{ kleister_term_initial }} → {{ kleister_term_improved }}.** Word-number durations ("two (2) years") and a "shall terminate" cue.

governing_law was already at {{ kleister_law }} and was left untouched. Two regression guards held throughout: the synthetic NDA eval stayed at {{ synthetic_nda_f1 }}, and the generic CUAD contract baseline is byte-identical — the NDA rules only *reuse* shared helpers; there is no core fork.

## Source-accuracy {{ kleister_source_acc }} — by construction, not luck

Every extracted field carries a `source_block_id`, and the metric verifies the extracted value actually appears in the text of that block (for the multi-valued party field, *every* extracted entity must appear in the cited block). The rule finders only ever emit spans of the block they matched, so attribution holds by construction — accuracy stayed {{ kleister_source_acc }} even while field-F1 was {{ kleister_f1_initial }}. Getting the value wrong and citing it honestly is a recoverable failure mode; citing the wrong evidence is not.

## Structured decoding: the bottleneck was reliability, not the model

A separate measurement, on the 40-doc CUAD *contract* set (not Kleister), same rig both runs — Lambda A100-40GB, Ollama, `qwen2.5:32b-instruct` @ 32K context:

- **Client-side TOOLS-mode function calling (instructor):** **{{ tools_schema_failures }}** docs (30%) failed schema validation outright (malformed nested JSON, e.g. an extra wrapper object) and counted as misses. Field-F1 **{{ mlx_f1 }}**, source-accuracy {{ mlx_source_acc }} — partly the model dropping the `#` prefix on cited block ids.
- **Server-side schema-constrained decoding (`response_format=json_schema`):** **{{ constrained_schema_failures }}** schema failures. Field-F1 **{{ constrained_f1 }}** (two runs; decoding is non-deterministic, treat ±0.01 as run noise), source-accuracy {{ constrained_source_acc }}.

Same model, same prompts, same documents. The 30% failure class was a *transport* problem, and grammar-constrained decoding removes it at the transport level. The remaining source-accuracy gap is `wrong_span` — the right value cited from the wrong block — which no output-format constraint can fix.

## Honest limitations

- **Kleister-NDA labels only 4 fields**, and two are sparsely labeled (effective_date 29, term 14 of {{ kleister_n_docs }}). The blended F1 averages over what is scoreable.
- **The remaining party misses are structural:** most are person names or suffix-less gold entities the corporate-entity regex cannot emit by design.
- **The rule extractor is a deterministic, credential-free floor**, not a capability ceiling — no LLM was used for the Kleister numbers.
- **The structured-decoding numbers are from CUAD contracts, not Kleister** — they measure output reliability of the local-LLM path, on a different corpus.
- **These are point-in-time numbers ({{ kleister_measured_date }})**, committed as data and injected at build time. They are not live-recomputed like our cleaning benchmark article, because the dataset cannot be committed (below) and the GPU runs cannot execute in CI.

## Reproduce it yourself

The Kleister-NDA repository states no explicit license, so its files are never committed here — point `KLEISTER_DIR` at your own checkout:

```bash
git clone https://github.com/applicaai/kleister-nda
git clone <repo> && cd contract-rag && uv sync --extra dev
KLEISTER_DIR=../kleister-nda uv run python -m contract_rag.verticals.nda.kleister         # build the deterministic 40-doc set
KLEISTER_DIR=../kleister-nda uv run python -m contract_rag.verticals.nda.kleister --eval  # field-F1 + source-accuracy
```

The set build is deterministic (train + dev-0, seeded selection), so you should reproduce field-F1 **{{ kleister_f1_improved }}** / source-accuracy **{{ kleister_source_acc }}** up to environment variance in the PDF parser. If your numbers differ materially, open an issue — that is the point of measuring in public.
