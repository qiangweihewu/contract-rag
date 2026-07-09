# contract-rag

A cleaning + structured-extraction pipeline for dirty contract PDFs: it turns a scanned or
export-mangled contract into sourced, verified, confidence-scored facts — the layer between
"garbage PDF" and a RAG or CLM system that can trust what it retrieves.

Website: **[contractrag.com](https://contractrag.com)**

## Why

Feed a real contract PDF — a fax-quality scan, an OCR export full of mojibake and repeated
headers, a mixed digital+scanned annex — into a naive RAG pipeline and you get garbage in,
garbage out: broken paragraphs, unattributed facts, silently dropped clauses. Every "AI reads
your contracts" pitch skips the part where the input document is the actual problem.

The thesis here is **measure, don't market**. Every number in this README is reproducible from
this repo, most of them are unflattering, and a few are outright negative results (things we
tried, benchmarked, and did not ship). If a technique doesn't move the number, the README says
so instead of quietly leaving it out.

## Quickstart

Requires Python 3.12 (pinned — see below) and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev        # runtime + dev deps (pytest, reportlab)
uv run pytest               # full suite: unit tests are fast and dependency-free;
                             # integration tests auto-skip without their backing service
```

Credential-free demo — no API key, no GPU, deterministic rules only:

```bash
EXTRACT_BACKEND=rule uv run python -m contract_rag.benchmark
```

This runs the seeded dirty→clean lift on the committed `examples/nda/` corpus and prints
field-F1 and quality-score before/after, plus a `benchmark_out/results.json` you can diff run to
run.

**A note on the scan path.** `docling` (native-digital PDFs) is a normal dependency and installs
with `uv sync`. `paddleocr`/`paddlepaddle` (the scanned/OCR path) are **not** pinned in
`pyproject.toml` — install them ad hoc if you need that branch. There are no `cp314` wheels for
paddle at the time of writing, which is why this project pins Python to 3.12
(`.python-version`, `requires-python = ">=3.12"`); Python 3.13/3.14 will break the paddle extra.
The unit suite never imports paddle — it's lazily imported inside the parser module — so
`uv run pytest` is green without it.

## Architecture

The spine is one Pydantic model: **`DocumentIR`** (`src/contract_rag/ir.py`) — a flat
`list[DocBlock]`, each block typed (`title`/`heading`/`paragraph`/`table`/`list_item`/`header`/
`footer`/`figure_caption`), carrying an optional page-anchored `bbox`, a `parent_id` for
hierarchy, a `confidence`, and a `source_engine` stamp. Every parser converges to this one model,
and every downstream layer (clean, quality scoring, extraction, chunking) consumes and returns
it — that's what makes parsers and extraction backends swappable and each layer independently
testable.

**Ingest → Parse (router) → Clean + Score → { Extract facts | Chunk → Enrich → Index → retrieve }.**

- **`parse/`** — `router.route()` picks an engine from a text-coverage probe
  (`probe.probe_document()`, via `pypdfium2`): `docling` above the coverage threshold (native
  digital text), else `vlm` if configured, else `paddleocr`. Each adapter
  (`docling_parser`/`vlm_parser`/`paddle_parser`) emits a `DocumentIR` and stamps
  `source_engine`, so runs are A/B-comparable. `router.parse(..., per_page=True)` is the
  mixed-document path: `probe.probe_pages()` profiles each page, `page_route()` routes it
  independently, contiguous same-engine page ranges are split out (`split_pdf_pages`) and parsed
  separately, then merged back in original page order — a pure single-engine document collapses
  to one segment and is byte-identical to the non-per-page path.
- **`clean/`** — `pipeline.clean_ir()` runs an ordered list of pure IR→IR steps (`fix_unicode`,
  `dehyphenate`, `strip_headers_footers`, `dedupe_blocks`, `strip_whitespace_noise`).
  `quality.compute_quality_score()` returns an explainable, weighted `QualityReport` (garble /
  table-integrity / emptiness / OCR confidence) with a `needs_review` flag.
- **`extract/`** — `extractor.get_extractor()` selects an `Extractor` by `EXTRACT_BACKEND`. Every
  backend emits the identical facts schema (each field an `{value, source_block_id, confidence}`
  clause), so evaluation is backend-agnostic. See the backends table below.
- **`chunk/` → `enrich/` → `index/`** — the RAG half. `chunk_ir()` groups blocks under their
  heading into `Chunk`s that carry `block_ids` (attribution survives retrieval). `enrich/` adds a
  rule-based clause type + permission tags. `index/` is pluggable retrieval: `BM25Index`
  (lexical), `DenseIndex` over a swappable `Embedder` (`HashingEmbedder` default/CI, gated
  `OpenAIEmbedder`) with a pluggable `VectorStore` (in-memory default, `PgVectorStore` optional),
  and `HybridIndex` fusing both via Reciprocal Rank Fusion with an ABAC tag filter and an
  optional reranker stage (free lexical, gated LLM, or a local cross-encoder).
- **`eval/`** — golden-set loading and normalization, metrics (field-F1 + source-attribution
  accuracy + an error taxonomy of correct/omission/invention), a seeded `dirtify` corruption
  suite that mirrors what `clean/` removes, and one harness per dataset (see below).

Read the module you're extending before changing it — the docstrings carry the reasoning, not
just the "what".

## Results

All numbers below are reproducible from this repo against the cited public datasets, which are
**never committed** — each eval harness downloads or expects a local checkout (paths are env
vars, gitignored). Full reproduction commands are in the next section.

| Benchmark | Result | Dataset |
|---|---|---|
| Field-F1, rule backend | **0.676** (95% CI 0.594–0.746, bootstrap) | [CUAD](https://www.atticusprojectai.org/cuad), 40-doc set |
| Field-F1, constrained backend (schema-decoded local LLM) | **0.661–0.672**, 0/40 schema failures (vs 12/40 under TOOLS-mode function calling) | CUAD, same 40-doc set |
| Field-F1, NDA vertical on real docs | **0.523 → 0.697** after targeted rule fixes | [Kleister-NDA](https://github.com/applicaai/kleister-nda) |
| Signature-presence detector | **F1 0.864** (precision 0.981, recall 0.773) | Tobacco800 scans + GEDI groundtruth |
| Fact-level omission vs. quality score | **7.7%** of expert-labeled facts missing from OCR output while the doc-level quality score reads **0.998** | [FinCriticalED](https://huggingface.co/datasets/TheFinAI/FinCriticalED) |
| Mixed-document parse routing | **98.8%** of mixed docs misroute at least one page under whole-document routing (204/464 pages) — motivated building per-page routing | [EDiTh / Véracier Industries](https://huggingface.co/datasets/lightonai/veracier-industries) |

Source-attribution accuracy on CUAD is 1.0 for the rule backend (every cited value is verified
to appear in its cited block).

### Negative results we published

Not everything we measured shipped. Publishing the misses is part of the "measure, don't
market" thesis:

- **Definition-injection retrieval (DAPEI-style)** — injecting resolved defined-term text into
  retrieval context: under a hashing embedder it *hurt* BM25 (0.690 → 0.664) with hybrid flat at
  0.586; under a semantic (OpenAI) embedder, dense-only injection was cell-for-cell identical to
  baseline (0.698) and full injection dragged hybrid down to 0.681 via the degraded BM25 side.
  Not adopted, under either embedder.
- **FrankenOCR spike** — an alternate OCR engine benchmarked against paddleocr: parity on
  field-F1, but 25–50× slower, and it hallucinates text on shredded/heavily-degraded scans.
  Not adopted.
- **OCR confidence as an omission signal** — on FinCriticalED, every located fact lands in the
  top two OCR-confidence bins regardless of whether nearby facts were dropped; confidence cannot
  flag a fact that produced no block at all. A geometric ink-coverage signal (`clean/coverage.py`)
  is a partial, region-scale substitute — validated as a strong signal for occluded
  signatures/stamps (5.4× denser uncovered ink inside GEDI-annotated zones), but only a weak
  document-level signal for individual small-fact omission.

## Eval reproduction

Build a golden set from a CUAD release, then run the eval entry points against it. All of these
need a golden set + local PDFs (built once, gitignored) — none are runnable straight off a
clone.

```bash
# Build golden_set/*.json + copy matching PDFs into data/ (gitignored)
CUAD_DIR=path/to/cuad uv run python -m contract_rag.eval.cuad   # GOLDEN_SET_SIZE=N caps count

# Field-F1 + source-attribution accuracy, with bootstrap CI
EXTRACT_BACKEND=rule STATS_CI=1 uv run python -m contract_rag.baseline

# Docling baseline vs. the parse router, same statistics flag
EXTRACT_BACKEND=rule STATS_CI=1 uv run python -m contract_rag.compare

# Dirty vs. cleaned: field-F1 and quality-score lift
EXTRACT_BACKEND=rule uv run python -m contract_rag.clean.lift

# Retrieval: Context Recall for bm25 / dense / hybrid, with optional
# definition-injection (the negative result above) and embedder choice
uv run python -m contract_rag.eval.retrieval               # EMBED_BACKEND=hashing|openai
INJECT_DEFS=1 uv run python -m contract_rag.eval.retrieval  # reproduces the DAPEI negative result
```

Dataset-specific harnesses (`eval/signature.py`, `eval/fincritical.py`, `eval/edith.py`,
`eval/coverage.py`, `eval/realscan.py`, `eval/degrade.py`) each document their required env var
(a `*_DIR` pointing at a local dataset checkout) at the top of the module and in
`uv run python -m contract_rag.<module> --help`-style docstrings; none of the source datasets
are redistributed here.

## Extraction backends

All backends emit the identical facts schema, so switching backends never changes what
downstream code consumes.

| Backend | What it is | Gated? |
|---|---|---|
| `fake` | Returns empty facts. Default. | — |
| `rule` | Deterministic regex/jurisdiction extraction. No credentials, no network. | No |
| `openai` | `instructor`-based structured extraction against the OpenAI API. | **Yes** — requires `ALLOW_EXTERNAL_LLM=true` |
| `local` | `instructor` over an OpenAI-compatible local endpoint (vLLM/SGLang). | No — document never leaves the local server |
| `mlx` | Same shape, pointed at a local Ollama endpoint. | No |
| `constrained` | Server-side `response_format=json_schema` structured decoding (any OpenAI-compatible server ≥ vLLM/SGLang/Ollama 0.5) instead of client-side function calling. | No |

`config.assert_backend_allowed()` enforces the gate: it raises unless `ALLOW_EXTERNAL_LLM=true`
whenever `EXTRACT_BACKEND=openai`, or whenever `local`/`mlx`/`constrained` point at a non-local
endpoint. This is the one governance rule in the codebase that should never be weakened — it's
what stops a contract from silently leaving your network.

## License

Apache-2.0 — see [LICENSE](LICENSE).
