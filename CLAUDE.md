# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## What this is

`contract-rag` — a cleaning + structured-extraction pipeline for dirty contract PDFs: sourced,
verified, confidence-scored facts, proven against public golden sets (CUAD, Kleister-NDA,
Tobacco800, FinCriticalED, EDiTh). See `README.md` for the pitch, architecture summary, and the
measured results table.

## Commands

Python 3.12, managed with `uv`. Heavy deps (docling, paddleocr, openai/instructor, ftfy) are
imported lazily inside functions, so the unit suite runs without them installed or reachable.

```bash
uv sync --extra dev                          # runtime + dev deps (pytest, reportlab)
uv sync --extra dev --extra app               # + streamlit, for the interactive demo app
uv run pytest                                 # full suite — unit fast, integration auto-skips
uv run pytest tests/test_clean_pipeline.py    # one file
uv run pytest tests/test_metrics.py::test_aggregate_counts_true_positives  # one test

EXTRACT_BACKEND=rule uv run python -m contract_rag.benchmark   # credential-free dirty->clean demo
```

Eval entry points (`baseline`, `compare`, `clean.lift`, `eval.retrieval`, and the per-dataset
harnesses under `eval/`) each need a golden set + local data, built once via
`CUAD_DIR=... uv run python -m contract_rag.eval.cuad` or the corresponding vertical builder.
None of the source datasets are committed or redistributed — see each harness module's docstring
for its required env var(s).

Integration tests (`tests/integration/`) skip unless their backing service or fixture is
present: `OPENAI_API_KEY` + `ALLOW_EXTERNAL_LLM` (OpenAI extractor), `LOCAL_ENDPOINT`/
`MLX_ENDPOINT`/`CONSTRAINED_ENDPOINT` (on-device backends), `VLM_ENDPOINT` (SGLang VLM),
`CUAD_DIR`/`KLEISTER_DIR`/etc. (dataset builders), `PGVECTOR_URL` (pgvector store), or a
`tests/fixtures/sample_contract.pdf`. Everything else is a pure unit test — no network, no GPU,
no credentials.

## Architecture

**The Document IR is the spine.** `src/contract_rag/ir.py` defines `DocumentIR` →
`list[DocBlock]` (typed, optionally bbox-anchored, with `parent_id`, `confidence`,
`source_engine`). Every parser converges to this one Pydantic model, and every downstream layer
(clean, quality, extract, chunk) consumes and returns it — this is what makes parsers and
extraction backends interchangeable and each layer independently testable.

Flow: **Ingest → Parse (router) → Clean+Score → { Extract → facts | Chunk → Enrich → Index →
retrieve }.**

- `parse/` — `router.parse()` probes text coverage and routes to `docling` (native digital),
  `vlm`, or `paddleocr`; `per_page=True` routes per-page for mixed digital+scanned documents.
- `clean/` — `pipeline.clean_ir()` runs pure IR→IR normalization steps; `quality.py` scores the
  result into an explainable `QualityReport`.
- `extract/` — `extractor.get_extractor()` returns a protocol-conformant `Extractor` by
  `EXTRACT_BACKEND`; every backend emits the identical facts schema.
- `chunk/` → `enrich/` → `index/` — the RAG half: heading-scoped chunking, rule-based clause
  typing + ABAC tags, pluggable BM25/dense/hybrid retrieval with an optional reranker.
- `eval/` — golden-set loading, metrics (field-F1, source-attribution accuracy, error taxonomy),
  a seeded `dirtify` corruption suite, and one harness per benchmark dataset.

Read the relevant module's docstrings before extending a layer — they carry the reasoning behind
the design, not just what the code does.

## Conventions that matter

- **IR transforms are pure and immutable.** Clean/dirtify steps never mutate — they rebuild via
  `ir.model_copy(update={"blocks": ...})`. Match this when adding steps.
- **`dirtify` mirrors `clean`.** `eval/dirtify.py` injects the exact noise classes `clean/`
  removes (mojibake, hyphenation, repeated headers, near-dupes, whitespace noise). They're a
  paired set powering the cleaning-lift metric — adding a cleaner generally means adding its
  dirtifier so the lift stays measurable. Both are seeded for reproducibility.
- **Gold is canonicalized to match the extractor's answer space**, using the *same* helper the
  extractor uses, so both sides canonicalize identically and the extractor still has to *locate*
  the right value among all blocks (the part it can actually fail).
- **Multi-valued fields (e.g. `counterparty`) score by entity-set overlap** (Jaccard), not exact
  match — see `metrics._SET_FIELDS`. All other fields are scalar exact-match.
- **Errors are reported by type and by field risk, not one blended F1.** `aggregate()` emits an
  `error_taxonomy` (each labeled prediction is exactly one of correct/omission/invention;
  zero-gold predictions are `unscored`, never false positives) and `risk_tiers`. The optional
  vertical seam `field_risk` resolves defensively — verticals without it default to `medium`.
- **External-LLM governance is enforced, not advisory.**
  `config.assert_backend_allowed()` raises unless `ALLOW_EXTERNAL_LLM=true` when
  `EXTRACT_BACKEND=openai`, or when a local/mlx/constrained endpoint resolves to a non-local
  host. Default backend is `fake`. **Never weaken this gate** — it's what prevents a document
  from silently leaving the network.
- **Config is env-driven** through `get_settings()` → `Settings`. `get_settings()` auto-loads a
  gitignored `.env` from the cwd; real shell env vars always take precedence, so secrets live in
  `.env` while behavior flags can be passed inline. See `.env.example` for the full list.
- **Dependency injection for testability.** Functions take seams as params —
  `run_baseline(..., extractor, parse_fn)`, `router.parse(..., probe_fn, adapters)`,
  `clean_ir(ir, steps)`. Unit tests pass fakes/hand-built IRs and never touch a network or model.
- **New optional signals are additive and byte-identical by default.** When a change adds a new
  field or capability (e.g. an optional `QualityReport` field, a new reranker), the existing
  default-path output must stay byte-for-byte unchanged unless the new seam is explicitly
  invoked — regression-test that explicitly, not just "it still passes".
- **Pydantic v2 everywhere**, `from __future__ import annotations` at the top of every module.
- **Adding a new vertical (domain)** should not require forking core engines — see
  `contract_rag/verticals/nda/` for the reference shape: a facts schema, field finders, an
  enrichment classifier, a prompt, and a `normalize_gold`, registered with one line in
  `verticals/registry.py`.
