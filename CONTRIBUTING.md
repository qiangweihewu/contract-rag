# Contributing

## Setup

```bash
uv sync --extra dev        # runtime + dev deps (pytest, reportlab)
uv run pytest               # full suite
```

Python is pinned to 3.12 (`.python-version`) because the optional scan path (`paddleocr`/
`paddlepaddle`) has no `cp314` wheels at time of writing. The unit suite never needs paddle
installed — it's imported lazily inside the parser module.

Run `./setup.sh` for a one-shot install + sanity check + a printout of the credential-free demo
commands.

## Test invariants

- **Unit tests are dependency-free.** No network calls, no GPU, no API keys, no external
  services. They use hand-built `DocumentIR` fixtures and fake extractors/embedders/clients
  injected through function parameters. If a test you're adding needs a real credential or a
  live endpoint, it belongs in `tests/integration/`, not `tests/`.
- **Integration tests are gated, not deleted.** Each lives under `tests/integration/` and skips
  itself (`pytest.mark.skipif`) when its backing resource is absent: an API key + governance
  flag, a local endpoint URL, a dataset directory env var, or a fixture file. Never make an
  integration test fail-hard when its dependency is missing — skip it, with a `reason=` that
  says what to set.
- **`uv run pytest` must stay green with zero external setup.** That's the whole point of the
  unit/integration split — a contributor with a fresh clone and no credentials should get a full
  passing run.

## Conventions

See `CLAUDE.md` for the details (pure IR transforms, `dirtify` mirroring `clean`, gold
canonicalization, dependency-injection seams, the external-LLM governance gate, additive/
byte-identical defaults for new optional signals). The short version: every IR-to-IR
transformation is pure and immutable (`model_copy`, never in-place mutation), every extraction
backend must emit the identical facts schema, and `config.assert_backend_allowed()` is the one
rule that should never be relaxed — it's what keeps `EXTRACT_BACKEND=openai` (or a non-local
local/mlx/constrained endpoint) from running without an explicit `ALLOW_EXTERNAL_LLM=true`.

## Commits and PRs

- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `chore:`) — it keeps the history skimmable and enables automated
  changelogs later.
- Keep PRs scoped to one change. If you're adding a cleaner, add its matching dirtifier in the
  same PR so the cleaning-lift metric stays measurable.
- Include the eval/test output that demonstrates the change works — a new metric or a changed
  number should show its before/after, the same way the README's results table cites a
  reproduction command for every figure.

## Reporting bugs / requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`.
