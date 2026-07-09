#!/usr/bin/env bash
# One-shot setup: install deps, run the unit suite, print the credential-free demo commands.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "==> uv sync --extra dev --extra api"
uv sync --extra dev --extra api

echo "==> uv run pytest -q"
uv run pytest -q

cat <<'EOF'

Setup complete. Everything above ran with no API key, no GPU, and no external service.

Try the credential-free demo (deterministic rule-based extraction, no credentials required):

  EXTRACT_BACKEND=rule uv run python -m contract_rag.benchmark
      # dirty -> clean field-F1 and quality-score lift on the committed examples/nda/ corpus

  uv run streamlit run src/contract_rag/demo/app.py
      # interactive dashboard (needs: uv sync --extra dev --extra app)

  uv run python -m contract_rag.demo.report path/to/contract.pdf report.html
      # self-contained before/after data-quality HTML report for one PDF

See README.md for the full architecture overview, results table, and eval reproduction commands.
EOF
