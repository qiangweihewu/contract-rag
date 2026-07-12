"""One-command PoC health-check pack — the automated form of the G1 30-day PoC
run-book (`docs/fde/g1-poc-report-pack.md`): point at a folder of a customer's
worst contracts and get the full report pack back.

See `contract_rag.healthcheck.core` for the pure orchestration logic and
`python -m contract_rag.healthcheck` for the CLI.
"""
from __future__ import annotations
