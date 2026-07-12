"""python -m contract_rag.healthcheck <input_dir> [out_dir]

The one-command PoC health-check pack: point at a folder of a customer's worst
contracts (arbitrary PDF/DOCX) and get the full deliverable pack back — per-doc
data-quality reports, a combined CLM facts export, and a corpus summary.

`EXTRACT_BACKEND` is honored (default `rule`, credential-free) via the shared
`get_settings()`/`get_extractor()` seam, exactly like `demo.report`/`demo.batch`.
"""
from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path

from contract_rag.config import get_settings
from contract_rag.extract.rules import RuleExtractor
from contract_rag.healthcheck.core import (
    DEFAULT_TIMEOUT_S,
    default_parse_fn,
    run_healthcheck,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m contract_rag.healthcheck",
        description="30-day-PoC health-check pack for a folder of customer PDF/DOCX contracts.",
    )
    ap.add_argument("input_dir", help="folder of customer PDF/DOCX contracts")
    ap.add_argument("out_dir", nargs="?", default="healthcheck_out",
                    help="output folder for the report pack (default: healthcheck_out)")
    ap.add_argument("--clm", choices=("salesforce", "ironclad", "generic"),
                    default=os.environ.get("EXPORT_CLM", "generic"),
                    help="CLM field-name mapping for the combined facts export")
    ap.add_argument("--timeout", type=float,
                    default=float(os.environ.get("HEALTHCHECK_TIMEOUT", DEFAULT_TIMEOUT_S)),
                    help="per-document timeout in seconds (default: generous)")
    args = ap.parse_args()

    settings = get_settings()
    extractor = RuleExtractor()
    if settings.extract_backend not in ("fake", "rule"):
        from contract_rag.extract.extractor import get_extractor

        extractor = get_extractor(settings)

    summary = run_healthcheck(
        Path(args.input_dir), Path(args.out_dir), extractor,
        parse_fn=functools.partial(default_parse_fn, settings=settings),
        clm=args.clm, timeout=args.timeout,
    )
    print(f"processed {summary.n_ok}/{summary.n_docs} documents "
          f"({summary.n_failed} failed) -> {args.out_dir}/")
    print(f"mean quality {summary.quality_mean:.2f}, field STP {summary.stp['stp_rate']:.0%}"
          if summary.quality_mean is not None else "no documents processed")
    print(f"wrote {args.out_dir}/summary.html, summary.json, facts.csv, facts.json")


if __name__ == "__main__":
    main()
