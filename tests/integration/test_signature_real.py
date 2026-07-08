"""Gated on the real Tobacco800 cache: TIFFs under SIGNATURE_DIR (or REALSCAN_DIR)
+ GEDI XML under SIGNATURE_GT_DIR (or REALSCAN_GT_DIR). Runs the signature detector
end-to-end over the paddle IRs (cached, shared with the realscan harness) and asserts
it beats the trivial always-signed baseline on F1. This is the honest measurement."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_img = os.environ.get("SIGNATURE_DIR") or os.environ.get("REALSCAN_DIR", "")
_gt = os.environ.get("SIGNATURE_GT_DIR") or os.environ.get("REALSCAN_GT_DIR", "")
pytestmark = pytest.mark.skipif(
    not (_img and _gt and Path(_img).exists() and Path(_gt).exists()),
    reason="set SIGNATURE_DIR + SIGNATURE_GT_DIR (or REALSCAN_*) to the Tobacco800 cache",
)


def test_detector_scores_and_beats_baseline():
    # probe + image→PDF need these; paddle itself is skipped when the IR cache hits.
    pytest.importorskip("pypdfium2")
    pytest.importorskip("PIL")

    from contract_rag.config import get_settings
    from contract_rag.eval.signature import run_signature

    cache = Path(
        os.environ.get(
            "SIGNATURE_CACHE", str(Path.home() / ".cache" / "contract-rag" / "realscan")
        )
    )
    cap = int(os.environ.get("SIGNATURE_SET_SIZE", "100"))
    results, ev = run_signature(Path(_img), Path(_gt), cache, get_settings(), cap=cap)

    assert results and ev.n_docs == len(results)
    assert 0 < ev.n_signed < ev.n_docs, "labeled set must contain both classes"
    # every prediction carries a probability and, when signed, cited evidence
    for r in results:
        assert 0.0 <= r.prediction.confidence <= 1.0
        if r.prediction.signed:
            assert r.prediction.evidence_block_ids
    # the whole point: beat the always-signed trivial baseline on F1
    assert ev.f1 >= ev.baseline_f1
