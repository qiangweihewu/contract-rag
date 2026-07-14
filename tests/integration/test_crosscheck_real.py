"""Gated on the real FinCriticalED IR caches: the paddle (primary) cache under
CROSSCHECK_PRIMARY_CACHE (default ~/.cache/contract-rag/fincriticaled-run/ir)
and the dots.ocr (verifier) cache under CROSSCHECK_VERIFIER_CACHE (default
~/.cache/contract-rag/fincriticaled-run-dots/ir). Runs evaluate_crosscheck
end-to-end over the real load_samples set and asserts directional evidence —
some omissions exist and at least one is caught — not the pre-registered bar
itself (that's a separate, honestly-reported measurement)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_home = Path.home() / ".cache" / "contract-rag"
_primary = Path(os.environ.get("CROSSCHECK_PRIMARY_CACHE", str(_home / "fincriticaled-run" / "ir")))
_verifier = Path(os.environ.get("CROSSCHECK_VERIFIER_CACHE", str(_home / "fincriticaled-run-dots" / "ir")))
pytestmark = pytest.mark.skipif(
    not (_primary.exists() and _verifier.exists()),
    reason="set CROSSCHECK_PRIMARY_CACHE + CROSSCHECK_VERIFIER_CACHE to the "
           "FinCriticalED paddle/dots IR caches (or populate the defaults under "
           "~/.cache/contract-rag/)",
)


def test_evaluate_crosscheck_over_real_caches():
    from contract_rag.eval.crosscheck import evaluate_crosscheck
    from contract_rag.eval.fincritical import ensure_dataset, load_samples

    fin_dir = os.environ.get("FINCRITICAL_DIR")
    data_dir = ensure_dataset(Path(fin_dir) if fin_dir else _home / "fincriticaled")
    cap = int(os.environ.get("CROSSCHECK_SET_SIZE", "100"))
    samples = load_samples(data_dir, cap=cap)

    rows, s = evaluate_crosscheck(samples, _primary, _verifier)

    assert rows
    assert s.n_omitted_facts > 0
    assert s.caught_facts >= 1
