import os
from pathlib import Path

import pytest

from contract_rag.eval.cuad import build_golden_from_cuad
from contract_rag.eval.golden import load_golden_set

CUAD_DIR = os.environ.get("CUAD_DIR")


@pytest.mark.skipif(not CUAD_DIR, reason="set CUAD_DIR to the extracted CUAD release")
def test_build_golden_from_cuad_writes_docs(tmp_path: Path):
    out = tmp_path / "golden_set"
    data = tmp_path / "data"
    count = build_golden_from_cuad(Path(CUAD_DIR), out, data, n=5)
    assert count > 0
    docs = load_golden_set(out)
    assert len(docs) == count
    assert all("counterparty" in d.facts for d in docs)
