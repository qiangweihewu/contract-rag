from __future__ import annotations

import json
from pathlib import Path

from contract_rag.baseline import run_baseline
from contract_rag.config import Settings
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.base import ExtractedClause
from contract_rag.verticals.contract.schema import ContractFacts


class _Extractor:
    def extract(self, ir):
        return ContractFacts(
            counterparty=ExtractedClause(),
            effective_date=ExtractedClause(),
            governing_law=ExtractedClause(value="New York", source_block_id="b1", confidence=0.9))


def test_run_baseline_accepts_vertical(tmp_path: Path):
    golden = {"doc_id": "d", "source_pdf": "d.pdf",
              "facts": {"governing_law": "New York"}}
    (tmp_path / "d.json").write_text(json.dumps(golden))
    settings = Settings(golden_set_dir=tmp_path, data_dir=tmp_path)

    def parse_fn(path):
        return DocumentIR(doc_id="d", source_uri="file:///x", file_hash="h",
                          mime_type="application/pdf", metadata={}, blocks=[
            DocBlock(block_id="b1", type=BlockType.PARAGRAPH, text="governed by New York",
                     confidence=1.0, source_engine="docling")])

    agg = run_baseline(settings, _Extractor(), parse_fn, vertical=None)
    assert agg["n_docs"] == 1
    assert agg["per_field"]["governing_law"] == 1.0
