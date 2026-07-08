from pathlib import Path

import pytest

from contract_rag.parse.docling_parser import parse_with_docling

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_contract.pdf"


@pytest.mark.skipif(not FIXTURE.exists(), reason="sample_contract.pdf fixture not present")
def test_parse_real_pdf_yields_nonempty_ir():
    ir = parse_with_docling(FIXTURE)
    assert len(ir.blocks) > 0
    assert all(b.source_engine == "docling" for b in ir.blocks)
    assert any(b.text.strip() for b in ir.blocks)
