"""Evaluate the NDA vertical over the committed synthetic golden set, reusing the
generic run_baseline + metrics harness (which is itself proof the eval is
vertical-agnostic). Credential-free: synthetic prose, no docling/network."""
from __future__ import annotations

import re
from pathlib import Path

from contract_rag.baseline import format_report, run_baseline
from contract_rag.config import Settings
from contract_rag.ir import BlockType, DocBlock, DocumentIR
from contract_rag.verticals.nda.vertical import NDAVertical

# examples/nda/ at the repo root (this file is src/contract_rag/verticals/nda/eval.py)
NDA_GOLDEN_DIR = Path(__file__).resolve().parents[4] / "examples" / "nda"


def text_to_ir(path: Path) -> DocumentIR:
    """Build a DocumentIR from synthetic NDA prose (blank-line-separated paragraphs)."""
    path = Path(path)
    text = path.read_text()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks = [
        DocBlock(block_id=f"#/b/{i}", type=BlockType.PARAGRAPH, text=p,
                 confidence=1.0, source_engine="synthetic")
        for i, p in enumerate(paras)
    ]
    return DocumentIR(doc_id=path.stem, source_uri=path.as_uri(), file_hash="synthetic",
                      mime_type="text/plain", blocks=blocks, metadata={})


def evaluate_nda(golden_dir: Path | None = None) -> dict:
    golden_dir = Path(golden_dir) if golden_dir is not None else NDA_GOLDEN_DIR
    vertical = NDAVertical()
    settings = Settings(vertical="nda", golden_set_dir=golden_dir, data_dir=golden_dir)
    return run_baseline(settings, vertical.rule_extractor, text_to_ir, vertical=vertical)


def main() -> None:
    print(format_report(evaluate_nda()))


if __name__ == "__main__":
    main()
