from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from contract_rag.text import normalize  # re-exported for back-compat

__all__ = ["GoldenDoc", "load_golden_set", "normalize"]


class GoldenDoc(BaseModel):
    doc_id: str
    source_pdf: str
    facts: dict[str, str]


def load_golden_set(dir: Path) -> list[GoldenDoc]:
    dir = Path(dir)
    return [GoldenDoc.model_validate_json(p.read_text()) for p in sorted(dir.glob("*.json"))]
