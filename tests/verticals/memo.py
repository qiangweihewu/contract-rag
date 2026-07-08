from __future__ import annotations

import re
from collections.abc import Mapping
from typing import ClassVar

from pydantic import BaseModel, Field

from contract_rag.chunk.models import Chunk
from contract_rag.ir import DocumentIR
from contract_rag.verticals.base import ExtractedClause


class MemoFacts(BaseModel):
    author: ExtractedClause = Field(default_factory=ExtractedClause)
    date: ExtractedClause = Field(default_factory=ExtractedClause)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = ("author", "date")


_AUTHOR = re.compile(r"From:\s*([A-Z][A-Za-z ]+)")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


class _MemoRuleExtractor:
    def extract(self, ir: DocumentIR) -> MemoFacts:
        facts = MemoFacts()
        for b in ir.blocks:
            am = _AUTHOR.search(b.text)
            if am and not facts.author.value:
                facts.author = ExtractedClause(
                    value=am.group(1).strip(), source_block_id=b.block_id, confidence=0.9)
            dm = _DATE.search(b.text)
            if dm and not facts.date.value:
                facts.date = ExtractedClause(
                    value=dm.group(0), source_block_id=b.block_id, confidence=0.9)
        return facts


class MemoVertical:
    name = "memo"
    facts_model = MemoFacts
    field_names = MemoFacts.FIELD_NAMES
    set_fields: tuple[str, ...] = ()
    judgment_fields: tuple[str, ...] = ()
    extraction_prompt = "Extract author and date from the memo.\n\n"

    def __init__(self) -> None:
        self.rule_extractor = _MemoRuleExtractor()

    def classify_clause(self, chunk: Chunk) -> str:
        return "header" if "From:" in chunk.text else "body"

    def permission_tags(self, chunk: Chunk) -> list[str]:
        return ["internal"]

    def normalize_gold(self, raw: Mapping[str, str]) -> dict[str, str]:
        return dict(raw)

    def canonicalize_value(self, name: str, value: str) -> str:
        return value

    def entities(self, value: str) -> list[str]:
        return []

    def empty_facts(self) -> MemoFacts:
        return MemoFacts()
