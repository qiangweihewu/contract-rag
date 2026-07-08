"""Contract-vertical clause taxonomy: classify_clause, permission_tags, and their rule tables.

These definitions were moved verbatim from enrich/enricher.py so the generic engine
(enrich/enricher.py) can later accept any vertical's classify/tag functions.

Credential-free, deterministic — the retrieval-layer analogue of the `rule` extractor.
A clause LLM classifier could replace `classify_clause` behind the same signature later.
"""
from __future__ import annotations

import re

from contract_rag.chunk.models import Chunk

# First match wins; order from most-specific to least.
_CLAUSE_RULES: list[tuple[str, str]] = [
    ("governing_law", r"governing law|governed by|jurisdiction|\bvenue\b"),
    ("confidentiality", r"confidential|non-?disclosure|proprietary information"),
    ("indemnification", r"indemnif"),
    ("intellectual_property", r"intellectual property|patent|copyright|trademark|\blicense\b"),
    ("termination", r"terminat"),
    ("renewal", r"renew|successive (?:term|period)"),
    ("payment", r"payment|\bfees?\b|invoice|compensation|\bprice\b|royalt"),
    ("liability", r"liabilit|limitation of liability|\bdamages\b"),
    ("warranty", r"warrant"),
    ("assignment", r"assign"),
]

_TAG_BY_TYPE = {
    "payment": "finance",
    "intellectual_property": "legal:ip",
    "confidentiality": "restricted",
    "indemnification": "legal",
    "liability": "legal",
    "governing_law": "legal",
    "termination": "legal",
    "renewal": "legal",
    "warranty": "legal",
    "assignment": "legal",
}


def classify_clause(chunk: Chunk) -> str:
    hay = f"{chunk.heading or ''} {chunk.text}".lower()
    for label, pattern in _CLAUSE_RULES:
        if re.search(pattern, hay):
            return label
    return "other"


def permission_tags(chunk: Chunk) -> list[str]:
    clause_type = chunk.clause_type or classify_clause(chunk)
    tags = {_TAG_BY_TYPE.get(clause_type, "general")}
    if re.search(r"\$\s?\d", chunk.text):
        tags.add("finance")
    if re.search(r"confidential|proprietary", chunk.text, re.IGNORECASE):
        tags.add("restricted")
    return sorted(tags)
