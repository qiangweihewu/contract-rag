"""NDA-vertical clause taxonomy: classify_clause + permission_tags + rule tables.
Same shape as the contract vertical's enrich; NDA-specific clause types.
Credential-free, deterministic."""
from __future__ import annotations

import re

from contract_rag.chunk.models import Chunk

# First match wins; order from most-specific to least.
_CLAUSE_RULES: list[tuple[str, str]] = [
    ("governing_law", r"governing law|governed by|jurisdiction|\bvenue\b"),
    ("return_of_materials", r"return or destroy|return and destroy|return all|promptly destroy|destroy all"),
    ("definition", r"definition of|\bmeans any\b|shall mean|defined (?:as|term)"),
    ("term", r"remain in (?:full )?(?:force|effect)|term of this|for a (?:term|period) of"),
    ("permitted_use", r"permitted (?:use|purpose)|solely (?:for|to)|sole purpose|solely in connection"),
    ("remedies", r"injunctive relief|irreparable harm|equitable relief|\bremed(?:y|ies)\b"),
    ("exclusions", r"publicly available|already known|independently developed|in the public domain"),
    ("confidentiality", r"confidential|non-?disclosure|proprietary"),
]

_TAG_BY_TYPE = {
    "confidentiality": "restricted",
    "definition": "restricted",
    "return_of_materials": "legal",
    "term": "legal",
    "governing_law": "legal",
    "permitted_use": "legal",
    "remedies": "legal",
    "exclusions": "legal",
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
    if re.search(r"confidential|proprietary", chunk.text, re.IGNORECASE):
        tags.add("restricted")
    return sorted(tags)
