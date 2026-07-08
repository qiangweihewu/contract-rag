"""High-precision regex PII detection. Credential-free; covers the field types
common in commercial contracts (email, phone, SSN, payment card, IP)."""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel


class PIIType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP = "ip"


class PIIMatch(BaseModel):
    type: PIIType
    value: str
    start: int
    end: int
    block_id: str | None = None


# Order matters only for tie-breaks; matches are returned sorted by start offset.
_PATTERNS: dict[PIIType, re.Pattern] = {
    PIIType.EMAIL: re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    PIIType.CREDIT_CARD: re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b"),
    # Bounded against adjacent digits/dots so a dotted quad inside a longer run
    # (a version string like 1.2.3.4.5) is not matched; octet range + section
    # context are validated in _ip_is_real below.
    PIIType.IP: re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    PIIType.PHONE: re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
}

# Words that, immediately before a dotted quad, mark it as a document reference
# (section/clause numbering), not an IP address. Contract-domain high precision:
# IPs are rare, hierarchical section numbers are common.
_IP_SECTION_MARKERS = frozenset({
    "section", "sections", "sec", "§", "article", "articles", "art",
    "clause", "clauses", "exhibit", "schedule", "appendix", "annex",
    "attachment", "paragraph", "para", "no", "item", "part", "figure",
    "fig", "table", "version", "v", "rev",
})

_IP_PREFIX_WORD = re.compile(r"([A-Za-z§]+)\.?\s*$")


def _ip_is_real(text: str, match: re.Match) -> bool:
    """A dotted quad is an IP iff every octet is 0-255 and it is not preceded by
    a section-marker word (the 'Section 1.2.3.4' over-redaction case)."""
    if any(int(o) > 255 for o in match.group(0).split(".")):
        return False
    prefix = text[: match.start()].rstrip()
    m = _IP_PREFIX_WORD.search(prefix)
    if m and m.group(1).lower() in _IP_SECTION_MARKERS:
        return False
    return True


def detect_pii(text: str, types: list[PIIType] | None = None) -> list[PIIMatch]:
    selected = types if types is not None else list(_PATTERNS)
    matches: list[PIIMatch] = []
    for t in selected:
        for m in _PATTERNS[t].finditer(text):
            if t is PIIType.IP and not _ip_is_real(text, m):
                continue
            matches.append(PIIMatch(type=t, value=m.group(0), start=m.start(), end=m.end()))
    matches.sort(key=lambda m: m.start)
    return matches
