"""Legal-domain primitives shared across verticals — jurisdiction lookup, corporate-
entity extraction, and date matching. Lives outside any single vertical package so a
new vertical reuses these directly instead of reaching into another vertical's
internals (the coupling `verticals.nda` used to have on `verticals.contract.rules`,
which would have repeated again for a third vertical).

`verticals/contract/rules.py` re-imports these names (so its own finders and any
existing consumer of `contract.rules` keep working unchanged); `verticals/nda/*`
imports them from here directly.
"""
from __future__ import annotations

import re

_MONTHS = (
    "January|February|March|April|May|June|July|"
    "August|September|October|November|December"
)
_DATE = re.compile(
    # tolerate the stray space-before-comma CUAD spans sometimes carry ("July 11 , 2006")
    rf"(?:(?:{_MONTHS})\s+\d{{1,2}}\s*,?\s*\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})",
    re.IGNORECASE,
)

# Canonical jurisdiction vocabulary. Shared by extractors AND gold normalizers so both
# sides canonicalize text -> the same answer space; the extractor still has to LOCATE
# the governing-law clause among all blocks (the part it can fail).
_JURISDICTIONS = (
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "District of Columbia",
    # common non-US jurisdictions that appear in CUAD
    "Ontario", "Quebec", "British Columbia", "Alberta",
    "England", "Wales", "Scotland", "Canada",
)
# longest-first so "West Virginia" wins over "Virginia", "New York" over a bare match
_JUR_RE = re.compile(
    r"\b(" + "|".join(re.escape(j) for j in sorted(_JURISDICTIONS, key=len, reverse=True)) + r")\b"
)
_JUR_SET = {j.lower() for j in _JURISDICTIONS}

# A corporate entity: a Capitalized/ALL-CAPS name sequence ending in a legal suffix.
# Matching stops AT the suffix, so "Mount Knowledge Holdings Inc., a Delaware corp" -> the
# name only. Bare defined-term aliases ("Company", "MA", "Marketing Affiliate") lack a
# suffix and are skipped. Separators allow a comma before the suffix ("Acme, Inc.").
_CORP = (
    r"(?i:incorporated|inc|corporation|corp|company|co|limited|ltd|"
    r"llc|l\.l\.c|llp|lp|l\.p|plc|gmbh|s\.a|n\.a)"
)
_NAME = r"[A-Z][A-Za-z0-9&'.\-]*"
_CONN = r"(?i:and|of|the|&)"
_ENTITY = re.compile(
    rf"({_NAME}(?:[\s,]+(?:{_NAME}|{_CONN})){{0,6}}?[\s,]+{_CORP}\b\.?)"
)


def jurisdiction_in(text: str) -> str | None:
    m = _JUR_RE.search(text)
    return m.group(1) if m else None


def _is_jurisdiction_descriptor(e: str) -> bool:
    """True for 'Delaware limited liability company' style descriptors (a jurisdiction
    followed by a lowercase word) — not a party name. Keeps 'New York Life ...'."""
    parts = e.split()
    for jlen in (2, 1):
        if len(parts) > jlen and " ".join(parts[:jlen]).lower() in _JUR_SET:
            return parts[jlen][:1].islower()
    return False


def party_entities(text: str) -> list[str]:
    """Corporate entity names found in `text`, de-duplicated, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _ENTITY.finditer(text):
        e = m.group(1).strip().rstrip(".").strip()
        key = e.lower()
        if e and key not in seen and not _is_jurisdiction_descriptor(e):
            seen.add(key)
            out.append(e)
    return out
