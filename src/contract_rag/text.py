from __future__ import annotations

import re
import string

_WS = re.compile(r"\s+")
_PUNCT = str.maketrans("", "", string.punctuation)
_TOKENS = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    """Lowercase alphanumeric tokens — shared by BM25 and the hashing embedder."""
    return _TOKENS.findall(s.lower())


def normalize(s: str) -> str:
    """Lenient text canonicalizer: lowercase, collapse whitespace, strip punctuation.
    Shared by golden matching, eval metrics, and the extraction verifier so they all
    agree on when two strings are 'the same'."""
    s = _WS.sub(" ", s.lower().strip())
    return s.translate(_PUNCT).strip()
