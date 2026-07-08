"""Rule-based defined-term extraction. Detects `"Term" means ...`-style clauses so a
later layer (chunk enrichment) can inject a term's definition into any chunk that USES
it. Deterministic, credential-free, like extract/rules.py — no config, no I/O, no LLM;
an LLM extractor can join later behind the same `Definition` return type.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from contract_rag.chunk.models import Chunk
from contract_rag.ir import BlockType, DocumentIR

_HEADINGS = (BlockType.TITLE, BlockType.HEADING)
_FURNITURE = (BlockType.HEADER, BlockType.FOOTER)

_DEFINITIONS_HEADING_RE = re.compile(r"definitions?", re.IGNORECASE)

# Pattern 1: "Term" means / shall mean / has the meaning / refers to / is defined as ...
# Straight (") and curly (“ ”) double quotes both accepted.
_QUOTED_DEFINES_RE = re.compile(
    r'["“](?P<term>[A-Z][^"”]{0,80}?)["”]\s+'
    r"(?P<verb>shall mean|means|has the meaning|refers to|is defined as)\b"
)

# Pattern 2: ... (the "Term") / (each a "Term") / (hereinafter "Term") /
# (collectively, the "Terms")
_PAREN_REF_RE = re.compile(
    r'\((?:[a-z][a-z ,]{0,30})?["“](?P<term>[A-Z][^"”]{0,60}?)["”]\)'
)

# Pattern 3: definition-list entry — "Term": ... / Term: ... (1-6 Capitalized Words),
# gated to blocks under a Definitions heading (checked by the caller).
_DEF_LIST_RE = re.compile(
    r'^["“]?(?P<term>[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,5})["”]?\s*:\s*(?P<rest>.+)',
    re.DOTALL,
)

# Sentence boundary: a period followed by whitespace and a capital letter (optionally
# behind an opening quote, so back-to-back quoted definitions in one block don't bleed
# into each other's clause text). Used both to find where a defining clause ends and,
# for pattern 2, where its containing sentence starts/ends.
_SENT_BOUND_RE = re.compile(r'\.\s+(?=["“]?[A-Z])')

_MAX_LEN = 300


class Definition(BaseModel):
    term: str  # as defined, original casing, no quotes
    definition: str  # the defining sentence/clause text (trimmed, capped at 300 chars)
    block_id: str  # block where the definition lives


def _valid_term(term: str) -> bool:
    if len(term) < 3 or not term[0].isupper():
        return False
    words = term.split()
    return 1 <= len(words) <= 6


def _sentence_end(text: str, pos: int) -> int:
    """End index (exclusive, period included) of the sentence containing `pos`,
    searching forward from `pos`."""
    m = _SENT_BOUND_RE.search(text, pos)
    return m.start() + 1 if m else len(text)


def _sentence_span(text: str, pos: int) -> tuple[int, int]:
    """(start, end) of the sentence containing `pos` (period included at end)."""
    start = 0
    for m in _SENT_BOUND_RE.finditer(text):
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            return start, m.start() + 1
    return start, len(text)


def extract_definitions(ir: DocumentIR) -> list[Definition]:
    defs: list[Definition] = []
    seen: set[str] = set()
    heading: str | None = None

    def add(raw_term: str, raw_definition: str, block_id: str) -> None:
        term = raw_term.strip()
        if not _valid_term(term):
            return
        definition = raw_definition.strip()[:_MAX_LEN]
        if not definition:
            return
        key = term.lower()
        if key in seen:  # first occurrence wins
            return
        seen.add(key)
        defs.append(Definition(term=term, definition=definition, block_id=block_id))

    for b in ir.blocks:
        if b.type in _FURNITURE:
            continue
        text = b.text
        if not text.strip():
            continue

        # Collect all pattern matches in this block as (position, term, definition)
        # candidates, then apply them in TEXTUAL order — so when two patterns hit the
        # same term in one block, the textually-first occurrence wins the dedupe, not
        # whichever pattern happens to be processed first. Cross-block order is the
        # block iteration order, already first-wins.
        candidates: list[tuple[int, str, str]] = []

        for m in _QUOTED_DEFINES_RE.finditer(text):
            verb_start = m.start("verb")
            end = _sentence_end(text, verb_start)
            candidates.append((m.start(), m.group("term"), text[verb_start:end]))

        for m in _PAREN_REF_RE.finditer(text):
            start, end = _sentence_span(text, m.start())
            candidates.append((m.start(), m.group("term"), text[start:end]))

        is_heading = b.type in _HEADINGS
        if not is_heading and heading and _DEFINITIONS_HEADING_RE.search(heading):
            m = _DEF_LIST_RE.match(text)
            if m:  # pattern-3 entries anchor at block start (position 0)
                candidates.append((m.start(), m.group("term"), m.group("rest")))

        for _pos, term, definition in sorted(candidates, key=lambda c: c[0]):
            add(term, definition, b.block_id)

        if is_heading:
            heading = text.strip() or heading

    return defs


def _usage_pattern(term: str) -> re.Pattern[str]:
    """Case-sensitive whole-word usage pattern for `term`, tolerating plural/possessive
    suffixes, plus the `ies` plural of the last word when it ends in `y`
    (`Party` -> `Parties`)."""
    alternatives = [re.escape(term) + r"(?:s|'s|es)?"]
    if term and term.split()[-1].endswith("y"):
        alternatives.append(re.escape(term[:-1] + "ies"))
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b")


def term_used(text: str, term: str) -> bool:
    """Whether `term` is USED in `text`, via the same case-sensitive whole-word matcher
    `inject_definitions` uses to decide what to inject. Public wrapper around
    `_usage_pattern` so callers outside this module (e.g. the Context-Recall
    defs-dependent/independent split) don't duplicate the regex."""
    return bool(_usage_pattern(term).search(text))


def inject_definitions(chunks: list[Chunk], definitions: list[Definition], *,
                        max_defs_per_chunk: int = 3,
                        max_chars_per_chunk: int = 600) -> list[Chunk]:
    """Inject the definitions of terms USED in each chunk's text into that chunk's
    `index_extra` (retrieval-only text; `text`/`block_ids` are never touched). Chunks
    with no matching definition are returned as the SAME object — a provable no-op."""
    if not definitions:
        return chunks

    out: list[Chunk] = []
    for chunk in chunks:
        candidates: list[tuple[int, int, Definition]] = []  # (-freq, first_pos, def)
        for d in definitions:
            if d.block_id in chunk.block_ids:
                continue  # the definition lives in this chunk — nothing to inject
            matches = list(_usage_pattern(d.term).finditer(chunk.text))
            if not matches:
                continue
            candidates.append((-len(matches), matches[0].start(), d))

        if not candidates:
            out.append(chunk)
            continue

        candidates.sort(key=lambda c: (c[0], c[1]))

        pieces: list[str] = []
        def_block_ids: list[str] = []
        for _neg_freq, _pos, d in candidates:
            if len(pieces) >= max_defs_per_chunk:
                break
            piece = f'"{d.term}" means {d.definition}'
            trial = pieces + [piece]
            trial_extra = "[DEFINITIONS: " + " | ".join(trial) + "]"
            if len(trial_extra) > max_chars_per_chunk:
                break
            pieces.append(piece)
            def_block_ids.append(d.block_id)

        if not pieces:
            out.append(chunk)
            continue

        index_extra = "[DEFINITIONS: " + " | ".join(pieces) + "]"
        out.append(chunk.model_copy(update={
            "index_extra": index_extra,
            "definition_block_ids": chunk.definition_block_ids + def_block_ids,
        }))

    return out
