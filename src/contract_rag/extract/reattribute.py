"""wrong_span re-attribution post-pass.

Motivation (measured, see CLAUDE.md): the `constrained` backend's error taxonomy
shows `wrong_span` — the extracted *value* is correct but the cited
`source_block_id` does not actually contain it (source-accuracy 0.744, 16
wrong_span cases on the 40-doc CUAD run). Attribution is the product's core
guarantee, so a wrong citation is worth repairing when the true source block can
be found deterministically, rather than quarantining an otherwise-correct value.
(Measured in anger 2026-07-12 inside the constrained-child ensemble on the
A100/Ollama rig: 9 repairs fired; that run's taxonomy showed wrong_span 1 /
source-accuracy 0.951 vs the standalone constrained run's 16 / 0.702.)

`reattribute_facts` is a pure IR-in/facts-in, facts-out/counts-out function: for
every populated `ExtractedClause` whose cited block does NOT contain its value
(same containment semantics as `eval.metrics.source_attribution_ok` /
`extract.verify._attributed` — scalar fields by normalized substring, set fields
by vertical.entities() all appearing in the block), it searches every IR block for
one that DOES contain the value. If exactly one or several qualify, the block
nearest the *originally cited* block in reading order wins (ties broken toward the
earliest index). If no block contains the value anywhere, the clause is left
untouched — this pass only relocates a correct value to its real source, it never
invents one.

This is an OPT-IN step. Nothing in the existing pipeline calls it automatically;
default behavior everywhere (extractors, `verify()`, eval harnesses) is
byte-identical unless a caller explicitly reaches for it. Two ways in:
  - the pure function itself, e.g. a `verify()` caller can do
    `facts, _ = reattribute_facts(facts, ir)` before `verify(facts, ir)`;
  - `ReattributingExtractor`, an `Extractor`-protocol wrapper an eval driver can
    place around any child extractor, e.g.
    `ReattributingExtractor(get_extractor(settings))`.
The `ensemble` backend (extract/ensemble.py) also runs this pass by default on
its merged output, since it is a new backend with no byte-identical constraint.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from contract_rag.ir import DocumentIR
from contract_rag.text import normalize

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical


def _contains(vertical: Vertical, name: str, value: str, block_text: str) -> bool:
    """Same containment semantics as `eval.metrics.source_attribution_ok` /
    `extract.verify._attributed`: scalar fields by normalized substring, set
    fields (e.g. counterparty) require every extracted entity to appear in the
    block. An empty entity list is never "attributed" (mirrors the metrics/verify
    guard against vacuous `all([])` truth on an empty set)."""
    if name in vertical.set_fields:
        entities = [normalize(e) for e in vertical.entities(value)]
        return bool(entities) and all(e in block_text for e in entities)
    return normalize(value) in block_text


def reattribute_facts(
    facts: BaseModel, ir: DocumentIR, vertical: Vertical | None = None,
) -> tuple[BaseModel, dict[str, int]]:
    """Rewrite `source_block_id` for any populated, mis-cited field to the nearest
    block (in reading order to the original citation) that actually contains the
    value. Returns `(facts', repairs)` where `repairs` maps field name -> 1 for
    every field that was re-attributed (fields left unchanged, whether because they
    were already correct or because no block contains the value, are omitted).

    Pure / immutable: `facts` and `ir` are never mutated; the result is rebuilt via
    `model_copy`, matching the rest of the codebase's IR/facts transform style."""
    from contract_rag.verticals.registry import default_vertical

    v = vertical or default_vertical()
    blocks = ir.blocks
    block_text = [normalize(b.text) for b in blocks]
    block_index = {b.block_id: i for i, b in enumerate(blocks)}

    repairs: dict[str, int] = {}
    updates: dict[str, object] = {}
    for name in v.field_names:
        clause = getattr(facts, name)
        if not clause.value or name in v.judgment_fields:
            continue  # nothing to attribute: empty clause, or a derived judgment field

        cited_idx = block_index.get(clause.source_block_id or "")
        cited_text = block_text[cited_idx] if cited_idx is not None else ""
        if _contains(v, name, clause.value, cited_text):
            continue  # already correctly attributed; leave unchanged

        ref_index = cited_idx if cited_idx is not None else 0
        best_index: int | None = None
        best_distance: int | None = None
        for i, text in enumerate(block_text):
            if not _contains(v, name, clause.value, text):
                continue
            distance = abs(i - ref_index)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = i
        if best_index is None:
            continue  # the value appears in no block anywhere; never invent a citation

        updates[name] = clause.model_copy(update={"source_block_id": blocks[best_index].block_id})
        repairs[name] = 1

    if not updates:
        return facts, repairs
    return facts.model_copy(update=updates), repairs


class ReattributingExtractor:
    """`Extractor`-protocol wrapper: runs `reattribute_facts` on a child
    extractor's output. Purely additive/opt-in — construct it explicitly around
    another extractor (e.g. `ReattributingExtractor(get_extractor(settings))`) to
    get re-attribution in an eval driver or demo path; nothing wires this in by
    default. `last_repairs` mirrors the `last_repairs`/`last_tokens` accounting
    convention used by the constrained/instructor backends; `last_tokens` and
    `last_cost_usd` are passed through from the child when present so obs/cost
    dashboards keep working unchanged."""

    def __init__(self, extractor: object, vertical: Vertical | None = None) -> None:
        self._extractor = extractor
        self._vertical = vertical
        self.last_repairs: dict[str, int] = {}

    def extract(self, ir: DocumentIR) -> BaseModel:
        facts = self._extractor.extract(ir)
        facts, repairs = reattribute_facts(facts, ir, self._vertical)
        self.last_repairs = repairs
        for attr in ("last_tokens", "last_cost_usd"):
            if hasattr(self._extractor, attr):
                setattr(self, attr, getattr(self._extractor, attr))
        return facts
