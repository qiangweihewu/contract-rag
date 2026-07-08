"""Pure IR→IR PII redaction. Mirrors the clean/ step discipline: never mutate,
rebuild via model_copy. Redaction-at-ingest is the spec's Day-1 compliance must."""
from __future__ import annotations

from pydantic import BaseModel

from contract_rag.ir import DocumentIR
from contract_rag.security.pii import PIIMatch, PIIType, detect_pii


class RedactionResult(BaseModel):
    ir: DocumentIR
    matches: list[PIIMatch]


def _placeholder(t: PIIType) -> str:
    return f"[REDACTED:{t.name}]"


def redact_text(text: str, types: list[PIIType] | None = None) -> tuple[str, list[PIIMatch]]:
    matches = detect_pii(text, types=types)
    if not matches:
        return text, matches
    # detect_pii collects matches per-pattern independently, so spans can overlap
    # (e.g. an IP abutting a phone). Merge overlapping spans into disjoint regions
    # so every PII character is masked — dropping a span could leave PII exposed.
    # A merged region is labeled with the first span's type; label precision is
    # secondary to leak-freeness for a security primitive.
    regions: list[list] = []  # [start, end, type_of_first_span_in_region]
    for m in sorted(matches, key=lambda m: (m.start, m.end)):
        if regions and m.start < regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], m.end)  # extend the open region
        else:
            regions.append([m.start, m.end, m.type])
    out = text
    for start, end, ptype in sorted(regions, key=lambda r: r[0], reverse=True):
        out = out[:start] + _placeholder(ptype) + out[end:]
    return out, matches


def redact_ir(ir: DocumentIR, types: list[PIIType] | None = None) -> RedactionResult:
    new_blocks = []
    all_matches: list[PIIMatch] = []
    for block in ir.blocks:
        redacted, matches = redact_text(block.text, types=types)
        for m in matches:
            all_matches.append(m.model_copy(update={"block_id": block.block_id}))
        new_blocks.append(block.model_copy(update={"text": redacted}) if matches else block)
    return RedactionResult(ir=ir.model_copy(update={"blocks": new_blocks}), matches=all_matches)
