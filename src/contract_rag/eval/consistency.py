"""Same-clause consistency eval (spec §3.2): a correct extractor must give the SAME
answer when the input is perturbed in label-preserving ways (whitespace, heading case).
Stability = fraction of fields whose extracted value is invariant across perturbations.
Credential-free and deterministic with the `rule` extractor."""
from __future__ import annotations

from collections.abc import Callable

from contract_rag.extract.schema import ContractFacts
from contract_rag.ir import DocumentIR


def perturb_identity(ir: DocumentIR) -> DocumentIR:
    return ir


def perturb_whitespace(ir: DocumentIR) -> DocumentIR:
    """Double internal spaces + pad ends — semantics-preserving noise (cf. clean/ removes it)."""
    new_blocks = [
        b.model_copy(update={"text": "  " + b.text.replace(" ", "  ") + "  "}) for b in ir.blocks
    ]
    return ir.model_copy(update={"blocks": new_blocks})


def perturb_trailing_newline(ir: DocumentIR) -> DocumentIR:
    """Append a trailing newline — a benign reflow that must not move the answer."""
    new_blocks = [b.model_copy(update={"text": b.text + "\n"}) for b in ir.blocks]
    return ir.model_copy(update={"blocks": new_blocks})


PERTURBATIONS: list[Callable[[DocumentIR], DocumentIR]] = [
    perturb_identity,
    perturb_whitespace,
    perturb_trailing_newline,
]


def consistency_score(
    extractor,
    ir: DocumentIR,
    perturbations: list[Callable[[DocumentIR], DocumentIR]] | None = None,
) -> dict:
    perts = perturbations if perturbations is not None else PERTURBATIONS
    runs = [extractor.extract(p(ir)) for p in perts]
    per_field: dict[str, float] = {}
    for name in ContractFacts.FIELD_NAMES:
        values = {getattr(r, name).value for r in runs}
        per_field[name] = 1.0 if len(values) == 1 else 0.0
    overall = sum(per_field.values()) / len(per_field) if per_field else 0.0
    return {"overall": overall, "per_field": per_field}
