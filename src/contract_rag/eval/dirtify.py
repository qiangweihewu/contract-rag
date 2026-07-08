from __future__ import annotations

import random

from contract_rag.ir import BlockType, DocBlock, DocumentIR


def inject_mojibake(ir: DocumentIR, seed: int = 0, rate: float = 0.9) -> DocumentIR:
    """Inject classic utf-8/latin-1 mojibake — the #1 real encoding bug, reversible by
    ftfy. Re-encoding mangles any non-ASCII; the "Â " non-breaking-space artifact also
    garbles pure-ASCII contracts (the old re-encode-only version was a no-op on ASCII)."""
    rng = random.Random(seed)
    blocks = []
    for b in ir.blocks:
        if b.text.strip() and rng.random() < rate:
            garbled = b.text.encode("utf-8").decode("latin-1", errors="ignore")
            garbled = garbled.replace(" ", "Â ")  # space -> "Â " (mojibake'd nbsp)
            blocks.append(b.model_copy(update={"text": garbled}))
        else:
            blocks.append(b)
    return ir.model_copy(update={"blocks": blocks})


def inject_hyphenation(ir: DocumentIR, seed: int = 0, rate: float = 0.5) -> DocumentIR:
    rng = random.Random(seed)
    blocks = []
    for b in ir.blocks:
        toks = b.text.split()
        idxs = [i for i, w in enumerate(toks) if len(w) >= 6]
        if idxs and rng.random() < rate:
            i = rng.choice(idxs)
            w = toks[i]
            cut = max(1, len(w) // 2)
            toks[i] = f"{w[:cut]}-\n{w[cut:]}"
            blocks.append(b.model_copy(update={"text": " ".join(toks)}))
        else:
            blocks.append(b)
    return ir.model_copy(update={"blocks": blocks})


def inject_repeated_headers(
    ir: DocumentIR, seed: int = 0, header_text: str = "CONFIDENTIAL DRAFT", copies: int = 3
) -> DocumentIR:
    extra = [
        DocBlock(block_id=f"#/dirt/hdr/{i}", type=BlockType.HEADER, text=header_text,
                 bbox=None, confidence=1.0, source_engine="dirtify")
        for i in range(copies)
    ]
    return ir.model_copy(update={"blocks": extra + list(ir.blocks)})


def inject_near_duplicates(ir: DocumentIR, seed: int = 0, rate: float = 0.5) -> DocumentIR:
    rng = random.Random(seed)
    dupes = [
        b.model_copy(update={"block_id": f"{b.block_id}/dup"})
        for b in ir.blocks
        if b.text.strip() and rng.random() < rate
    ]
    return ir.model_copy(update={"blocks": list(ir.blocks) + dupes})


def inject_whitespace_noise(ir: DocumentIR, seed: int = 0, rate: float = 0.5) -> DocumentIR:
    rng = random.Random(seed)
    blocks = []
    for b in ir.blocks:
        if rng.random() < rate:
            noisy = "   " + b.text.replace(" ", "    ") + "  \n\n  "
            blocks.append(b.model_copy(update={"text": noisy}))
        else:
            blocks.append(b)
    blocks.append(DocBlock(block_id="#/dirt/empty/0", type=BlockType.PARAGRAPH, text="   ",
                           bbox=None, confidence=1.0, source_engine="dirtify"))
    return ir.model_copy(update={"blocks": blocks})


ALL_DIRTIFIERS = [
    inject_mojibake,
    inject_hyphenation,
    inject_repeated_headers,
    inject_near_duplicates,
    inject_whitespace_noise,
]


def dirtify(ir: DocumentIR, seed: int = 0, steps: list | None = None) -> DocumentIR:
    for i, step in enumerate(ALL_DIRTIFIERS if steps is None else steps):
        ir = step(ir, seed=seed + i)
    return ir
