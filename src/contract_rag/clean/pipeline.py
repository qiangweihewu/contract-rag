from __future__ import annotations

from typing import Callable

from contract_rag.clean.normalize import (
    dedupe_blocks,
    dehyphenate,
    fix_unicode,
    strip_headers_footers,
    strip_whitespace_noise,
)
from contract_rag.ir import DocumentIR

DEFAULT_STEPS: list[Callable[[DocumentIR], DocumentIR]] = [
    fix_unicode,
    dehyphenate,
    strip_headers_footers,
    dedupe_blocks,
    strip_whitespace_noise,
]


def clean_ir(ir: DocumentIR, steps: list | None = None) -> DocumentIR:
    for step in (DEFAULT_STEPS if steps is None else steps):
        ir = step(ir)
    return ir
