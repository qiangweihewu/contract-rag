"""Parse-once IR cache. Re-parsing PDFs with docling is the slow step (~30-80 s/doc),
so the eval entry points take a `parse_fn`/`ir_for` seam: parse each PDF once, dump
the `DocumentIR` as JSON under `cache_dir`, and reload it on every subsequent run.

`ir_cache(cache_dir, parse_fn)` returns a drop-in `parse_fn(path) -> DocumentIR`."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from contract_rag.ir import DocumentIR


def ir_cache(
    cache_dir: Path, parse_fn: Callable[[Path], DocumentIR]
) -> Callable[[Path], DocumentIR]:
    cache_dir = Path(cache_dir)

    def _parse(pdf_path: Path) -> DocumentIR:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{pdf_path.stem}.ir.json"
        if cache_file.exists():
            return DocumentIR.model_validate_json(cache_file.read_text())
        ir = parse_fn(pdf_path)
        cache_file.write_text(ir.model_dump_json())
        return ir

    return _parse
