"""Reproducible, credential-free before/after cleaning benchmark.

Runs the exact production machinery (dirtify -> clean_ir -> quality/extraction)
on the committed synthetic NDA corpus so anyone can reproduce the numbers with
no CUAD download, no API key, and no network. The injected dirt is SIMULATED
(the dirtify suite); this proves the pipeline's recovery behavior end-to-end,
not real-world OCR accuracy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import BaseModel, computed_field

from contract_rag.baseline import run_baseline
from contract_rag.clean.pipeline import clean_ir
from contract_rag.clean.quality import compute_quality_score
from contract_rag.config import Settings
from contract_rag.eval.dirtify import dirtify
from contract_rag.eval.golden import load_golden_set
from contract_rag.ir import DocumentIR


class DocQuality(BaseModel):
    doc_id: str
    quality_dirty: float
    quality_clean: float


class BenchmarkResult(BaseModel):
    corpus: str
    seed: int
    n_docs: int
    # extraction field-F1 (corpus-level) on the dirty vs cleaned IR
    f1_dirty: float
    f1_clean: float
    source_accuracy: float
    # quality score (mean over docs) on the dirty vs cleaned IR
    quality_dirty_mean: float
    quality_clean_mean: float
    per_doc: list[DocQuality]

    @computed_field
    @property
    def f1_lift(self) -> float:
        return self.f1_clean - self.f1_dirty

    @computed_field
    @property
    def quality_lift(self) -> float:
        return self.quality_clean_mean - self.quality_dirty_mean


def run_benchmark(
    settings: Settings, extractor, parse_fn: Callable[[Path], DocumentIR],
    *, vertical, seed: int = 0, corpus: str = "synthetic-nda",
) -> BenchmarkResult:
    """Measure quality + extraction lift from cleaning simulated dirt.

    `parse_fn` builds the clean IR for a source path (e.g. text_to_ir). The dirty
    run corrupts it; the cleaned run corrupts then cleans it — mirroring
    clean.lift.measure_cleaning_lift, but on a committed corpus + typed result.
    """
    dirty = run_baseline(settings, extractor,
                         lambda p: dirtify(parse_fn(p), seed=seed), vertical=vertical)
    cleaned = run_baseline(settings, extractor,
                           lambda p: clean_ir(dirtify(parse_fn(p), seed=seed)), vertical=vertical)

    per_doc: list[DocQuality] = []
    for g in load_golden_set(settings.golden_set_dir):
        base = parse_fn(settings.data_dir / g.source_pdf)
        d = dirtify(base, seed=seed)
        per_doc.append(DocQuality(
            doc_id=g.doc_id,
            quality_dirty=compute_quality_score(d).quality_score,
            quality_clean=compute_quality_score(clean_ir(d)).quality_score,
        ))

    n = len(per_doc)
    return BenchmarkResult(
        corpus=corpus, seed=seed, n_docs=n,
        f1_dirty=dirty["field_f1"], f1_clean=cleaned["field_f1"],
        source_accuracy=cleaned["source_accuracy"],
        quality_dirty_mean=sum(p.quality_dirty for p in per_doc) / n if n else 0.0,
        quality_clean_mean=sum(p.quality_clean for p in per_doc) / n if n else 0.0,
        per_doc=per_doc,
    )


def run_nda_benchmark(seed: int = 0) -> BenchmarkResult:
    """The committed-corpus benchmark: dirtify -> clean the 8 synthetic NDAs."""
    from contract_rag.verticals.nda.eval import NDA_GOLDEN_DIR, text_to_ir
    from contract_rag.verticals.nda.vertical import NDAVertical

    vertical = NDAVertical()
    settings = Settings(vertical="nda", golden_set_dir=NDA_GOLDEN_DIR, data_dir=NDA_GOLDEN_DIR)
    return run_benchmark(settings, vertical.rule_extractor, text_to_ir,
                         vertical=vertical, seed=seed, corpus="synthetic-nda")
