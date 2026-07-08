"""Signature-presence detector for scanned documents — the CLM differentiator
("which of these 8,000 historical contracts were never actually signed?").

`detect_signature(ir)` predicts, from a parsed (paddle/scanned-path) `DocumentIR`,
whether the document is physically signed, returning {signed, confidence,
evidence_block_ids, signals}. The heuristic was designed from the real Tobacco800
IRs and combines three block-level signals:

  1. **closing salutation** — a block matching "Sincerely / Regards / Truly yours / …".
     By far the strongest signal (73% of signed Tobacco800 docs carry one vs 6% of
     unsigned), and near-perfect precision: a letter that signs off was signed.
  2. **explicit signature cue** — "/s/", "duly authorized", "authorized signature",
     "By:".
  3. **signature block** — a typed personal-name line in the lower page with a
     *low-confidence* OCR token just above it: the handwritten squiggle over the
     printed name/title. This is what OCR does to an ink signature (e.g. "wm.Hobbr"
     @0.62 above "Wm. D. Hobbs" @1.0). It recovers signed docs that lack a closing
     salutation (memos, forms) at zero measured false-positive cost on Tobacco800.

The per-block-confidence occlusion signal (realscan) is deliberately *not* a primary
feature: at block granularity it barely discriminates signed from unsigned scans
(unsigned faxes/telexes are just as noisy), so leaning on it hurt precision.

Ground-truth for evaluation is GEDI (`eval.gedi.has_signature_zone` — a page is signed
iff it carries a `DLSignature` zone). Pure logic (detection, scoring) is separated from
the `__main__` shell; unit tests use hand-built IRs and fake zones (no OCR/network).

Env (entry point): SIGNATURE_DIR (dir of Tobacco800 TIFFs; default reuses the realscan
image dir if unset), SIGNATURE_GT_DIR (GEDI XML dir), SIGNATURE_CACHE (default reuses
the realscan cache so the paddle IRs are shared), SIGNATURE_SET_SIZE (default 100),
SIGNATURE_OUT (optional JSON dump).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from contract_rag.ir import DocBlock, DocumentIR

# --------------------------------------------------------------- signal patterns

# Letter/memo sign-off — the single strongest signature signal.
_CLOSING = re.compile(
    r"\b(sincerely|best regards|regards|very truly yours|truly yours|yours truly"
    r"|cordially|respectfully|yours faithfully|faithfully)\b",
    re.I,
)
# Explicit signature cues.
_SIGWORD = re.compile(
    r"/s/|duly authorized|authorized (?:signature|representative)|\bby:", re.I
)
# A typed personal-name line: "R. M. Neel", "Wm. D. Hobbs", "Thomas A. Vollmuth".
_NAMELINE = re.compile(
    r"^\s*(?:(?:[A-Z]\.|[A-Z][a-z]+\.?)\s+){1,3}[A-Z][a-z]{2,}[.,]?\s*$"
)

# Signal strengths (probability-of-signed contributions), combined via probabilistic OR.
_W_CLOSING = 0.90
_W_SIGWORD = 0.80
_W_SIGBLOCK = 0.75
# Lower page begins at this fraction of the tallest block's bottom edge.
_LOWER_FRAC = 0.5
# Vertical gap (rendered px @300dpi) between a squiggle token and the name below it.
_SIG_GAP = 300.0
# Horizontal alignment tolerance between squiggle and name (rendered px).
_SIG_XTOL = 400.0
# A squiggle token reads as low-confidence OCR.
_SIG_CONF = 0.85
# Signed iff estimated P(signed) reaches this; a single primary signal clears it.
_DECISION = 0.5
# P(signed) when no positive signal fires — absence of evidence is weak, not proof.
_NO_SIGNAL_PRIOR = 0.2


class SignaturePrediction(BaseModel):
    signed: bool
    confidence: float  # estimated P(document is physically signed), in [0, 1]
    evidence_block_ids: list[str]
    signals: list[str]  # which detectors fired: closing | sigword | sigblock


def _page_bottom(blocks: list[DocBlock]) -> float:
    ys = [b.bbox.y1 for b in blocks if b.bbox is not None]
    return max(ys) if ys else 0.0


def _closing_hits(blocks: list[DocBlock]) -> list[str]:
    return [b.block_id for b in blocks if _CLOSING.search(b.text)]


def _sigword_hits(blocks: list[DocBlock]) -> list[str]:
    return [b.block_id for b in blocks if _SIGWORD.search(b.text)]


def _sigblock_hits(blocks: list[DocBlock]) -> list[str]:
    """Typed name line in the lower page with a low-confidence token just above it
    (the ink squiggle). Returns the [name, squiggle] block ids of the first match."""
    bottom = _page_bottom(blocks)
    if bottom <= 0:
        return []
    lower_cut = _LOWER_FRAC * bottom
    names = [
        b
        for b in blocks
        if b.bbox is not None
        and b.bbox.y0 > lower_cut
        and _NAMELINE.match(b.text.strip())
    ]
    for name in names:
        ny, nx = name.bbox.y0, name.bbox.x0  # type: ignore[union-attr]
        for b in blocks:
            if b is name or b.bbox is None:
                continue
            gap = ny - b.bbox.y0
            if 0 < gap < _SIG_GAP and b.confidence < _SIG_CONF and abs(b.bbox.x0 - nx) < _SIG_XTOL:
                return [name.block_id, b.block_id]
    return []


def detect_signature(ir: DocumentIR) -> SignaturePrediction:
    """Predict physical-signature presence from a parsed IR. Pure; no I/O."""
    blocks = ir.blocks
    fired: list[tuple[str, float, list[str]]] = []
    closing = _closing_hits(blocks)
    if closing:
        fired.append(("closing", _W_CLOSING, closing))
    sigword = _sigword_hits(blocks)
    if sigword:
        fired.append(("sigword", _W_SIGWORD, sigword))
    sigblock = _sigblock_hits(blocks)
    if sigblock:
        fired.append(("sigblock", _W_SIGBLOCK, sigblock))

    if not fired:
        return SignaturePrediction(
            signed=False, confidence=_NO_SIGNAL_PRIOR, evidence_block_ids=[], signals=[]
        )

    # probabilistic OR of independent-ish signals
    prob_unsigned = 1.0
    for _, w, _ids in fired:
        prob_unsigned *= 1.0 - w
    confidence = round(1.0 - prob_unsigned, 4)
    evidence: list[str] = []
    for _, _w, ids in fired:
        for bid in ids:
            if bid not in evidence:
                evidence.append(bid)
    return SignaturePrediction(
        signed=confidence >= _DECISION,
        confidence=confidence,
        evidence_block_ids=evidence,
        signals=[name for name, _w, _ids in fired],
    )


# ------------------------------------------------------------------- evaluation

class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0


class SignatureEval(BaseModel):
    n_docs: int
    n_signed: int  # gold positives
    matrix: ConfusionMatrix
    precision: float
    recall: float
    f1: float
    accuracy: float
    # trivial always-signed baseline, for honest comparison
    baseline_precision: float
    baseline_f1: float


def evaluate_predictions(pairs: list[tuple[bool, bool]]) -> SignatureEval:
    """`pairs` = list of (predicted_signed, gold_signed)."""
    if not pairs:
        raise ValueError("no predictions to evaluate")
    tp = sum(1 for pred, gt in pairs if pred and gt)
    fp = sum(1 for pred, gt in pairs if pred and not gt)
    fn = sum(1 for pred, gt in pairs if not pred and gt)
    tn = sum(1 for pred, gt in pairs if not pred and not gt)
    m = ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)
    n = len(pairs)
    n_signed = sum(1 for _pred, gt in pairs if gt)
    base_p = n_signed / n
    base_f1 = 2 * base_p * 1.0 / (base_p + 1.0)  # always-signed: recall 1.0
    return SignatureEval(
        n_docs=n,
        n_signed=n_signed,
        matrix=m,
        precision=round(m.precision, 4),
        recall=round(m.recall, 4),
        f1=round(m.f1, 4),
        accuracy=round(m.accuracy, 4),
        baseline_precision=round(base_p, 4),
        baseline_f1=round(base_f1, 4),
    )


def format_eval(ev: SignatureEval) -> str:
    m = ev.matrix
    return "\n".join(
        [
            f"signature detector — {ev.n_docs} docs ({ev.n_signed} signed / "
            f"{ev.n_docs - ev.n_signed} unsigned by GEDI)",
            f"  confusion: tp={m.tp} fp={m.fp} fn={m.fn} tn={m.tn}",
            f"  precision={ev.precision:.3f} recall={ev.recall:.3f} "
            f"f1={ev.f1:.3f} accuracy={ev.accuracy:.3f}",
            f"  baseline (always-signed): precision={ev.baseline_precision:.3f} "
            f"recall=1.000 f1={ev.baseline_f1:.3f}",
            f"  Δf1 vs baseline: {ev.f1 - ev.baseline_f1:+.3f}",
        ]
    )


# ---------------------------------------------------------------- impure runner

class DocSignature(BaseModel):
    name: str
    gold_signed: bool
    prediction: SignaturePrediction


def run_signature(
    image_dir: Path,
    gt_dir: Path,
    cache_dir: Path,
    settings,
    cap: int = 100,
    detect_fn: Callable[[DocumentIR], SignaturePrediction] = detect_signature,
) -> tuple[list[DocSignature], SignatureEval]:
    """Convert Tobacco800 scans → parse (paddle, IR-cached, shared with realscan) →
    detect signature, and score against GEDI groundtruth. Only docs that have a GEDI
    XML with at least one zone are scored (the labeled set)."""
    from contract_rag.eval.gedi import has_signature_zone, parse_gedi
    from contract_rag.eval.ir_cache import ir_cache
    from contract_rag.eval.realscan import list_input_docs
    from contract_rag.eval.scanio import IMAGE_SUFFIXES, ensure_pdf
    from contract_rag.parse.probe import probe_document
    from contract_rag.parse.router import _default_adapters, route

    cache_dir = Path(cache_dir)
    real_adapters = _default_adapters()
    adapters = {
        eng: (lambda e, fn: lambda p, s: ir_cache(cache_dir / "ir" / e, lambda pp: fn(pp, s))(p))(
            eng, fn
        )
        for eng, fn in real_adapters.items()
    }

    docs = list_input_docs(image_dir, cap)
    results: list[DocSignature] = []
    for doc in docs:
        if doc.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        xml = Path(gt_dir) / f"{doc.stem}.xml"
        if not xml.exists():
            continue  # no GEDI groundtruth → no label
        # Tobacco800 annotates signatures AND logos comprehensively, so a page whose
        # GEDI XML carries no DLSignature zone (including a zero-zone page) is a genuine
        # negative — not "unannotated". Every page with an XML is a labeled example.
        pz = parse_gedi(xml.read_text(errors="replace"))
        pdf = ensure_pdf(doc, cache_dir / "pdf")
        engine = route(probe_document(pdf), settings)
        ir = adapters[engine](pdf, settings)
        results.append(
            DocSignature(
                name=doc.stem,
                gold_signed=has_signature_zone(pz),
                prediction=detect_fn(ir),
            )
        )
    if not results:
        raise SystemExit(
            f"no GEDI-annotated scans found under {image_dir} / {gt_dir}"
        )
    ev = evaluate_predictions([(r.prediction.signed, r.gold_signed) for r in results])
    return results, ev


def main() -> None:
    import json
    import os

    from contract_rag.config import get_settings

    default_cache = Path.home() / ".cache" / "contract-rag" / "realscan"
    cache = Path(os.environ.get("SIGNATURE_CACHE", str(default_cache)))
    image_dir = os.environ.get("SIGNATURE_DIR") or os.environ.get("REALSCAN_DIR")
    gt_dir = os.environ.get("SIGNATURE_GT_DIR") or os.environ.get("REALSCAN_GT_DIR")
    if not image_dir or not gt_dir:
        raise SystemExit(
            "set SIGNATURE_DIR (Tobacco800 TIFF dir) and SIGNATURE_GT_DIR (GEDI XML dir)"
        )
    cap = int(os.environ.get("SIGNATURE_SET_SIZE", "100"))
    results, ev = run_signature(
        Path(image_dir), Path(gt_dir), cache, get_settings(), cap=cap
    )
    print(format_eval(ev))
    # a few misclassifications for inspection
    fps = [r for r in results if r.prediction.signed and not r.gold_signed]
    fns = [r for r in results if not r.prediction.signed and r.gold_signed]
    if fps:
        print(f"\nfalse positives ({len(fps)}): " + ", ".join(r.name for r in fps[:10]))
    if fns:
        print(f"false negatives ({len(fns)}): " + ", ".join(r.name for r in fns[:10]))
    out = os.environ.get("SIGNATURE_OUT")
    if out:
        payload = {
            "eval": ev.model_dump(),
            "results": [r.model_dump() for r in results],
        }
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
