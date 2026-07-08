"""FinCriticalED fact-level omission measurement + OCR-confidence calibration.

Why: the Tobacco800 realscan run refuted "real scans score low quality" — the
quality formula scores what OCR *emitted*, not what it *missed*, so omissions are
invisible to it; block-level OCR confidence, however, is a real signal. This
harness (1) makes omissions measurable against expert fact-level ground truth and
(2) calibrates block confidence -> fact-survival accuracy, so `verify()`'s HITL
confidence threshold can be data-driven instead of a hand-picked constant.

Dataset: HuggingFace `TheFinAI/FinCriticalED` (Apache-2.0; gated "auto" — accept
once on the dataset page with a logged-in account, then `hf auth login`). The 848
public samples are real degraded SEC EDGAR financial pages: `raw_input.csv` with
columns (id, image = base64 PNG of the rendered page, matched_html), plus
`gold_annotation_html/gold_{id}.txt` — the page HTML with expert-annotated fact
tags <number> <temporal> <monetaryunit> <reportingentity> <financialconcepts>.
Dataset files are never committed (cache dirs are gitignored).

Method per page (base64 PNG -> single-page PDF -> parse router -> paddleocr IR):

a) OMISSION: the fraction of gold facts whose canonicalized value appears nowhere
   in the parsed IR text — the quality-formula blind spot, made measurable.
   Reported overall and per fact kind.
b) CALIBRATION: each gold fact is LOCATED by its surrounding context tokens
   (rarity-weighted overlap; the value's own tokens are excluded so a garbled
   value cannot hide its block). "Correct" = the value survives, canonicalized,
   in the located block or its ±`NEIGHBORHOOD` reading-order neighbours (tables
   put a value and its row label on different OCR lines). The paired confidence
   is the matching block's when found, else the context block's (a proxy for
   local OCR quality). Output: a reliability table (confidence bins -> accuracy,
   n), a garbled-vs-clean split, and the minimum confidence achieving each
   target accuracy — the number `verify()`'s tier thresholds are derived from.

Canonicalization (`canon_fact_text`, applied identically to gold values and OCR
text — tolerant to formatting, strict on meaning): html-unescape, NFKC,
lowercase; thousands separators dropped (1,200,000 == 1200000) and currency
symbols dropped, so pure formatting differences never count as OCR errors;
decimal points, minus signs and '%' attached to digits are KEPT — a shifted
decimal or a dropped sign IS a critical error and must never be normalized away;
all other punctuation becomes a token boundary. Matching is token-boundary
substring (" 3.5 " never matches inside "13.5"). Known limits, by design:
accounting-style negatives ("(5.2)" vs "-5.2") are treated as different, and a
value OCR-split mid-word across two lines counts as lost.

Env: FINCRITICAL_DIR (a local snapshot; default auto-download to
~/.cache/contract-rag/fincriticaled), FINCRITICAL_SET_SIZE (default 100 — paddle
OCR is the slow step; parses are IR-cached so re-runs are fast),
FINCRITICAL_CACHE (png/pdf/IR cache), FINCRITICAL_OUT (optional JSON dump).

Run: uv run python -m contract_rag.eval.fincritical
"""
from __future__ import annotations

import base64
import csv
import html as html_lib
import re
import unicodedata
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from contract_rag.clean.quality import compute_quality_score, is_garbled
from contract_rag.config import Settings
from contract_rag.eval.scanio import ensure_pdf
from contract_rag.ir import DocBlock, DocumentIR
from contract_rag.text import tokenize

HF_REPO = "TheFinAI/FinCriticalED"
FACT_KINDS = ("number", "temporal", "monetaryunit", "reportingentity", "financialconcepts")
NEIGHBORHOOD = 2  # reading-order window for "value survives near its context"

_FACT_TAG_RE = re.compile(
    rf"<({'|'.join(FACT_KINDS)})\b[^>]*>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL
)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_CURRENCY = set("$€£¥")


# ------------------------------------------------------------- canonicalization

def strip_tags(text: str) -> str:
    """HTML -> whitespace-collapsed plain text (tags become spaces)."""
    return " ".join(html_lib.unescape(_ANY_TAG_RE.sub(" ", text)).split())


def canon_fact_text(s: str) -> str:
    """The tolerant-but-meaning-preserving canonicalizer (see module docstring).
    Applied to BOTH sides of every match, so it can never bias one side."""
    s = unicodedata.normalize("NFKC", html_lib.unescape(s)).lower()
    s = _THOUSANDS_RE.sub("", s)
    out: list[str] = []
    n = len(s)
    for i, ch in enumerate(s):
        if ch.isalnum():
            out.append(ch)
            continue
        prev = s[i - 1] if i else " "
        nxt = s[i + 1] if i + 1 < n else " "
        if ch == "." and prev.isdigit() and nxt.isdigit():
            out.append(ch)  # decimal point
        elif ch == "-" and nxt.isdigit() and not prev.isdigit():
            out.append(ch)  # numeric sign (a digit-digit hyphen is a range -> boundary)
        elif ch == "%" and prev.isdigit():
            out.append(ch)
        elif ch in _CURRENCY:
            pass  # currency symbol drops without becoming a boundary ($1200 -> 1200)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def contains_value(canon_value: str, canon_text: str) -> bool:
    """Token-boundary substring on already-canonicalized strings."""
    return bool(canon_value) and f" {canon_value} " in f" {canon_text} "


# ------------------------------------------------------------------ gold facts

class GoldFact(BaseModel):
    kind: str
    value: str    # tag inner text, nested tags stripped
    context: str  # surrounding plain text (locator; the value's own tokens are excluded later)


def parse_gold_html(text: str, context_chars: int = 300) -> list[GoldFact]:
    """Extract the expert fact tags from a gold_{id}.txt page. The raw context
    window is 4x `context_chars` because markup inflates raw HTML; after tag
    stripping it is trimmed to `context_chars` on each side."""
    raw_window = 4 * context_chars
    facts: list[GoldFact] = []
    for m in _FACT_TAG_RE.finditer(text):
        value = strip_tags(m.group(2))
        if not value:
            continue
        before = strip_tags(text[max(0, m.start() - raw_window) : m.start()])[-context_chars:]
        after = strip_tags(text[m.end() : m.end() + raw_window])[:context_chars]
        facts.append(GoldFact(kind=m.group(1).lower(), value=value, context=f"{before} {after}"))
    return facts


# ------------------------------------------------------------ locating + scoring

def locate_block(fact: GoldFact, block_token_sets: list[set[str]]) -> int | None:
    """Index of the block whose tokens best match the fact's CONTEXT tokens,
    weighted by rarity across blocks (1/df) so boilerplate words can't dominate.
    The value's own tokens are excluded — a garbled value must not hide its block.
    Requires >= 2 distinct context-token hits (1 for single-token contexts)."""
    value_toks = set(tokenize(canon_fact_text(fact.value)))
    ctx_toks = {t for t in tokenize(canon_fact_text(fact.context))} - value_toks
    if not ctx_toks or not block_token_sets:
        return None
    weights = {
        t: 1.0 / (1 + sum(t in bs for bs in block_token_sets)) for t in ctx_toks
    }
    best_i, best_score, best_hits = None, 0.0, 0
    for i, bs in enumerate(block_token_sets):
        hit = ctx_toks & bs
        score = sum(weights[t] for t in hit)
        if score > best_score:
            best_i, best_score, best_hits = i, score, len(hit)
    if best_i is None or best_hits < min(2, len(ctx_toks)):
        return None
    return best_i


class FactOutcome(BaseModel):
    kind: str
    value: str
    in_document: bool                # value's canon span survives ANYWHERE in the OCR
    located: bool                    # a block confidence could be paired to this fact
    confidence: float | None = None  # paired block conf: the value-block if survived,
    #                                  else the context-located block (where it belonged)
    garbled: bool | None = None      # is_garbled() of the paired block
    correct: bool | None = None      # == in_document, for the located facts (calibration)


def _value_block(cv: str, canon_blocks: list[str], neighborhood: int) -> int | None:
    """Index of the block that carries the value. Direct block-substring first;
    then a ±neighborhood reading-order window, so a value OCR-split across two lines
    (its digits on one block, unit on the next) still resolves — attributed to the
    block holding the value's first token (its numeric/lead word), i.e. where the
    critical part was read."""
    direct = next((i for i, cb in enumerate(canon_blocks) if contains_value(cv, cb)), None)
    if direct is not None:
        return direct
    first_tok = cv.split()[0]
    for i in range(len(canon_blocks)):
        lo, hi = max(0, i - neighborhood), min(len(canon_blocks), i + neighborhood + 1)
        if contains_value(cv, " ".join(canon_blocks[lo:hi])):
            core = next(
                (j for j in range(lo, hi) if contains_value(first_tok, canon_blocks[j])), i
            )
            return core
    return None


def evaluate_page(
    ir: DocumentIR, facts: list[GoldFact], neighborhood: int = NEIGHBORHOOD
) -> list[FactOutcome]:
    """Pure: score every measurable gold fact against one parsed page IR.

    `correct` (fact survived OCR) is whole-document, NOT restricted to the
    context-located block — these rendered financial pages fragment into hundreds
    of OCR lines whose reading order diverges from the gold HTML, so a present
    value routinely sits far from its textual context; conflating that locator
    noise with OCR error would understate survival. The paired confidence is the
    block that actually holds the value (survived) or, for an omission, the
    context-located block — the local OCR quality where the fact belonged.
    Facts with an empty canonical value (e.g. a bare ``$`` monetaryunit marker)
    are unmeasurable and dropped."""
    blocks: list[DocBlock] = ir.blocks
    canon_blocks = [canon_fact_text(b.text) for b in blocks]
    block_token_sets = [set(tokenize(cb)) for cb in canon_blocks]

    outcomes: list[FactOutcome] = []
    for fact in facts:
        cv = canon_fact_text(fact.value)
        if not cv:
            continue  # unmeasurable (bare symbol / punctuation-only tag)
        vidx = _value_block(cv, canon_blocks, neighborhood)
        in_doc = vidx is not None
        paired = vidx if in_doc else locate_block(fact, block_token_sets)
        if paired is None:
            # survived-but-somehow-unindexed can't happen; a true omission we also
            # couldn't context-locate → no confidence to pair, excluded from calibration
            outcomes.append(FactOutcome(
                kind=fact.kind, value=fact.value, in_document=in_doc, located=False,
            ))
            continue
        b = blocks[paired]
        outcomes.append(FactOutcome(
            kind=fact.kind, value=fact.value, in_document=in_doc, located=True,
            confidence=b.confidence, garbled=is_garbled(b.text), correct=in_doc,
        ))
    return outcomes


# ----------------------------------------------------------------- aggregation

DEFAULT_BIN_EDGES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0)


class ReliabilityBin(BaseModel):
    lo: float
    hi: float
    n: int
    accuracy: float | None  # None when the bin is empty


def reliability_table(
    pairs: list[tuple[float, bool]], edges: tuple[float, ...] = DEFAULT_BIN_EDGES
) -> list[ReliabilityBin]:
    """(block confidence, fact survived?) pairs -> accuracy per confidence bin.
    Last bin is closed on the right so conf == 1.0 lands in it."""
    bins: list[ReliabilityBin] = []
    for lo, hi in zip(edges, edges[1:]):
        last = hi == edges[-1]
        sel = [ok for conf, ok in pairs if lo <= conf < hi or (last and conf == hi)]
        bins.append(ReliabilityBin(
            lo=lo, hi=hi, n=len(sel),
            accuracy=round(sum(sel) / len(sel), 3) if sel else None,
        ))
    return bins


def threshold_for_accuracy(
    pairs: list[tuple[float, bool]],
    target: float,
    candidates: tuple[float, ...] = DEFAULT_BIN_EDGES,
    min_n: int = 30,
) -> float | None:
    """Smallest candidate c such that facts with confidence >= c survive OCR with
    accuracy >= target (on at least `min_n` facts). None if no c achieves it —
    an honest 'the signal can't buy you this accuracy' answer."""
    for c in sorted(candidates):
        sel = [ok for conf, ok in pairs if conf >= c]
        if len(sel) >= min_n and sum(sel) / len(sel) >= target:
            return c
    return None


class KindStats(BaseModel):
    n_gold: int
    n_omitted: int
    omission_rate: float
    n_scored: int                 # located facts contributing calibration pairs
    accuracy: float | None        # survival accuracy among scored


class PageResult(BaseModel):
    page_id: int
    n_facts: int
    n_omitted: int
    quality_score: float          # the formula that can't see the omissions
    mean_confidence: float
    outcomes: list[FactOutcome]


class Summary(BaseModel):
    n_pages: int
    n_facts: int
    omission_rate: float
    by_kind: dict[str, KindStats] = Field(default_factory=dict)
    n_located: int
    n_unlocated: int
    reliability: list[ReliabilityBin]
    accuracy_overall: float | None
    accuracy_garbled: float | None
    accuracy_clean: float | None
    n_garbled: int
    n_clean: int
    thresholds: dict[str, float | None] = Field(default_factory=dict)  # "0.95" -> conf
    mean_quality: float           # mean page quality_score, for the blind-spot headline
    quality_vs_omission: list[tuple[float, float]]  # (quality, page omission rate)


def summarize(
    results: list[PageResult], targets: tuple[float, ...] = (0.9, 0.95, 0.97, 0.99)
) -> Summary:
    if not results:
        raise ValueError("no pages evaluated")
    outcomes = [o for r in results for o in r.outcomes]
    located = [o for o in outcomes if o.located]
    pairs = [(o.confidence, bool(o.correct)) for o in located]
    garbled = [bool(o.correct) for o in located if o.garbled]
    clean = [bool(o.correct) for o in located if not o.garbled]

    by_kind: dict[str, KindStats] = {}
    for kind in FACT_KINDS:
        ks = [o for o in outcomes if o.kind == kind]
        if not ks:
            continue
        scored = [o for o in ks if o.located]
        by_kind[kind] = KindStats(
            n_gold=len(ks),
            n_omitted=sum(not o.in_document for o in ks),
            omission_rate=round(sum(not o.in_document for o in ks) / len(ks), 3),
            n_scored=len(scored),
            accuracy=round(sum(bool(o.correct) for o in scored) / len(scored), 3)
            if scored else None,
        )

    def _acc(sel: list[bool]) -> float | None:
        return round(sum(sel) / len(sel), 3) if sel else None

    n_facts = len(outcomes)
    return Summary(
        n_pages=len(results),
        n_facts=n_facts,
        omission_rate=round(sum(not o.in_document for o in outcomes) / n_facts, 3),
        by_kind=by_kind,
        n_located=len(pairs),
        n_unlocated=n_facts - len(pairs),
        reliability=reliability_table(pairs),
        accuracy_overall=_acc([ok for _, ok in pairs]),
        accuracy_garbled=_acc(garbled),
        accuracy_clean=_acc(clean),
        n_garbled=len(garbled),
        n_clean=len(clean),
        thresholds={str(t): threshold_for_accuracy(pairs, t) for t in targets},
        mean_quality=round(sum(r.quality_score for r in results) / len(results), 3),
        quality_vs_omission=[
            (r.quality_score, round(r.n_omitted / r.n_facts, 3))
            for r in results if r.n_facts
        ],
    )


def format_report(summary: Summary) -> str:
    s = summary
    lines = [
        "=== FinCriticalED fact-level omission + confidence calibration ===",
        f"pages={s.n_pages} gold_facts={s.n_facts}"
        f" mean_page_quality={s.mean_quality} (the formula that can't see omissions)",
        "",
        f"OMISSION RATE (value nowhere in parsed IR): {s.omission_rate:.1%}",
        f"{'kind':<18} {'gold':>5} {'omitted':>8} {'om.rate':>8} {'scored':>7} {'survival':>9}",
    ]
    for kind, ks in s.by_kind.items():
        lines.append(
            f"{kind:<18} {ks.n_gold:>5} {ks.n_omitted:>8} {ks.omission_rate:>8.1%}"
            f" {ks.n_scored:>7} {('%.3f' % ks.accuracy) if ks.accuracy is not None else '—':>9}"
        )
    lines += [
        "",
        f"CALIBRATION (located facts: {s.n_located}; unlocated, excluded: {s.n_unlocated})",
        f"{'conf bin':<16} {'n':>6} {'survival':>9}",
    ]
    for b in s.reliability:
        lines.append(
            f"[{b.lo:.2f}, {b.hi:.2f}) {b.n:>6}"
            f" {('%.3f' % b.accuracy) if b.accuracy is not None else '—':>9}"
        )
    lines.append(
        f"overall={s.accuracy_overall}"
        f"  garbled blocks: {s.accuracy_garbled} (n={s.n_garbled})"
        f"  clean blocks: {s.accuracy_clean} (n={s.n_clean})"
    )
    lines.append("min confidence for target survival accuracy:")
    for target, conf in s.thresholds.items():
        lines.append(f"  >= {float(target):.0%}: {conf if conf is not None else 'unreachable'}")
    return "\n".join(lines)


# -------------------------------------------------------------------- dataset IO

class Sample(BaseModel):
    page_id: int
    image_b64: str
    gold_html: str


def ensure_dataset(dir_: Path) -> Path:
    """Download raw_input.csv + gold_annotation_html/ to `dir_` if absent.
    The HF repo is gated (auto-approve): accept once on the dataset page with a
    logged-in account and authenticate (`hf auth login` or HF_TOKEN)."""
    dir_ = Path(dir_)
    if (dir_ / "raw_input.csv").exists():
        return dir_
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=HF_REPO, repo_type="dataset", local_dir=dir_,
            allow_patterns=["raw_input.csv", "gold_annotation_html/*"],
        )
    except Exception as exc:  # GatedRepoError / HTTP 401
        raise SystemExit(
            f"could not download {HF_REPO}: {exc}\n"
            "The dataset is gated: visit https://huggingface.co/datasets/"
            f"{HF_REPO}, click 'Agree and access', then `hf auth login` "
            "(or set HF_TOKEN), or point FINCRITICAL_DIR at a local snapshot."
        ) from exc
    return dir_


def load_samples(dir_: Path, cap: int = 100) -> list[Sample]:
    """Deterministic selection: ascending id, first `cap` rows that have a gold
    annotation file. Images are base64 PNG strings embedded in the CSV."""
    dir_ = Path(dir_)
    csv.field_size_limit(1 << 30)  # base64 page images blow the default field cap
    rows: list[tuple[int, str]] = []
    with (dir_ / "raw_input.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row["id"]), row["image"]))
    samples: list[Sample] = []
    for page_id, image_b64 in sorted(rows, key=lambda r: r[0]):
        if len(samples) >= max(0, cap):
            break
        gold = dir_ / "gold_annotation_html" / f"gold_{page_id}.txt"
        if not gold.exists():
            continue
        samples.append(Sample(
            page_id=page_id, image_b64=image_b64,
            gold_html=gold.read_text(encoding="utf-8", errors="replace"),
        ))
    return samples


def decode_image(image: str) -> bytes:
    """The CSV `image` column is a `data:image/png;base64,<payload>` data URI;
    tolerate a bare base64 payload too (a snapshot stored without the prefix)."""
    payload = image.split(",", 1)[1] if image.startswith("data:") else image
    return base64.b64decode(payload)


def sample_to_pdf(sample: Sample, cache_dir: Path) -> Path:
    """data-URI PNG -> cached PNG -> cached single-page PDF (scanio.ensure_pdf)."""
    cache_dir = Path(cache_dir)
    png_dir = cache_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png = png_dir / f"fincritical_{sample.page_id}.png"
    if not png.exists():
        png.write_bytes(decode_image(sample.image_b64))
    return ensure_pdf(png, cache_dir / "pdf")


# ----------------------------------------------------------------- impure runner

def run_fincritical(
    fin_dir: Path,
    settings: Settings,
    cache_dir: Path,
    cap: int = 100,
    parse_fn: Callable[[Path], DocumentIR] | None = None,
) -> tuple[list[PageResult], Summary]:
    """The real thing: select pages, wrap each image in a PDF, parse via the
    router (IR-cached per engine so re-runs are fast), score omission +
    calibration per page. `parse_fn` is the test seam."""
    cache_dir = Path(cache_dir)
    if parse_fn is None:
        from contract_rag.eval.ir_cache import ir_cache
        from contract_rag.parse.router import parse as router_parse

        parse_fn = ir_cache(cache_dir / "ir", lambda p: router_parse(p, settings))

    samples = load_samples(fin_dir, cap)
    if not samples:
        raise SystemExit(f"no usable samples (csv rows with a gold file) in {fin_dir}")

    results: list[PageResult] = []
    for sample in samples:
        pdf = sample_to_pdf(sample, cache_dir)
        ir = parse_fn(pdf)
        facts = parse_gold_html(sample.gold_html)
        if not facts:
            continue
        outcomes = evaluate_page(ir, facts)
        quality = compute_quality_score(ir)
        results.append(PageResult(
            page_id=sample.page_id,
            n_facts=len(outcomes),
            n_omitted=sum(not o.in_document for o in outcomes),
            quality_score=quality.quality_score,
            mean_confidence=quality.mean_confidence,
            outcomes=outcomes,
        ))
    return results, summarize(results)


def main() -> None:
    import json
    import os

    from contract_rag.config import get_settings

    cache = Path(os.environ.get(
        "FINCRITICAL_CACHE", str(Path.home() / ".cache" / "contract-rag" / "fincriticaled-run")
    ))
    fin_dir = os.environ.get("FINCRITICAL_DIR")
    if fin_dir:
        data_dir = Path(fin_dir)
        if not (data_dir / "raw_input.csv").exists():
            data_dir = ensure_dataset(data_dir)
    else:
        data_dir = ensure_dataset(
            Path.home() / ".cache" / "contract-rag" / "fincriticaled"
        )
    cap = int(os.environ.get("FINCRITICAL_SET_SIZE", "100"))
    results, summary = run_fincritical(data_dir, get_settings(), cache, cap=cap)
    print(format_report(summary))
    out = os.environ.get("FINCRITICAL_OUT")
    if out:
        payload = {
            "summary": summary.model_dump(),
            "pages": [r.model_dump() for r in results],
        }
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
