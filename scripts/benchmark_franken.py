"""FrankenOCR vs cached PaddleOCR — a two-arm benchmark driver, not a `contract_rag`
module (`focr` is an external CPU-only OCR binary, not a project dependency; this
stays a script so `src/` is untouched).

Compares `parse_with_franken` (src/contract_rag/parse/franken_parser.py) against the
PaddleOCR results already cached by two existing eval harnesses:

  realscan arm — 100 real Tobacco800 scans (`eval/realscan.py`'s cache). Metrics per
  doc: quality_score, block count, block-text char count, franken wall-clock. Paddle
  wall-clock isn't cached; `--time-paddle N` re-runs paddle on N docs for a timing
  sample (paddleocr import is guarded so the script still runs without it installed).

  degrade arm — CUAD pages degraded by `eval/degrade.py` at a named intensity
  (light/medium/fax/shred), already OCR'd by paddle and cached. Metrics per doc:
  quality_score AND field-F1/source-accuracy (rule extractor against the golden set),
  franken vs paddle, on the identical degraded PDF. Field-F1 reuses `eval.metrics`'
  `row_for`/`aggregate` — the exact functions `eval/degrade.py`'s own per-doc `_f1`
  closure calls — rather than reimplementing scoring; golden-set loading reuses
  `eval.golden.load_golden_set` the same way `eval/degrade.py.run_degrade` does. If
  the golden set / extractor can't be loaded, the run degrades to quality-only and
  says so, rather than crashing.

Franken IRs are cached under ~/.cache/contract-rag/franken-bench/<arm>/ir/... using
the exact `contract_rag.eval.ir_cache.ir_cache` serialization (so cache files are
byte-compatible with the rest of the codebase's IR caches), so an interrupted run
(focr takes minutes/doc on CPU) loses no completed work on resume. One doc's focr
failure (RuntimeError) is caught and recorded — it never kills the run.

Run:
  FRANKEN_BIN=/path/to/focr uv run python scripts/benchmark_franken.py realscan --size 12
  FRANKEN_BIN=/path/to/focr uv run python scripts/benchmark_franken.py degrade --level shred

Sanity-check the plumbing before focr's weights are available:
  uv run python scripts/benchmark_franken.py realscan --size 2 --paddle-only
  EXTRACT_BACKEND=rule uv run python scripts/benchmark_franken.py degrade --level light --paddle-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

from contract_rag.clean.quality import compute_quality_score
from contract_rag.config import Settings, get_settings
from contract_rag.ir import DocumentIR
from contract_rag.parse.franken_parser import parse_with_franken

_CACHE_ROOT = Path.home() / ".cache" / "contract-rag"
_REALSCAN_CACHE = _CACHE_ROOT / "realscan"
_DEGRADE_CALIB_CACHE = _CACHE_ROOT / "degrade-calib"
_BENCH_CACHE = _CACHE_ROOT / "franken-bench"
_LEVELS = ("light", "medium", "fax", "shred")


# ============================================================ pure helpers

def list_docs(dir_: Path, pattern: str, cap: int | None) -> list[Path]:
    """Deterministic (name-sorted) file list, optionally capped."""
    docs = sorted(Path(dir_).glob(pattern), key=lambda p: p.name)
    if cap is not None and cap > 0:
        return docs[:cap]
    return docs


def load_cached_ir(cache_file: Path) -> DocumentIR | None:
    """Load a `DocumentIR` written by `contract_rag.eval.ir_cache.ir_cache`
    (`{stem}.ir.json`), or None if it isn't cached."""
    cache_file = Path(cache_file)
    if not cache_file.exists():
        return None
    return DocumentIR.model_validate_json(cache_file.read_text())


def ir_metrics(ir: DocumentIR) -> dict:
    """quality_score + block count + total block-text chars — the core comparison
    that doesn't need a golden set."""
    q = compute_quality_score(ir)
    return {
        "quality_score": round(q.quality_score, 4),
        "n_blocks": len(ir.blocks),
        "total_chars": sum(len(b.text) for b in ir.blocks),
    }


def make_franken_ir_cache(
    cache_dir: Path, settings: Settings
) -> Callable[[Path], tuple[DocumentIR, float | None, bool]]:
    """Wraps `contract_rag.eval.ir_cache.ir_cache` (same serialization as every other
    IR cache in the repo) so re-runs skip already-parsed docs, while also reporting
    per-call wall-clock seconds and whether the result came from cache (elapsed is
    None on a cache hit, so cache hits never pollute the timing stats)."""
    from contract_rag.eval.ir_cache import ir_cache

    parse_fn = lambda p: parse_with_franken(p, settings)  # noqa: E731
    cached = ir_cache(cache_dir, parse_fn)

    def _load(pdf_path: Path) -> tuple[DocumentIR, float | None, bool]:
        cache_file = Path(cache_dir) / f"{Path(pdf_path).stem}.ir.json"
        was_cached = cache_file.exists()
        t0 = time.time()
        ir = cached(pdf_path)
        elapsed = time.time() - t0
        return ir, (None if was_cached else round(elapsed, 2)), was_cached

    return _load


def field_f1_for(ir: DocumentIR, gold, vertical, extractor) -> tuple[float | None, float | None]:
    """Single-doc field-F1 / source-accuracy, via the exact `eval.metrics` functions
    `eval/degrade.py`'s own per-doc `_f1` closure calls (`row_for` + `aggregate` on a
    one-row list) — reused rather than reimplemented. None/None if gold or an
    extractor isn't available for this doc."""
    if gold is None or extractor is None:
        return None, None
    from contract_rag.eval.metrics import aggregate, row_for

    row = row_for(extractor.extract(ir), gold, ir, vertical)
    agg = aggregate([row], vertical)
    return round(agg["field_f1"], 3), round(agg["source_accuracy"], 3)


def _stats(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize_rows(rows: list[dict], with_f1: bool = False) -> dict:
    paddle_q = [r["paddle"]["quality_score"] for r in rows if r["paddle"]]
    franken_q = [r["franken"]["quality_score"] for r in rows if r["franken"]]
    franken_secs = [r["franken_seconds"] for r in rows if r["franken_seconds"] is not None]
    n_errors = sum(1 for r in rows if r["error"])
    out = {
        "n_docs": len(rows),
        "n_errors": n_errors,
        "paddle_mean_quality": _mean(paddle_q),
        "franken_mean_quality": _mean(franken_q),
        "franken_seconds": _stats(franken_secs),
    }
    if with_f1:
        paddle_f1 = [r["paddle"]["field_f1"] for r in rows if r["paddle"] and r["paddle"]["field_f1"] is not None]
        franken_f1 = [r["franken"]["field_f1"] for r in rows if r["franken"] and r["franken"]["field_f1"] is not None]
        paddle_src = [r["paddle"]["source_accuracy"] for r in rows if r["paddle"] and r["paddle"]["source_accuracy"] is not None]
        franken_src = [r["franken"]["source_accuracy"] for r in rows if r["franken"] and r["franken"]["source_accuracy"] is not None]
        out.update(
            paddle_mean_field_f1=_mean(paddle_f1),
            franken_mean_field_f1=_mean(franken_f1),
            paddle_mean_source_accuracy=_mean(paddle_src),
            franken_mean_source_accuracy=_mean(franken_src),
        )
    return out


def _fmt(x, p: int = 3) -> str:
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def format_table(rows: list[dict], with_f1: bool = False) -> str:
    if with_f1:
        header = (
            f"{'doc':<40} {'pdl_q':>6} {'frk_q':>6} {'pdl_f1':>7} {'frk_f1':>7}"
            f" {'frk_s':>7} {'note':<20}"
        )
    else:
        header = (
            f"{'doc':<40} {'pdl_q':>6} {'frk_q':>6} {'pdl_ch':>7} {'frk_ch':>7}"
            f" {'frk_s':>7} {'note':<20}"
        )
    lines = [header]
    for r in rows:
        pdl = r["paddle"]
        frk = r["franken"]
        note = r["error"] or ("cache" if r["franken_cached"] else "")
        if with_f1:
            lines.append(
                f"{r['doc'][:40]:<40} {_fmt(pdl['quality_score']) if pdl else '—':>6}"
                f" {_fmt(frk['quality_score']) if frk else '—':>6}"
                f" {_fmt(pdl['field_f1']) if pdl else '—':>7}"
                f" {_fmt(frk['field_f1']) if frk else '—':>7}"
                f" {_fmt(r['franken_seconds'], 1) if r['franken_seconds'] is not None else '—':>7}"
                f" {note[:20]:<20}"
            )
        else:
            lines.append(
                f"{r['doc'][:40]:<40} {_fmt(pdl['quality_score']) if pdl else '—':>6}"
                f" {_fmt(frk['quality_score']) if frk else '—':>6}"
                f" {pdl['total_chars'] if pdl else '—':>7}"
                f" {frk['total_chars'] if frk else '—':>7}"
                f" {_fmt(r['franken_seconds'], 1) if r['franken_seconds'] is not None else '—':>7}"
                f" {note[:20]:<20}"
            )
    return "\n".join(lines)


def format_summary(summary: dict, with_f1: bool = False) -> str:
    lines = [
        "",
        f"docs={summary['n_docs']} errors={summary['n_errors']}",
        f"mean quality: paddle={_fmt(summary['paddle_mean_quality'])}"
        f" franken={_fmt(summary['franken_mean_quality'])}",
    ]
    if with_f1:
        lines.append(
            f"mean field_f1: paddle={_fmt(summary['paddle_mean_field_f1'])}"
            f" franken={_fmt(summary['franken_mean_field_f1'])}"
        )
        lines.append(
            f"mean source_accuracy: paddle={_fmt(summary['paddle_mean_source_accuracy'])}"
            f" franken={_fmt(summary['franken_mean_source_accuracy'])}"
        )
    fs = summary["franken_seconds"]
    lines.append(
        f"franken wall-clock (fresh parses only, n={fs['n']}): "
        f"mean={_fmt(fs['mean'], 1)}s min={_fmt(fs['min'], 1)}s max={_fmt(fs['max'], 1)}s"
    )
    return "\n".join(lines)


# ============================================================ realscan arm

def run_realscan_arm(args: argparse.Namespace) -> None:
    settings = get_settings()
    pdf_dir = _REALSCAN_CACHE / "pdf"
    paddle_ir_dir = _REALSCAN_CACHE / "ir" / "paddleocr"
    franken_ir_dir = _BENCH_CACHE / "realscan" / "ir" / "frankenocr"

    docs = list_docs(pdf_dir, "*.pdf", args.size)
    if not docs:
        raise SystemExit(f"no PDFs found in {pdf_dir} — is REALSCAN_DIR's cache built?")

    franken_loader = None if args.paddle_only else make_franken_ir_cache(franken_ir_dir, settings)

    rows: list[dict] = []
    for i, pdf_path in enumerate(docs, start=1):
        stem = pdf_path.stem
        row: dict = {
            "doc": stem, "paddle": None, "franken": None,
            "franken_seconds": None, "franken_cached": None, "error": None,
        }
        paddle_ir = load_cached_ir(paddle_ir_dir / f"{stem}.ir.json")
        if paddle_ir is None:
            row["error"] = "missing cached paddle IR"
        else:
            row["paddle"] = ir_metrics(paddle_ir)

        if franken_loader is not None:
            try:
                ir, secs, cached = franken_loader(pdf_path)
                row["franken"] = ir_metrics(ir)
                row["franken_seconds"] = secs
                row["franken_cached"] = cached
            except Exception as exc:  # focr RuntimeError, or anything else — never fatal
                row["error"] = ((row["error"] + "; ") if row["error"] else "") + f"franken failed: {exc}"

        rows.append(row)
        print(f"[{i}/{len(docs)}] {stem}: {row['error'] or 'ok'}", flush=True)

    paddle_timing = None
    if args.time_paddle and not args.paddle_only:
        paddle_timing = _time_paddle_sample(docs[: args.time_paddle])

    summary = summarize_rows(rows)
    print()
    print(format_table(rows))
    print(format_summary(summary))
    if paddle_timing is not None:
        print(
            f"paddle wall-clock sample (n={paddle_timing['n']}): "
            f"mean={_fmt(paddle_timing['mean'], 1)}s min={_fmt(paddle_timing['min'], 1)}s"
            f" max={_fmt(paddle_timing['max'], 1)}s"
        )

    _write_results(
        "realscan",
        {
            "arm": "realscan",
            "size": len(docs),
            "paddle_only": args.paddle_only,
            "rows": rows,
            "summary": summary,
            "paddle_timing_sample": paddle_timing,
        },
    )


def _time_paddle_sample(docs: list[Path]) -> dict | None:
    """Re-parse N docs with paddleocr directly (not cached) purely for a wall-clock
    sample — realscan's cached paddle IRs carry no timing. Guarded: if paddleocr
    isn't importable/installed, warn and return None rather than crash."""
    try:
        from contract_rag.parse.paddle_parser import parse_with_paddle
    except Exception as exc:
        print(f"warning: --time-paddle skipped, paddleocr unavailable ({exc})", file=sys.stderr)
        return None

    secs: list[float] = []
    for pdf_path in docs:
        try:
            t0 = time.time()
            parse_with_paddle(pdf_path)
            secs.append(time.time() - t0)
        except Exception as exc:
            print(f"warning: paddle timing failed on {pdf_path.name}: {exc}", file=sys.stderr)
    return _stats(secs)


# ============================================================ degrade arm

def run_degrade_arm(args: argparse.Namespace) -> None:
    settings = get_settings()
    level = args.level
    pdf_dir = _DEGRADE_CALIB_CACHE / "pdf" / f"{level}_s0"
    paddle_ir_dir = _DEGRADE_CALIB_CACHE / "ir" / f"{level}_s0"
    franken_ir_dir = _BENCH_CACHE / "degrade" / "ir" / f"{level}_s0"

    if not pdf_dir.exists():
        raise SystemExit(
            f"no degraded PDFs at {pdf_dir}; build the degrade-calib cache first with "
            f"`DEGRADE_LEVEL={level} DEGRADE_CACHE={_DEGRADE_CALIB_CACHE} "
            "uv run python -m contract_rag.eval.degrade`"
        )

    docs = list_docs(pdf_dir, "*.pdf", args.size if args.size else None)
    if not docs:
        raise SystemExit(f"no PDFs found in {pdf_dir}")

    # Reuse eval.degrade's own seams: golden-set loading + the row_for/aggregate
    # field-F1 flow its per-doc `_f1` closure uses. If either import/load fails
    # (no golden set built, etc.) fall back to a quality-only run and say so.
    golden_by_stem: dict[str, object] = {}
    extractor = vertical = None
    with_f1 = True
    try:
        from contract_rag.eval.golden import load_golden_set

        golden_by_stem = {
            Path(g.source_pdf).stem: g for g in load_golden_set(settings.golden_set_dir)
        }
        from contract_rag.extract.extractor import get_extractor
        from contract_rag.verticals.registry import get_vertical_for

        vertical = get_vertical_for(settings)
        extractor = get_extractor(settings, vertical)
    except Exception as exc:
        print(
            f"warning: golden set / extractor unavailable ({exc}); "
            "falling back to quality-only (no field-F1)",
            file=sys.stderr,
        )
        with_f1 = False

    franken_loader = None if args.paddle_only else make_franken_ir_cache(franken_ir_dir, settings)

    rows: list[dict] = []
    for i, pdf_path in enumerate(docs, start=1):
        stem = pdf_path.stem
        gold = golden_by_stem.get(stem)
        row: dict = {
            "doc": stem, "paddle": None, "franken": None,
            "franken_seconds": None, "franken_cached": None, "error": None,
        }
        paddle_ir = load_cached_ir(paddle_ir_dir / f"{stem}.ir.json")
        if paddle_ir is None:
            row["error"] = "missing cached paddle IR"
        else:
            m = ir_metrics(paddle_ir)
            if with_f1:
                f1, src = field_f1_for(paddle_ir, gold, vertical, extractor)
                m["field_f1"], m["source_accuracy"] = f1, src
            row["paddle"] = m

        if franken_loader is not None:
            try:
                ir, secs, cached = franken_loader(pdf_path)
                m = ir_metrics(ir)
                if with_f1:
                    f1, src = field_f1_for(ir, gold, vertical, extractor)
                    m["field_f1"], m["source_accuracy"] = f1, src
                row["franken"] = m
                row["franken_seconds"] = secs
                row["franken_cached"] = cached
            except Exception as exc:
                row["error"] = ((row["error"] + "; ") if row["error"] else "") + f"franken failed: {exc}"

        rows.append(row)
        print(f"[{i}/{len(docs)}] {stem}: {row['error'] or 'ok'}", flush=True)

    summary = summarize_rows(rows, with_f1=with_f1)
    print()
    print(format_table(rows, with_f1=with_f1))
    print(format_summary(summary, with_f1=with_f1))

    _write_results(
        f"degrade-{level}",
        {
            "arm": "degrade",
            "level": level,
            "size": len(docs),
            "paddle_only": args.paddle_only,
            "with_f1": with_f1,
            "rows": rows,
            "summary": summary,
        },
    )


# ============================================================ output + CLI

def _write_results(name: str, payload: dict) -> None:
    _BENCH_CACHE.mkdir(parents=True, exist_ok=True)
    out_path = _BENCH_CACHE / f"results-{name}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="arm", required=True)

    p_real = sub.add_parser("realscan", help="Tobacco800 real scans vs cached paddle")
    p_real.add_argument("--size", type=int, default=12, help="cap on doc count (default 12)")
    p_real.add_argument(
        "--paddle-only", action="store_true",
        help="skip franken entirely; just load cached paddle IRs and score them "
             "(plumbing sanity check before focr weights are available)",
    )
    p_real.add_argument(
        "--time-paddle", type=int, default=0, metavar="N",
        help="re-run paddle (not cached) on the first N docs for a wall-clock sample",
    )

    p_deg = sub.add_parser("degrade", help="degraded CUAD pages vs cached paddle")
    p_deg.add_argument("--level", required=True, choices=_LEVELS)
    p_deg.add_argument(
        "--size", type=int, default=0, help="cap on doc count (default: all docs at this level)"
    )
    p_deg.add_argument("--paddle-only", action="store_true", help="skip franken entirely")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.arm == "realscan":
        run_realscan_arm(args)
    elif args.arm == "degrade":
        run_degrade_arm(args)
    else:  # pragma: no cover - argparse enforces choices
        raise SystemExit(f"unknown arm {args.arm!r}")


if __name__ == "__main__":
    main()
