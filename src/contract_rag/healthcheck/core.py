"""One-command PoC health-check pipeline — pure orchestration.

Automates the manual G1 30-day-PoC run-book (`docs/fde/g1-poc-report-pack.md`):
point at a folder of a customer's worst contracts (arbitrary PDF/DOCX, real dirt —
not the simulated-dirt golden-set path) and get the full report pack back: one
self-contained HTML report per document (`demo.report` machinery), a combined
CLM-aligned facts export (`demo.export`), and a corpus-level summary (STP rate,
routing split, signature audit, per-field verified/quarantined counts).

Every engine call is composed, not modified — `demo.report.build_report_data` /
`render_html` / `stp_summary`, `demo.export.rows_from_report` / `serialize`,
`eval.signature.detect_signature`. This module only adds: (1) real-world file
discovery + DOCX dispatch, (2) per-document robustness (one bad file must not sink
the batch) with an injectable timeout seam, and (3) the corpus-level aggregation
the individual demo scripts don't need (they operate on one doc, or a pre-labeled
golden set).

Vertical-generic: fields come from whatever vertical `build_report_data`/
`rows_from_report` resolve (default: env `VERTICAL`, contract).
"""
from __future__ import annotations

import concurrent.futures
import functools
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from contract_rag.demo.export import rows_from_report, serialize
from contract_rag.demo.report import build_report_data, render_html, stp_summary
from contract_rag.eval.signature import detect_signature
from contract_rag.ir import DocumentIR

# Only these are routable today: PDF through the real parse router (digital →
# docling, scanned → paddle/VLM), DOCX straight to docling (it has no scanned-page
# concept in this pipeline — docling reads it natively, no OCR probe applies).
SUPPORTED_SUFFIXES = (".pdf", ".docx")

# Generous default: a single scanned/OCR document can legitimately take minutes;
# the point of the seam is bounding worst-case wall-clock per doc, not tuning speed.
DEFAULT_TIMEOUT_S = 600.0


class DocTimeoutError(Exception):
    """Raised when one document's pipeline run exceeds its timeout budget."""


def run_with_timeout(fn: Callable[[], object], timeout: float) -> object:
    """Run the no-arg `fn` on a worker thread and enforce `timeout` seconds.

    Injectable seam (`timeout_fn`) — tests substitute a fast fn + short timeout
    (or a synchronous passthrough) instead of waiting on a real hang."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise DocTimeoutError(f"exceeded {timeout:.0f}s") from None


def discover_docs(input_dir: Path) -> list[Path]:
    """Every PDF/DOCX directly under `input_dir`, sorted for deterministic order."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"input_dir {input_dir} is not a directory")
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def is_scanned_ir(ir: DocumentIR) -> bool:
    """Same test `demo.report.main()` uses: a doc whose first block didn't come
    from docling took the OCR/VLM path — it's already dirty, not simulated-dirty."""
    return bool(ir.blocks) and ir.blocks[0].source_engine != "docling"


def default_parse_fn(path: Path, settings) -> DocumentIR:
    """Route-parse for real customer documents — the same router mode
    `demo.report` uses (digital → docling, scanned → paddle/VLM). `.docx` has no
    scanned-page concept here (the text-coverage probe is PDF-only via pypdfium2),
    so it always goes straight to docling, which reads `.docx` natively."""
    from contract_rag.parse.docling_parser import parse_with_docling
    from contract_rag.parse.router import parse as router_parse

    path = Path(path)
    if path.suffix.lower() == ".docx":
        return parse_with_docling(path)
    return router_parse(path, settings)


def _bbox_seams(path: Path) -> tuple[list[tuple[float, float]] | None, Callable | None]:
    """Best-effort source-provenance seams for `build_report_data` — PDF-only
    (pypdfium2). Any failure (encrypted/corrupt PDF, non-PDF) degrades to
    `(None, None)`, which `build_report_data` already treats as "no highlighting"
    rather than an error."""
    if Path(path).suffix.lower() != ".pdf":
        return None, None
    try:
        from contract_rag.demo.render import page_sizes_pt, render_page_png

        return page_sizes_pt(path), functools.partial(render_page_png, path)
    except Exception:
        return None, None


class DocOutcome(BaseModel):
    """One successfully processed document — the unit the corpus summary
    aggregates over."""

    filename: str
    doc_id: str
    source_engine: str  # "digital" | "scanned"
    quality_score: float
    needs_review: bool
    stp: dict
    signature: dict | None = None  # populated only when source_engine == "scanned"
    facts_rows: list[dict]
    report_html: str
    report_file: str = ""  # filled in by the caller once the output name is chosen


class DocFailure(BaseModel):
    """One document that could not be processed — recorded, never fatal to the batch."""

    filename: str
    reason: str


def process_document(
    path: Path,
    extractor,
    parse_fn: Callable[[Path], DocumentIR],
    *,
    vertical=None,
    clm: str = "generic",
    seed: int = 0,
    timeout: float = DEFAULT_TIMEOUT_S,
    timeout_fn: Callable[[Callable[[], object], float], object] = run_with_timeout,
) -> DocOutcome:
    """Parse -> clean -> score -> extract -> verify -> (scanned) signature audit ->
    render, for one document. Raises on failure (corrupt/encrypted/zero-page/
    unparseable file, or a timeout) — callers are expected to catch per document so
    one bad file cannot sink the batch; that catching lives in `run_healthcheck`,
    not here, so this function stays a plain, testable unit."""
    path = Path(path)

    def _run() -> DocOutcome:
        ir = parse_fn(path)
        scanned = is_scanned_ir(ir)
        page_sizes, render_page = _bbox_seams(path)
        data = build_report_data(
            ir, extractor, seed=seed, title=path.stem.replace("_", " "),
            vertical=vertical,
            page_sizes=page_sizes, render_page=render_page,
            dirtify_fn=(lambda i: i) if scanned else None,
        )
        signature = detect_signature(ir).model_dump() if scanned else None
        s = stp_summary(data.fields)
        rows = rows_from_report(data, clm=clm)
        return DocOutcome(
            filename=path.name,
            doc_id=data.doc_id,
            source_engine="scanned" if scanned else "digital",
            quality_score=data.cleaned_quality.quality_score,
            needs_review=data.cleaned_quality.needs_review,
            stp=s,
            signature=signature,
            facts_rows=rows,
            report_html=render_html(data),
        )

    return timeout_fn(_run, timeout)


def _unique_report_name(path: Path, taken: set[str]) -> str:
    """`<stem>.html`, de-duplicated (e.g. `a.pdf` and `a.docx` in the same folder)
    by appending the original suffix, then a numeric counter as a last resort."""
    stem = path.stem or "doc"
    name = f"{stem}.html"
    if name not in taken:
        taken.add(name)
        return name
    name = f"{stem}{path.suffix.lower()}.html"
    if name not in taken:
        taken.add(name)
        return name
    i = 2
    while f"{stem}-{i}.html" in taken:
        i += 1
    name = f"{stem}-{i}.html"
    taken.add(name)
    return name


class HealthcheckSummary(BaseModel):
    """Corpus-level rollup — the machine-readable twin of `summary.html`."""

    n_docs: int
    n_ok: int
    n_failed: int
    engine_counts: dict[str, int]  # {"digital": N, "scanned": N}
    quality_mean: float | None
    quality_min: float | None
    needs_review_count: int
    stp: dict  # corpus-wide STP rollup, same shape as demo.report.stp_summary
    signature_counts: dict[str, int]  # {"signed": N, "unsigned": N}; scanned docs only
    field_verified_counts: dict[str, int] = Field(default_factory=dict)
    field_quarantined_counts: dict[str, int] = Field(default_factory=dict)
    docs: list[dict] = Field(default_factory=list)
    failures: list[DocFailure] = Field(default_factory=list)


def _combine_stp(rollups: list[dict]) -> dict:
    """Same shape/semantics as `demo.batch._combine_stp` (independently written —
    `batch.py` is composed, not imported from, per the no-modify-and-don't-reach-
    into-private-helpers rule)."""
    stp_fields = sum(r["stp_fields"] for r in rollups)
    total_fields = sum(r["total_fields"] for r in rollups)
    review_fields = list(dict.fromkeys(name for r in rollups for name in r["review_fields"]))
    return {
        "stp_fields": stp_fields,
        "total_fields": total_fields,
        "stp_rate": (stp_fields / total_fields) if total_fields else 0.0,
        "straight_through": all(r["straight_through"] for r in rollups) if rollups else True,
        "review_fields": review_fields,
    }


def build_summary(outcomes: list[DocOutcome], failures: list[DocFailure]) -> HealthcheckSummary:
    """Pure aggregation over already-processed outcomes + recorded failures."""
    n_ok = len(outcomes)
    engine_counts = {"digital": 0, "scanned": 0}
    signature_counts = {"signed": 0, "unsigned": 0}
    field_verified: dict[str, int] = {}
    field_quarantined: dict[str, int] = {}
    docs: list[dict] = []

    for o in outcomes:
        engine_counts[o.source_engine] = engine_counts.get(o.source_engine, 0) + 1
        if o.signature is not None:
            signature_counts["signed" if o.signature["signed"] else "unsigned"] += 1
        for row in o.facts_rows:
            bucket = field_verified if row["verified"] else field_quarantined
            bucket[row["field"]] = bucket.get(row["field"], 0) + 1
        docs.append({
            "filename": o.filename, "doc_id": o.doc_id, "source_engine": o.source_engine,
            "quality_score": o.quality_score, "needs_review": o.needs_review,
            "stp_rate": o.stp["stp_rate"], "straight_through": o.stp["straight_through"],
            "signature": o.signature, "report_file": o.report_file,
        })

    quality_scores = [o.quality_score for o in outcomes]
    return HealthcheckSummary(
        n_docs=n_ok + len(failures),
        n_ok=n_ok,
        n_failed=len(failures),
        engine_counts=engine_counts,
        quality_mean=(sum(quality_scores) / len(quality_scores)) if quality_scores else None,
        quality_min=min(quality_scores) if quality_scores else None,
        needs_review_count=sum(1 for o in outcomes if o.needs_review),
        stp=_combine_stp([o.stp for o in outcomes]),
        signature_counts=signature_counts,
        field_verified_counts=field_verified,
        field_quarantined_counts=field_quarantined,
        docs=docs,
        failures=failures,
    )


# ---------------------------------------------------------------- HTML summary

_SUMMARY_CSS = """
:root{--paper:oklch(0.985 0.008 85);--ink:oklch(0.26 0.02 220);--muted:oklch(0.48 0.02 220);
  --line:oklch(0.88 0.012 80);--clean:oklch(0.52 0.10 190);--dirty:oklch(0.56 0.14 55)}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Newsreader",Georgia,serif;padding:clamp(1.4rem,5vw,4rem) clamp(1.1rem,6vw,7rem)}
.wrap{max-width:980px;margin:0 auto}
.overline{font-size:.74rem;letter-spacing:.32em;text-transform:uppercase;color:var(--clean);font-weight:600}
h1{font-family:"Fraunces",Georgia,serif;font-size:clamp(2rem,4vw,3rem);font-weight:560;letter-spacing:-.01em;margin:.3rem 0 .1rem}
h2{font-family:"Fraunces",Georgia,serif;font-size:1.3rem;font-weight:560;margin:2rem 0 .4rem}
.meta{color:var(--muted);font-style:italic;margin-bottom:.4rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:1rem;margin:1.4rem 0}
.kpi{border:1px solid var(--line);border-radius:2px;padding:.9rem 1rem}
.kpi .n{font-family:"Fraunces",serif;font-size:1.8rem;font-weight:560}
.kpi .l{font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
table{width:100%;border-collapse:collapse;margin:.4rem 0 1rem}
th{text-align:left;font-family:Fraunces,serif;font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);padding:.5rem .6rem;border-bottom:1.5px solid var(--ink)}
td{padding:.5rem .6rem;border-bottom:1px solid var(--line);font-size:.9rem}
a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--clean)}
a:hover{color:var(--clean)}
.num{font-variant-numeric:tabular-nums;text-align:right}
.warn{color:var(--dirty);font-weight:600}
.ok{color:var(--clean);font-weight:600}
"""


def _kpi(n: str, label: str) -> str:
    return f'<div class="kpi"><div class="n">{n}</div><div class="l">{label}</div></div>'


def build_summary_html(summary: HealthcheckSummary) -> str:
    """Self-contained corpus-level index — links each `<name>.html` report and
    surfaces the KPIs the sales deliverable needs (§R1/§R4 of the PoC run-book)."""
    q_mean = f"{summary.quality_mean:.2f}" if summary.quality_mean is not None else "—"
    q_min = f"{summary.quality_min:.2f}" if summary.quality_min is not None else "—"
    kpis = "".join([
        _kpi(str(summary.n_docs), "Documents"),
        _kpi(f"{summary.n_ok} / {summary.n_failed}", "Processed / failed"),
        _kpi(f'{summary.engine_counts.get("digital", 0)} / {summary.engine_counts.get("scanned", 0)}',
             "Digital / scanned"),
        _kpi(q_mean, "Mean quality"),
        _kpi(q_min, "Min quality"),
        _kpi(str(summary.needs_review_count), "Needs review"),
        _kpi(f'{summary.stp["stp_rate"]:.0%}', "Field STP rate"),
        _kpi(f'{summary.signature_counts["signed"]} / {summary.signature_counts["unsigned"]}',
             "Signed / unsigned"),
    ])
    doc_rows = "".join(
        f'<tr><td><a href="{d["report_file"]}">{d["filename"]}</a></td>'
        f'<td>{d["source_engine"]}</td>'
        f'<td class="num">{d["quality_score"]:.2f}</td>'
        f'<td>{"yes" if d["needs_review"] else "no"}</td>'
        f'<td class="num">{d["stp_rate"]:.0%}</td>'
        f'<td>{"—" if d["signature"] is None else ("signed" if d["signature"]["signed"] else "unsigned")}</td>'
        f"</tr>"
        for d in summary.docs
    )
    field_names = sorted(set(summary.field_verified_counts) | set(summary.field_quarantined_counts))
    field_rows = "".join(
        f"<tr><td>{name}</td>"
        f'<td class="num ok">{summary.field_verified_counts.get(name, 0)}</td>'
        f'<td class="num warn">{summary.field_quarantined_counts.get(name, 0)}</td></tr>'
        for name in field_names
    )
    failure_section = ""
    if summary.failures:
        rows = "".join(
            f"<tr><td>{f.filename}</td><td>{f.reason}</td></tr>" for f in summary.failures
        )
        failure_section = (
            '<h2>Failed documents</h2>'
            '<p class="meta">Excluded from every aggregate above — one bad file never '
            'blocks the rest of the pack.</p>'
            f'<table><thead><tr><th>File</th><th>Reason</th></tr></thead><tbody>{rows}</tbody></table>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contract Health Check — {summary.n_docs} documents</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=Newsreader:opsz@6..72&display=swap" rel="stylesheet">
<style>{_SUMMARY_CSS}</style></head>
<body><div class="wrap">
<div class="overline">Contract-RAG · Health Check</div>
<h1>{summary.n_docs}-document health check</h1>
<div class="meta">{summary.n_ok} processed, {summary.n_failed} failed. Click a document for its full report.</div>
<div class="kpis">{kpis}</div>
<h2>Documents</h2>
<table><thead><tr><th>Document</th><th>Route</th><th class="num">Quality</th>
<th>Needs review</th><th class="num">STP</th><th>Signature</th></tr></thead>
<tbody>{doc_rows}</tbody></table>
<h2>Per-field verified / quarantined</h2>
<table><thead><tr><th>Field</th><th class="num">Verified</th><th class="num">Quarantined</th></tr></thead>
<tbody>{field_rows}</tbody></table>
{failure_section}
</div></body></html>"""


# ------------------------------------------------------------- top-level runner

def run_healthcheck(
    input_dir: Path,
    out_dir: Path,
    extractor,
    parse_fn: Callable[[Path], DocumentIR],
    *,
    vertical=None,
    clm: str = "generic",
    seed: int = 0,
    timeout: float = DEFAULT_TIMEOUT_S,
    timeout_fn: Callable[[Callable[[], object], float], object] = run_with_timeout,
    docs: list[Path] | None = None,
) -> HealthcheckSummary:
    """Process every PDF/DOCX in `input_dir`, writing the full pack to `out_dir`:
    one `<name>.html` report per document, a combined `facts.csv` + `facts.json`,
    and `summary.html` + `summary.json`. A document that fails (corrupt/encrypted/
    zero-page/unparseable, or exceeds `timeout`) is recorded in the summary and
    never stops the rest of the batch. `docs` overrides discovery (mainly for
    tests); an empty input directory (no discoverable docs, and no override) is a
    clean `ValueError`, not a silent empty pack."""
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    doc_paths = docs if docs is not None else discover_docs(input_dir)
    if not doc_paths:
        raise ValueError(
            f"no PDF/DOCX documents found in {input_dir} "
            f"(supported: {', '.join(SUPPORTED_SUFFIXES)})"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[DocOutcome] = []
    failures: list[DocFailure] = []
    all_rows: list[dict] = []
    taken_names: set[str] = set()

    for path in doc_paths:
        try:
            outcome = process_document(
                path, extractor, parse_fn, vertical=vertical, clm=clm, seed=seed,
                timeout=timeout, timeout_fn=timeout_fn,
            )
        except Exception as exc:  # noqa: BLE001 - per-doc robustness is the point
            failures.append(DocFailure(filename=path.name, reason=f"{type(exc).__name__}: {exc}"))
            continue
        outcome.report_file = _unique_report_name(path, taken_names)
        (out_dir / outcome.report_file).write_text(outcome.report_html)
        all_rows.extend(outcome.facts_rows)
        outcomes.append(outcome)

    summary = build_summary(outcomes, failures)

    (out_dir / "facts.csv").write_text(serialize(all_rows, "csv"))
    (out_dir / "facts.json").write_text(serialize(all_rows, "json", stp=summary.stp))
    (out_dir / "summary.html").write_text(build_summary_html(summary))
    (out_dir / "summary.json").write_text(summary.model_dump_json(indent=2))

    return summary
