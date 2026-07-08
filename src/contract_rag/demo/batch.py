"""Batch-generate before/after data-quality reports across the golden set.

Writes one HTML report per contract plus an index.html linking them — a portfolio
view of the cleaning lift across many real contracts.

Run:  uv run python -m contract_rag.demo.batch        # reads golden_set/ + data/, writes reports/
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Callable

from contract_rag.demo.report import build_report_data, render_html, stp_summary
from contract_rag.eval.golden import GoldenDoc
from contract_rag.ir import DocumentIR

_INDEX_CSS = """
:root{--paper:oklch(0.985 0.008 85);--ink:oklch(0.26 0.02 220);--muted:oklch(0.48 0.02 220);
  --line:oklch(0.88 0.012 80);--clean:oklch(0.52 0.10 190);--dirty:oklch(0.56 0.14 55)}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Newsreader",Georgia,serif;padding:clamp(1.4rem,5vw,4rem) clamp(1.1rem,6vw,7rem)}
.wrap{max-width:900px;margin:0 auto}
.overline{font-size:.74rem;letter-spacing:.32em;text-transform:uppercase;color:var(--clean);font-weight:600}
h1{font-family:"Fraunces",Georgia,serif;font-size:clamp(2rem,4vw,3rem);font-weight:560;letter-spacing:-.01em;margin:.3rem 0 .1rem}
.meta{color:var(--muted);font-style:italic;margin-bottom:1.4rem}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-family:Fraunces,serif;font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);padding:.5rem .6rem;border-bottom:1.5px solid var(--ink)}
td{padding:.6rem .6rem;border-bottom:1px solid var(--line);font-size:.93rem}
a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--clean)}
a:hover{color:var(--clean)}
.num{font-variant-numeric:tabular-nums;text-align:right}
.d{color:var(--dirty);font-weight:600}.c{color:var(--clean);font-weight:600}
.delta{color:var(--clean)}
.stp{font-weight:600}
.stp.full{color:var(--clean)}
"""


def build_index_html(entries: list[dict]) -> str:
    """Each entry additionally carries `stp_rate` (green fields / total fields,
    see `demo.report.stp_summary`) and `straight_through` (bool); both default
    defensively (0.0 / False) so hand-built entries that predate the STP rollup
    still render."""
    n = len(entries)
    avg_d = sum(e["dirty"] for e in entries) / n if n else 0.0
    avg_c = sum(e["cleaned"] for e in entries) / n if n else 0.0
    avg_stp = sum(e.get("stp_rate", 0.0) for e in entries) / n if n else 0.0
    full_stp = sum(1 for e in entries if e.get("straight_through", False))
    full_stp_rate = (full_stp / n) if n else 0.0
    rows = "".join(
        f"<tr><td><a href='{escape(e['file'])}'>{escape(e['doc_id'])}</a></td>"
        f"<td class='num d'>{e['dirty']:.2f}</td>"
        f"<td class='num c'>{e['cleaned']:.2f}</td>"
        f"<td class='num delta'>+{e['cleaned'] - e['dirty']:.2f}</td>"
        f"<td class='num stp{' full' if e.get('straight_through', False) else ''}'>"
        f"{e.get('stp_rate', 0.0):.0%}</td></tr>"
        for e in entries
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Quality Reports — {n} contracts</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=Newsreader:opsz@6..72&display=swap" rel="stylesheet">
<style>{_INDEX_CSS}</style></head>
<body><div class="wrap">
<div class="overline">Contract-RAG · Data Quality</div>
<h1>Cleaning lift across {n} contracts</h1>
<div class="meta">Mean quality {avg_d:.2f} → {avg_c:.2f} (+{avg_c - avg_d:.2f}) after cleaning. Click a contract for its full report.</div>
<div class="meta">Mean field-STP rate {avg_stp:.0%} · {full_stp_rate:.0%} of docs fully straight-through (no human review required).</div>
<table><thead><tr><th>Contract</th><th class="num">Dirty</th><th class="num">Cleaned</th><th class="num">Δ</th><th class="num">STP</th></tr></thead>
<tbody>{rows}</tbody></table>
</div></body></html>"""


def _combine_stp(summaries: list[dict]) -> dict:
    """Aggregate the per-doc STP rollups into one summary for the combined
    facts export — same shape as `demo.report.stp_summary` (counts sum, rate
    is corpus-wide green/total, straight_through is true only if every doc is,
    review_fields is the de-duplicated union of field names needing review
    anywhere in the batch)."""
    stp_fields = sum(s["stp_fields"] for s in summaries)
    total_fields = sum(s["total_fields"] for s in summaries)
    review_fields = list(dict.fromkeys(
        name for s in summaries for name in s["review_fields"]
    ))
    return {
        "stp_fields": stp_fields,
        "total_fields": total_fields,
        "stp_rate": (stp_fields / total_fields) if total_fields else 0.0,
        "straight_through": all(s["straight_through"] for s in summaries) if summaries else True,
        "review_fields": review_fields,
    }


def run_batch(
    golden: list[GoldenDoc],
    data_dir: Path,
    out_dir: Path,
    extractor,
    parse_fn: Callable[[Path], DocumentIR],
    seed: int = 0,
    export: str | None = None,
    clm: str = "generic",
) -> list[dict]:
    """`export="csv"|"json"` additionally writes one facts file per doc plus a
    combined facts.{fmt} across all docs (CLM-aligned columns — see demo.export).
    Default (None) is byte-identical to the pre-export behavior. Every entry
    (and, for json, every facts file) also carries the STP rollup (see
    demo.report.stp_summary) — unconditional, independent of `export`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    all_rows: list[dict] = []
    stp_summaries: list[dict] = []
    for g in golden:
        pdf = Path(data_dir) / g.source_pdf
        if not pdf.exists():
            continue
        data = build_report_data(parse_fn(pdf), extractor, seed=seed,
                                 title=g.doc_id.replace("_", " "))
        s = stp_summary(data.fields)
        stp_summaries.append(s)
        fname = f"{g.doc_id}.report.html"
        (out_dir / fname).write_text(render_html(data))
        if export:
            from contract_rag.demo.export import rows_from_report, serialize

            rows = rows_from_report(data, clm=clm)
            (out_dir / f"{g.doc_id}.facts.{export}").write_text(serialize(rows, export, stp=s))
            all_rows.extend(rows)
        entries.append({
            "doc_id": g.doc_id, "file": fname,
            "dirty": data.dirty_quality.quality_score,
            "cleaned": data.cleaned_quality.quality_score,
            "stp_rate": s["stp_rate"], "straight_through": s["straight_through"],
        })
    if export:
        from contract_rag.demo.export import serialize

        (out_dir / f"facts.{export}").write_text(
            serialize(all_rows, export, stp=_combine_stp(stp_summaries)))
    (out_dir / "index.html").write_text(build_index_html(entries))
    return entries


def main() -> None:
    import os

    from contract_rag.config import get_settings
    from contract_rag.eval.golden import load_golden_set
    from contract_rag.extract.rules import RuleExtractor
    from contract_rag.parse.docling_parser import parse_with_docling

    settings = get_settings()
    golden = load_golden_set(settings.golden_set_dir)
    cap = int(os.environ.get("GOLDEN_SET_SIZE", "0"))
    if cap:
        golden = golden[:cap]
    out_dir = Path(os.environ.get("REPORT_OUT", "reports"))
    export = os.environ.get("EXPORT_FACTS") or None      # csv | json
    clm = os.environ.get("EXPORT_CLM", "generic")        # salesforce | ironclad | generic
    entries = run_batch(golden, settings.data_dir, out_dir, RuleExtractor(),
                        parse_with_docling, export=export, clm=clm)
    print(f"wrote {len(entries)} reports + index.html to {out_dir}/")
    if export:
        print(f"wrote per-doc facts.{export} files + combined facts.{export}")


if __name__ == "__main__":
    main()
