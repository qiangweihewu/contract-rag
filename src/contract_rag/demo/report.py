"""Customer-facing before/after data-quality report — the MVP's sales artifact.

Takes a parsed contract, simulates enterprise dirt (recoverable mojibake, broken
hyphenation, repeated headers, near-duplicates, whitespace), cleans it, and renders a
self-contained HTML report showing the quality recovery and the sourced, verified facts.
"""
from __future__ import annotations

import base64
import functools
from html import escape
from typing import Callable, Sequence

from pydantic import BaseModel, Field

from contract_rag.clean.pipeline import clean_ir
from contract_rag.clean.quality import QualityReport, compute_quality_score
from contract_rag.demo.highlight import HighlightRect, fact_highlights
from contract_rag.eval.dirtify import (
    dirtify,
    inject_hyphenation,
    inject_mojibake,
    inject_near_duplicates,
    inject_repeated_headers,
    inject_whitespace_noise,
)
from contract_rag.eval.metrics import RISK_TIERS, field_risk_map
from contract_rag.extract.verify import verify
from contract_rag.ir import DocumentIR


class FieldRow(BaseModel):
    field: str
    dirty_value: str
    cleaned_value: str
    source_block_id: str | None
    confidence: float
    verified: bool
    reasons: list[str]
    risk: str = "medium"
    highlight: HighlightRect | None = None  # page-fraction overlay; None = text-only source


class ReportData(BaseModel):
    doc_id: str
    dirty_quality: QualityReport
    cleaned_quality: QualityReport
    fields: list[FieldRow]
    dirty_sample: str
    cleaned_sample: str
    dirt_simulated: bool = True  # False when the input was already dirty (real scan)
    page_images: dict[int, str] = Field(default_factory=dict)  # page (1-based) → base64 PNG


def _harsh_dirtify(ir: DocumentIR, seed: int = 0) -> DocumentIR:
    """Aggressive but fully recoverable dirt, to make the before/after legible."""
    steps = [
        functools.partial(inject_mojibake, rate=0.95),
        functools.partial(inject_hyphenation, rate=0.6),
        inject_repeated_headers,
        functools.partial(inject_near_duplicates, rate=0.3),
        functools.partial(inject_whitespace_noise, rate=0.5),
    ]
    return dirtify(ir, seed=seed, steps=steps)


def build_report_data(ir: DocumentIR, extractor, seed: int = 0, title: str | None = None,
                      vertical=None,
                      page_sizes: Sequence[tuple[float, float]] | None = None,
                      render_page: Callable[[int], bytes] | None = None,
                      dirtify_fn: Callable[[DocumentIR], DocumentIR] | None = None) -> ReportData:
    """`page_sizes` (PDF points, from `demo.render.page_sizes_pt`) + `render_page`
    (0-based page → PNG bytes) enable source-provenance highlighting: facts whose
    cited block carries a usable bbox get an overlay rect, and each cited page is
    rendered once. Omit either seam (the default, e.g. batch) and the report is
    byte-identical to before. `dirtify_fn` replaces the simulated dirt — pass
    identity for a real scan, where the raw parse *is* the "before" side."""
    from contract_rag.verticals.registry import default_vertical

    v = vertical or default_vertical()
    risk = field_risk_map(v)
    dirty = dirtify_fn(ir) if dirtify_fn is not None else _harsh_dirtify(ir, seed)
    cleaned = clean_ir(dirty)
    dirty_facts = extractor.extract(dirty)
    cleaned_facts = extractor.extract(cleaned)
    checks = verify(cleaned_facts, cleaned, vertical=v).checks

    # highlights come from the *cleaned* IR — that's what cleaned_facts cite, and
    # cleaning preserves bbox on surviving blocks.
    rects: dict[str, HighlightRect | None] = {}
    page_images: dict[int, str] = {}
    if page_sizes is not None and render_page is not None:
        rects = {h.field: h.rect for h in fact_highlights(cleaned_facts, cleaned, page_sizes, v.field_names)}
        for r in rects.values():
            if r is not None and r.page not in page_images:
                page_images[r.page] = base64.b64encode(render_page(r.page - 1)).decode("ascii")

    fields = [
        FieldRow(
            field=name,
            dirty_value=getattr(dirty_facts, name).value,
            cleaned_value=getattr(cleaned_facts, name).value,
            source_block_id=getattr(cleaned_facts, name).source_block_id,
            confidence=getattr(cleaned_facts, name).confidence,
            verified=checks[name].passed,
            reasons=checks[name].reasons,
            risk=risk[name],
            highlight=rects.get(name),
        )
        for name in v.field_names
    ]

    dirty_by_id = {b.block_id: b.text for b in dirty.blocks}
    cleaned_by_id = {b.block_id: b.text for b in cleaned.blocks}
    sample_id = next(
        (bid for bid in dirty_by_id if bid in cleaned_by_id and "Â" in dirty_by_id[bid]),
        None,
    )
    dirty_sample = dirty_by_id.get(sample_id, "")[:260] if sample_id else ""
    cleaned_sample = cleaned_by_id.get(sample_id, "")[:260] if sample_id else ""

    return ReportData(
        doc_id=title or ir.doc_id,
        dirty_quality=compute_quality_score(dirty),
        cleaned_quality=compute_quality_score(cleaned),
        fields=fields,
        dirty_sample=dirty_sample,
        cleaned_sample=cleaned_sample,
        dirt_simulated=dirtify_fn is None,
        page_images=page_images,
    )


# ---------------------------------------------------------------- rendering

_CSS = """
:root{
  --paper:oklch(0.985 0.008 85); --panel:oklch(0.965 0.012 80);
  --ink:oklch(0.26 0.02 220); --muted:oklch(0.48 0.02 220);
  --line:oklch(0.88 0.012 80);
  --clean:oklch(0.52 0.10 190); --clean-soft:oklch(0.93 0.04 190);
  --dirty:oklch(0.56 0.14 55); --dirty-soft:oklch(0.93 0.05 65);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Newsreader",Georgia,"Times New Roman",serif;
  font-size:clamp(15px,1vw+12px,18px);line-height:1.55;
  padding:clamp(1.4rem,5vw,5rem) clamp(1.1rem,6vw,7rem)}
.wrap{max-width:980px;margin:0 auto}
h1,h2,h3,.score,.overline{font-family:"Fraunces","Newsreader",Georgia,serif}
.overline{font-size:.74rem;letter-spacing:.32em;text-transform:uppercase;
  color:var(--clean);font-weight:600;font-variation-settings:"opsz"20}
h1{font-size:clamp(2.1rem,4.6vw,3.6rem);line-height:1.02;margin:.35rem 0 .2rem;
  font-weight:560;letter-spacing:-.01em;overflow-wrap:anywhere}
.meta{color:var(--muted);font-size:.92rem;font-style:italic}
.rule{height:1px;background:var(--line);border:0;margin:clamp(1.6rem,3vw,2.6rem) 0}
.lede{font-size:clamp(1.05rem,1.4vw,1.3rem);max-width:62ch;color:var(--ink)}
.lede b{color:var(--dirty);font-weight:600}.lede .g{color:var(--clean);font-weight:600}

.diptych{display:grid;grid-template-columns:1fr auto 1fr;gap:clamp(.8rem,2vw,1.8rem);
  align-items:stretch;margin:1.2rem 0}
@media(max-width:680px){.diptych{grid-template-columns:1fr;}.arrow{transform:rotate(90deg);justify-self:center}}
.panel{border:1px solid var(--line);border-radius:2px;padding:clamp(1.1rem,2.4vw,1.8rem);
  background:var(--panel);position:relative}
.panel.before{background:linear-gradient(var(--dirty-soft),var(--panel) 60%)}
.panel.after{background:linear-gradient(var(--clean-soft),var(--panel) 60%)}
.panel .tag{font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;color:var(--muted)}
.score{font-size:clamp(3rem,7vw,4.8rem);line-height:.95;font-weight:550;letter-spacing:-.02em;margin:.1rem 0}
.before .score{color:var(--dirty)}.after .score{color:var(--clean)}
.badge{display:inline-block;font-family:"Newsreader",serif;font-size:.78rem;font-weight:600;
  letter-spacing:.12em;text-transform:uppercase;padding:.22rem .6rem;border-radius:2px}
.badge.review{background:var(--dirty);color:var(--paper)}
.badge.ready{background:var(--clean);color:var(--paper)}
.arrow{align-self:center;color:var(--clean);font-size:2rem;font-family:Fraunces,serif}
.metrics{margin-top:1rem;display:grid;gap:.45rem}
.metric{display:grid;grid-template-columns:8.5rem 1fr 2.4rem;align-items:center;gap:.6rem;font-size:.82rem}
.metric-label{color:var(--muted)}
.track{height:.42rem;background:oklch(0.9 0.01 80);border-radius:99px;overflow:hidden}
.fill{display:block;height:100%;border-radius:99px}
.fill.c{background:var(--clean)}.fill.d{background:var(--dirty)}
.metric-val{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}

h2{font-size:clamp(1.3rem,2.4vw,1.9rem);font-weight:540;margin:0 0 .2rem;letter-spacing:-.01em}
.sub{color:var(--muted);font-style:italic;margin:0 0 1rem}
.diff{display:grid;grid-template-columns:1fr;gap:.6rem;margin:.6rem 0}
.diff div{padding:.7rem .9rem;border-left:3px solid;border-radius:0 2px 2px 0;
  font-size:.92rem;word-break:break-word}
.diff .raw{border-color:var(--dirty);background:var(--dirty-soft)}
.diff .fix{border-color:var(--clean);background:var(--clean-soft)}
.diff .lbl{display:block;font-size:.68rem;letter-spacing:.2em;text-transform:uppercase;
  color:var(--muted);margin-bottom:.25rem}

table{width:100%;border-collapse:collapse;margin-top:.4rem}
th{text-align:left;font-family:Fraunces,serif;font-weight:560;font-size:.74rem;
  letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  padding:.5rem .7rem;border-bottom:1.5px solid var(--ink)}
td{padding:.62rem .7rem;border-bottom:1px solid var(--line);vertical-align:top;font-size:.93rem}
.fld{font-weight:600}
.was{color:var(--muted);text-decoration:line-through;text-decoration-color:var(--dirty)}
.now{color:var(--ink)}
.src{color:var(--muted);font-size:.82rem;font-variant-numeric:tabular-nums}
.pill{display:inline-block;font-size:.7rem;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;padding:.16rem .5rem;border-radius:99px;white-space:nowrap}
.pill.ok{background:var(--clean-soft);color:var(--clean)}
.pill.warn{background:var(--dirty-soft);color:var(--dirty)}
.pill.none{color:var(--muted)}
.tierhead td{font-family:Fraunces,serif;font-weight:560;font-size:.72rem;
  letter-spacing:.22em;text-transform:uppercase;color:var(--muted);
  padding:1rem .7rem .35rem;border-bottom:1px solid var(--ink)}
.dot{display:inline-block;width:.62rem;height:.62rem;border-radius:50%;
  margin-right:.45rem;vertical-align:baseline}
.dot.green{background:var(--clean)}
.dot.yellow{background:oklch(0.75 0.14 90)}
.dot.red{background:oklch(0.55 0.19 25)}
.dot.none{background:transparent;border:1px solid var(--line)}
details.page{margin:.9rem 0}
details.page summary{cursor:pointer;font-family:Fraunces,serif;font-weight:560;
  font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.pagewrap{position:relative;margin-top:.5rem;border:1px solid var(--line);
  border-radius:2px;overflow:hidden;background:#fff}
.pagewrap img{display:block;width:100%;height:auto}
.hl{position:absolute;border:2px solid var(--clean);border-radius:1px;
  background:oklch(0.52 0.10 190 / .16)}
.hl-lbl{position:absolute;top:-1.5em;left:-2px;font-family:"Newsreader",serif;
  font-size:.66rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--paper);background:var(--clean);padding:.06rem .38rem;border-radius:1px;
  white-space:nowrap}
footer{margin-top:2.4rem;color:var(--muted);font-size:.82rem;font-style:italic;max-width:70ch}
.stpline{font-size:.86rem;font-weight:600;letter-spacing:.01em;margin:.6rem 0 1rem;
  padding:.5rem .85rem;border-radius:2px;display:inline-block}
.stpline.ok{background:var(--clean-soft);color:var(--clean)}
.stpline.warn{background:var(--dirty-soft);color:var(--dirty);font-weight:500}
"""


def _metric(label: str, value: float, tone: str) -> str:
    pct = max(0, min(100, round(value * 100)))
    return (f'<div class="metric"><span class="metric-label">{label}</span>'
            f'<span class="track"><span class="fill {tone}" style="width:{pct}%"></span></span>'
            f'<span class="metric-val">{value:.2f}</span></div>')


def _panel(side: str, title: str, q: QualityReport, tone: str) -> str:
    badge = ('<span class="badge review">Needs review</span>' if q.needs_review
             else '<span class="badge ready">Ready to use</span>')
    metrics = "".join([
        _metric("Clean text", 1 - q.garble_ratio, tone),
        _metric("Non-empty", 1 - q.empty_ratio, tone),
        _metric("Table integrity", q.table_integrity, tone),
        _metric("OCR confidence", q.mean_confidence, tone),
    ])
    return (f'<div class="panel {side}"><div class="tag">{title}</div>'
            f'<div class="score">{q.quality_score:.2f}</div>{badge}'
            f'<div class="metrics">{metrics}</div></div>')


def field_status(f: FieldRow) -> str:
    """One of: 'verified' (auto-write), 'review' (HITL), 'not found' (no extraction)."""
    if not f.cleaned_value:
        return "not found"
    return "verified" if f.verified else "review"


def status_light(f: FieldRow) -> str:
    """Red/yellow/green from what exists at customer time (no gold), reusing verify()
    semantics: green = passed (attributed + confident); red = failed attribution, or
    nothing extracted for a high-risk field; yellow = quarantined for low confidence
    only (HITL). 'none' = empty on a lower-risk field — absence, not alarm."""
    if not f.cleaned_value:
        return "red" if f.risk == "high" else "none"
    if f.verified:
        return "green"
    if "unattributed" in f.reasons:
        return "red"
    return "yellow"  # low_confidence is the only other verify() reason


def stp_summary(fields: list[FieldRow]) -> dict:
    """Straight-Through Processing rollup — the KPI enterprise IDP buyers use.

    Derived from the exact same lights `status_light` already computes (no new
    status semantics): a field is straight-through iff its light is **green**
    (verified + confident). A document is straight-through iff it has **zero
    yellow** (low-confidence HITL) and **zero red** (unattributed, or empty on
    a high-risk field) fields — 'none' (empty on a lower-risk field, which
    `status_light` treats as absence rather than alarm) does not block STP or
    appear in `review_fields`.

    Returns a plain dict (JSON-serializable as-is): `stp_fields` (green count),
    `total_fields`, `stp_rate` (0.0 on a zero-field vertical, never divides by
    zero), `straight_through` (bool), `review_fields` (yellow+red field names,
    in field order)."""
    total = len(fields)
    stp_fields = 0
    review_fields: list[str] = []
    for f in fields:
        light = status_light(f)
        if light == "green":
            stp_fields += 1
        elif light in ("yellow", "red"):
            review_fields.append(f.field)
    return {
        "stp_fields": stp_fields,
        "total_fields": total,
        "stp_rate": (stp_fields / total) if total else 0.0,
        "straight_through": not review_fields,
        "review_fields": review_fields,
    }


def fields_by_tier(fields: list[FieldRow]) -> list[tuple[str, list[FieldRow]]]:
    """Group rows high → medium → low (unknown risk lands in medium); empty tiers omitted."""
    buckets: dict[str, list[FieldRow]] = {t: [] for t in RISK_TIERS}
    for f in fields:
        buckets[f.risk if f.risk in buckets else "medium"].append(f)
    return [(t, buckets[t]) for t in RISK_TIERS if buckets[t]]


def compare_fields(a: ReportData, b: ReportData) -> list[dict]:
    """Pair two backends' cleaned field rows for side-by-side display (a vs b)."""
    b_by = {f.field: f for f in b.fields}
    rows = []
    for fa in a.fields:
        fb = b_by.get(fa.field)
        rows.append({
            "field": fa.field,
            "a_value": fa.cleaned_value, "a_status": field_status(fa),
            "b_value": fb.cleaned_value if fb else "",
            "b_status": field_status(fb) if fb else "not found",
        })
    return rows


def _field_row(f: FieldRow) -> str:
    status = {
        "verified": '<span class="pill ok">verified</span>',
        "review": '<span class="pill warn">review</span>',
        "not found": '<span class="pill none">not found</span>',
    }[field_status(f)]
    light = f'<span class="dot {status_light(f)}"></span>'
    was = (f'<div class="was">{escape(f.dirty_value[:60])}</div>'
           if f.dirty_value and f.dirty_value != f.cleaned_value else "")
    now = f'<div class="now">{escape(f.cleaned_value)}</div>' if f.cleaned_value else '<span class="src">—</span>'
    src = escape(f.source_block_id or "—")
    if f.highlight is not None:
        src += f" · p.{f.highlight.page}"
    label = f.field.replace("_", " ")
    return (f"<tr><td class='fld'>{label}</td><td>{was}{now}</td>"
            f"<td class='src'>{src}</td><td>{light}{status}</td></tr>")


def _stp_banner(fields: list[FieldRow]) -> str:
    s = stp_summary(fields)
    if s["straight_through"]:
        return '<p class="stpline ok">Straight-through document — no human review required.</p>'
    names = ", ".join(escape(f.replace("_", " ")) for f in s["review_fields"])
    return (f'<p class="stpline warn">Straight-through: {s["stp_fields"]}/{s["total_fields"]} '
            f'fields ({s["stp_rate"]:.0%}) — needs review: {names}</p>')


def _provenance_section(data: ReportData) -> str:
    """One <details> per cited page: the rendered page image (embedded base64, so the
    report stays self-contained) with a CSS-positioned overlay div per fact whose
    cited block carries a bbox. Page images are rendered once and shared by every
    fact on that page. Empty when no fact has a highlight — the report is then
    exactly the pre-highlight report."""
    if not data.page_images:
        return ""
    by_page: dict[int, list[FieldRow]] = {}
    for f in data.fields:
        if f.highlight is not None and f.highlight.page in data.page_images:
            by_page.setdefault(f.highlight.page, []).append(f)
    figures = []
    for page in sorted(by_page):
        overlays = "".join(
            f'<div class="hl" style="left:{f.highlight.left:.2%};top:{f.highlight.top:.2%};'
            f'width:{f.highlight.width:.2%};height:{f.highlight.height:.2%}">'
            f'<span class="hl-lbl">{escape(f.field.replace("_", " "))}</span></div>'
            for f in by_page[page]
        )
        names = ", ".join(dict.fromkeys(f.field.replace("_", " ") for f in by_page[page]))
        figures.append(
            f'<details class="page" open><summary>Page {page} — {escape(names)}</summary>'
            f'<div class="pagewrap"><img alt="page {page}" '
            f'src="data:image/png;base64,{data.page_images[page]}">{overlays}</div></details>'
        )
    return ('<hr class="rule"><h2>Source provenance</h2>'
            '<p class="sub">Each extracted value highlighted where it lives on the original page '
            '(block-level granularity — the parser’s cited block, not the sub-span).</p>'
            + "".join(figures))


def render_html(data: ReportData) -> str:
    rows = "".join(
        f'<tr class="tierhead"><td colspan="4">{tier.capitalize()} risk</td></tr>'
        + "".join(_field_row(f) for f in tier_fields)
        for tier, tier_fields in fields_by_tier(data.fields)
    )
    diff = ""
    if data.dirty_sample:
        diff = (f'<div class="diff">'
                f'<div class="raw"><span class="lbl">As ingested</span>{escape(data.dirty_sample)}</div>'
                f'<div class="fix"><span class="lbl">After cleaning</span>{escape(data.cleaned_sample)}</div>'
                f'</div>')
    d, c = data.dirty_quality, data.cleaned_quality
    delta = c.quality_score - d.quality_score
    meta = ("simulated enterprise ingestion, cleaned and re-extracted" if data.dirt_simulated
            else "scanned document as ingested, cleaned and re-extracted")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Quality Report — {escape(data.doc_id)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400..600&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400&display=swap" rel="stylesheet">
<style>{_CSS}</style></head>
<body><div class="wrap">
<div class="overline">Contract-RAG · Data Quality Report</div>
<h1>{escape(data.doc_id)}</h1>
<div class="meta">Pipeline diagnostic — {meta}.</div>
<hr class="rule">
<p class="lede">The same contract, scored before and after the cleaning layer:
a <b>quality {d.quality_score:.2f}</b> document flagged for review becomes a
<span class="g">quality {c.quality_score:.2f}</span> document ready to use
(<span class="g">+{delta:.2f}</span>).</p>
<div class="diptych">
{_panel("before", "As ingested (dirty)", d, "d")}
<div class="arrow">→</div>
{_panel("after", "After cleaning", c, "c")}
</div>
<hr class="rule">
<h2>What the noise looks like</h2>
<p class="sub">Mojibake and whitespace damage, recovered losslessly.</p>
{diff}
<hr class="rule">
<h2>Extracted facts</h2>
<p class="sub">Grouped by field risk; every value is checked against its source block before it is trusted.</p>
{_stp_banner(data.fields)}
<table><thead><tr><th>Field</th><th>Extracted value</th><th>Source block</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table>
{_provenance_section(data)}
<footer>Quality score weights garble ratio, table integrity, empty ratio and OCR confidence;
<em>needs review</em> trips below 0.75. Extracted values carry a source block id and are
verified to appear in that block — unattributed or low-confidence values are quarantined for
human review rather than written automatically. Fields are grouped by risk tier; the light is
green when a value is verified and confident, yellow when it is quarantined for low confidence,
red when attribution fails or a high-risk field came back empty.</footer>
</div></body></html>"""


def main() -> None:
    import argparse
    from pathlib import Path

    from contract_rag.config import get_settings
    from contract_rag.demo.render import page_sizes_pt, render_page_png
    from contract_rag.extract.rules import RuleExtractor
    from contract_rag.parse.router import parse

    ap = argparse.ArgumentParser(
        prog="python -m contract_rag.demo.report",
        description="Before/after data-quality report (+ optional CLM facts export).",
    )
    ap.add_argument("pdf", help="contract PDF")
    ap.add_argument("out", nargs="?", help="output HTML path (default: <pdf>.report.html)")
    ap.add_argument("--export", choices=("csv", "json"),
                    help="also write the extracted facts as a sibling machine-readable file")
    ap.add_argument("--clm", choices=("salesforce", "ironclad", "generic"), default="generic",
                    help="CLM field-name mapping for the export (best-effort naming alignment)")
    args = ap.parse_args()
    settings = get_settings()
    path = Path(args.pdf)
    out = Path(args.out) if args.out else path.with_suffix(".report.html")

    extractor = RuleExtractor()
    if settings.extract_backend not in ("fake", "rule"):
        from contract_rag.extract.extractor import get_extractor

        extractor = get_extractor(settings)

    # route for real: digital → docling (as before), scanned → paddle/VLM. A scanned
    # doc is already dirty — skip the simulation and let the raw parse be "before".
    ir = parse(path, settings)
    scanned = bool(ir.blocks) and ir.blocks[0].source_engine != "docling"
    data = build_report_data(
        ir, extractor, title=path.stem.replace("_", " "),
        page_sizes=page_sizes_pt(path),
        render_page=functools.partial(render_page_png, path),
        dirtify_fn=(lambda i: i) if scanned else None,
    )
    out.write_text(render_html(data))
    print(f"wrote {out}")

    if args.export:
        from contract_rag.demo.export import rows_from_report, serialize

        facts_out = out.with_suffix(f".facts.{args.export}")
        facts_out.write_text(serialize(rows_from_report(data, clm=args.clm), args.export,
                                        stp=stp_summary(data.fields)))
        print(f"wrote {facts_out}")


if __name__ == "__main__":
    main()
