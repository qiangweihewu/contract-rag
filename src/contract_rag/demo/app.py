"""Interactive Streamlit front-end for the before/after data-quality report.

Run:  uv run streamlit run src/contract_rag/demo/app.py
It reuses contract_rag.demo.report for all logic; this file is only UI glue.
"""
from __future__ import annotations

import tempfile
from html import escape
from pathlib import Path

import streamlit as st

from contract_rag.config import Settings, get_settings
from contract_rag.demo.ask import answer_question
from contract_rag.demo.report import (
    build_report_data,
    compare_fields,
    field_status,
    fields_by_tier,
    render_html,
    status_light,
    stp_summary,
)
from contract_rag.extract.rules import RuleExtractor

st.set_page_config(page_title="Contract-RAG · Data Quality", page_icon="◆", layout="wide")

_CSS = """
<style>
:root{--clean:#1f8a82;--clean-soft:#dcefed;--dirty:#b5651d;--dirty-soft:#f6e6d6}
h1,h2,h3{font-family:"Fraunces",Georgia,serif!important;letter-spacing:-.01em}
.stApp{background:#fdfcf8}
.diff{padding:.7rem .9rem;border-left:3px solid;border-radius:0 2px 2px 0;margin:.3rem 0;word-break:break-word}
.diff.raw{border-color:var(--dirty);background:var(--dirty-soft)}
.diff.fix{border-color:var(--clean);background:var(--clean-soft)}
.diff .lbl{font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:#6b6257;display:block;margin-bottom:.25rem}
table.facts{width:100%;border-collapse:collapse}
table.facts th{text-align:left;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#6b6257;border-bottom:1.5px solid #2a2a2a;padding:.45rem .6rem}
table.facts td{padding:.55rem .6rem;border-bottom:1px solid #e7e2d6;vertical-align:top;font-size:.92rem}
table.facts .fld{font-weight:600}
.pill{font-size:.68rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.14rem .5rem;border-radius:99px;white-space:nowrap}
.pill.verified{background:var(--clean-soft);color:var(--clean)}
.pill.review{background:var(--dirty-soft);color:var(--dirty)}
.pill.notfound{color:#9a9384}
table.facts .tierhead td{font-size:.68rem;letter-spacing:.2em;text-transform:uppercase;color:#6b6257;padding:.9rem .6rem .3rem;border-bottom:1px solid #2a2a2a}
.dot{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;margin-right:.4rem;vertical-align:baseline}
.dot.green{background:var(--clean)}
.dot.yellow{background:#d7a021}
.dot.red{background:#b3372b}
.dot.none{background:transparent;border:1px solid #e7e2d6}
.src{color:#9a9384;font-size:.82rem}
.mlabel{font-size:.8rem;color:#6b6257;margin:.45rem 0 .15rem}
.track{height:.5rem;background:#ece7da;border-radius:99px;overflow:hidden}
.fill{display:block;height:100%;border-radius:99px}
.fill.c{background:var(--clean)}.fill.d{background:var(--dirty)}
</style>
"""


@st.cache_data(show_spinner=False)
def _parse(pdf_bytes: bytes, _name: str):
    from contract_rag.parse.docling_parser import parse_with_docling

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        path = Path(tmp.name)
    return parse_with_docling(path)


def _extractor(choice: str):
    if choice.startswith("rule"):
        return RuleExtractor()
    from contract_rag.extract.extractor import get_extractor

    return get_extractor(
        Settings(extract_backend="openai", allow_external_llm=True, openai_model="gpt-5-mini")
    )


def _sample_pdfs(settings: Settings) -> list[Path]:
    found: list[Path] = []
    for root in (settings.data_dir, settings.cuad_dir):
        if root.exists():
            found += sorted(root.rglob("*.pdf"))
        if len(found) >= 40:
            break
    return found[:40]


def _facts_table(data) -> str:
    pill = {"verified": "verified", "review": "review", "not found": "notfound"}
    rows = []
    for tier, tier_fields in fields_by_tier(data.fields):
        rows.append(f"<tr class='tierhead'><td colspan='4'>{tier} risk</td></tr>")
        for f in tier_fields:
            s = field_status(f)
            light = f"<span class='dot {status_light(f)}'></span>"
            val = escape(f.cleaned_value) if f.cleaned_value else "<span class='src'>—</span>"
            src = escape(f.source_block_id or "—")
            label = f.field.replace("_", " ")
            rows.append(
                f"<tr><td class='fld'>{label}</td><td>{val}</td>"
                f"<td class='src'>{src}</td>"
                f"<td>{light}<span class='pill {pill[s]}'>{s}</span></td></tr>"
            )
    return ("<table class='facts'><thead><tr><th>Field</th><th>Extracted value</th>"
            "<th>Source block</th><th>Status</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _bar(label: str, value: float, tone: str) -> str:
    pct = max(0, min(100, round(value * 100)))
    return (f"<div class='mlabel'>{label} · {value:.2f}</div>"
            f"<div class='track'><span class='fill {tone}' style='width:{pct}%'></span></div>")


def _quality_panel(col, title: str, q, accent: str, tone: str):
    col.caption(title)
    col.markdown(f"<div style='font-family:Fraunces,serif;font-size:3rem;font-weight:600;"
                 f"line-height:1;color:{accent}'>{q.quality_score:.2f}</div>", unsafe_allow_html=True)
    col.markdown(("⚠️ **Needs review**" if q.needs_review else "✅ **Ready to use**"))
    bars = "".join([
        _bar("Clean text", 1 - q.garble_ratio, tone),
        _bar("Non-empty", 1 - q.empty_ratio, tone),
        _bar("Table integrity", q.table_integrity, tone),
        _bar("OCR confidence", q.mean_confidence, tone),
    ])
    col.markdown(bars, unsafe_allow_html=True)


def _compare_table(rows: list[dict]) -> str:
    pill = {"verified": "verified", "review": "review", "not found": "notfound"}

    def cell(val, status):
        v = escape(val) if val else "<span class='src'>—</span>"
        return f"{v}<br><span class='pill {pill[status]}'>{status}</span>"

    body = "".join(
        f"<tr><td class='fld'>{r['field'].replace('_', ' ')}</td>"
        f"<td>{cell(r['a_value'], r['a_status'])}</td>"
        f"<td>{cell(r['b_value'], r['b_status'])}</td></tr>"
        for r in rows
    )
    return ("<table class='facts'><thead><tr><th>Field</th><th>rule (offline)</th>"
            "<th>gpt-5-mini</th></tr></thead><tbody>" + body + "</tbody></table>")


def _clauses_html(results) -> str:
    cards = []
    for r in results:
        head = escape(r.heading) if r.heading else ""
        ctype = f"<span class='pill verified'>{escape(r.clause_type)}</span>" if r.clause_type else ""
        src = escape(", ".join(r.block_ids)) or "—"
        cards.append(
            f"<div class='diff fix'><span class='lbl'>#{r.rank} · {head} {ctype}</span>"
            f"{escape(r.text)}<div class='src'>source blocks: {src}</div></div>"
        )
    return "".join(cards)


def _ask_section(ir) -> None:
    st.divider()
    st.subheader("Ask a question over this contract")
    st.caption("Hybrid BM25 + semantic retrieval over the cleaned document — every "
               "answer cites the source block it came from.")
    with st.form("ask", clear_on_submit=False):
        q = st.text_input("Your question",
                          placeholder="e.g. how many days notice to terminate?")
        submitted = st.form_submit_button("Find relevant clauses")
    if not (submitted and q.strip()):
        return
    results = answer_question(ir, q, k=5)
    if not results:
        st.warning("No relevant clauses found.")
        return
    st.markdown(_clauses_html(results), unsafe_allow_html=True)


def _run_pipeline(uploaded, chosen, samples, backend, seed: int):
    """Parse + clean + extract one contract; returns a dict to stash in session_state
    (so the report and the ask box survive Streamlit reruns), or None on no/failed run."""
    if uploaded is not None:
        pdf_bytes, name = uploaded.getvalue(), uploaded.name
    elif chosen != "—":
        path = next(p for p in samples if p.name == chosen)
        pdf_bytes, name = path.read_bytes(), path.name
    else:
        st.warning("No contract selected.")
        return None

    compare = backend.startswith("compare")
    title = Path(name).stem.replace("_", " ")
    try:
        with st.spinner("Parsing (first run downloads docling models)…"):
            ir = _parse(pdf_bytes, name)
        with st.spinner("Dirtying, cleaning, extracting, verifying…"):
            data = build_report_data(ir, RuleExtractor() if compare else _extractor(backend),
                                     seed=seed, title=title)
            data_llm = (build_report_data(ir, _extractor("openai"), seed=seed, title=title)
                        if compare else None)
    except Exception as exc:  # noqa: BLE001 — surface any backend/parse error to the user
        st.error(f"Pipeline failed: {exc}")
        return None
    return {"ir": ir, "name": name, "data": data, "data_llm": data_llm, "compare": compare}


def _render_report(rs) -> None:
    data, data_llm, compare, name = rs["data"], rs["data_llm"], rs["compare"], rs["name"]
    d, c = data.dirty_quality, data.cleaned_quality   # quality is backend-independent
    st.markdown(f"### {escape(data.doc_id)}")
    st.markdown(f"A **quality {d.quality_score:.2f}** document flagged for review becomes a "
                f"**quality {c.quality_score:.2f}** document ready to use "
                f"(**+{c.quality_score - d.quality_score:.2f}**).")

    left, mid, right = st.columns([1, 0.08, 1])
    _quality_panel(left, "As ingested (dirty)", d, "#b5651d", "d")
    mid.markdown("<div style='font-size:2rem;text-align:center;color:#1f8a82;margin-top:2rem'>→</div>",
                 unsafe_allow_html=True)
    _quality_panel(right, "After cleaning", c, "#1f8a82", "c")

    if data.dirty_sample:
        st.subheader("What the noise looks like")
        st.markdown(f"<div class='diff raw'><span class='lbl'>As ingested</span>{escape(data.dirty_sample)}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='diff fix'><span class='lbl'>After cleaning</span>{escape(data.cleaned_sample)}</div>",
                    unsafe_allow_html=True)

    s = stp_summary(data.fields)
    st.metric("Straight-through", f"{s['stp_rate']:.0%}",
              help=("No fields need human review." if s["straight_through"]
                    else f"Needs review: {', '.join(n.replace('_', ' ') for n in s['review_fields'])}"))

    if compare:
        st.subheader("Extracted facts — rule vs gpt-5-mini")
        st.caption("Same cleaned document, two extractors. Each value is verified against its source block.")
        st.markdown(_compare_table(compare_fields(data, data_llm)), unsafe_allow_html=True)
    else:
        st.subheader("Extracted facts")
        st.caption("Every value is checked against its source block before it is trusted.")
        st.markdown(_facts_table(data), unsafe_allow_html=True)

    st.download_button("⬇ Download full HTML report (rule backend)", data=render_html(data),
                       file_name=f"{Path(name).stem}.report.html", mime="text/html")


def run() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    settings = get_settings()  # loads .env (OPENAI_API_KEY) for the openai backend

    st.markdown("<div style='color:#1f8a82;letter-spacing:.3em;font-size:.72rem;font-weight:600'>"
                "CONTRACT-RAG · DATA QUALITY REPORT</div>", unsafe_allow_html=True)
    st.title("Is your contract data usable?")
    st.caption("Upload a contract (or pick a sample). We simulate enterprise ingestion noise, "
               "clean it, and re-extract sourced, verified facts — showing the before/after.")

    with st.sidebar:
        st.subheader("Run a contract")
        backend = st.radio("Extraction backend",
                           ["rule (offline, free)", "openai (gpt-5-mini)",
                            "compare: rule vs gpt-5-mini"])
        seed = st.number_input("Noise seed", min_value=0, max_value=999, value=0)
        uploaded = st.file_uploader("Contract PDF", type=["pdf"])
        samples = _sample_pdfs(settings)
        names = ["—"] + [p.name for p in samples]
        chosen = st.selectbox("…or pick a sample", names)
        go = st.button("Run pipeline", type="primary", use_container_width=True)

    if go:
        result = _run_pipeline(uploaded, chosen, samples, backend, int(seed))
        if result is not None:
            st.session_state["run"] = result

    run_state = st.session_state.get("run")
    if run_state is None:
        st.info("Choose a PDF in the sidebar and press **Run pipeline**.")
        return

    # Render from session_state every rerun, so the report stays put while the ask
    # box (which triggers its own reruns) is used.
    _render_report(run_state)
    _ask_section(run_state["ir"])


run()
