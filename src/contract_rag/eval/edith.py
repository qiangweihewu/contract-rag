"""EDiTh mixed-document routing measurement — does the doc-level parse router lose
pages on documents that are part-digital, part-scanned?

Motivation: the router's `probe.probe_document()` collapses a PDF to ONE
text-coverage number, and `route()` picks ONE parser for the whole file. That is
correct for the all-or-nothing documents it has been exercised on (CUAD ≈ coverage
1.0 → docling; Tobacco800/FinCriticalED = coverage 0.0 → paddle). It has never been
tested on the most common real enterprise shape: a **mixed** PDF — a digital body
with a scanned/faxed signed annex. There the single coverage number is an average
that describes no actual page, so the whole document is sent to one engine and the
other half's pages are parsed by the wrong one:

  * a scanned page routed to **docling** (OCR off) yields ~no text — content LOST;
  * a digital page routed to **paddleocr/vlm** is re-OCR'd — content DEGRADED.

Dataset: **EDiTh / Véracier Industries** (HuggingFace `lightonai/veracier-industries`,
Apache-2.0, ungated). 1 004 synthetic enterprise PDFs; its README defines the
**mixed** format precisely as "first half searchable + second half scanned (signed
annexes, attachments)" — 85–87 documents flagged `format == "mixed"` in
`MASTER_INDEX.csv`. This harness focuses on that subset (the scan-artifact-only
`scanned` subset is the fallback, but mixed is cleanly labelled so we use it).

Two measurements:

1. **Structural (fast, probe-only, all mixed docs).** Per-page `probe.probe_pages()`
   vs the document-level `route()` decision: how many pages does the single-engine
   decision send to the wrong parser? Split into content-loss (scanned→docling) vs
   degradation (digital→OCR). This is the core result and needs no OCR.

2. **Parse confirmation (optional, EDITH_PARSE_SIZE>0, IR-cached).** Actually parse a
   handful of mixed docs the way production does (doc-level route) AND with the new
   per-page router, and count characters recovered per page — grounding the
   structural prediction in real parser output (a scanned page under doc-level docling
   really does come back near-empty; per-page routing recovers it via paddle).

The 36 EDiTh QA scenarios are **not** run here: their ground truth is document-level
retrieval (which of 1 004 files answers a question), not the chunk-level supporting
spans our `eval/retrieval.py` Context Recall needs, and scoring them would require
OCR-ingesting the entire 1.7 GB corpus. That mapping isn't clean, so it is skipped by
design (see module note), not forced.

Env: EDITH_DIR (a local snapshot; default auto-download to
~/.cache/contract-rag/edith), EDITH_SET_SIZE (cap on mixed docs, default all),
EDITH_PARSE_SIZE (docs to also parse for the confirmation, default 0),
EDITH_CACHE (pdf/IR cache), EDITH_OUT (optional JSON dump).

Run: uv run python -m contract_rag.eval.edith
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from contract_rag.config import Settings
from contract_rag.ir import DocumentIR
from contract_rag.parse.probe import PageProfile, probe_pages, profile_from_pages
from contract_rag.parse.router import contiguous_segments, page_route, route

HF_REPO = "lightonai/veracier-industries"


# ------------------------------------------------------------------ dataset index

class DocEntry(BaseModel):
    doc_id: str
    entity: str
    filename: str  # path within the entity folder (may contain subdirs)
    language: str
    format: str
    pages: int

    def rel_path(self) -> str:
        return f"by_entity/{self.entity}/{self.filename}"


def load_index(edith_dir: Path) -> list[DocEntry]:
    """Unique PHYSICAL documents from MASTER_INDEX.csv. A file recurs across use cases
    under a fresh `doc_id` each time (doc_id = hash of (question_id, filename)), so we
    key on the physical `by_entity/{entity}/{filename}` path — dedup by doc_id would
    count the same PDF many times. Deterministic by first-seen row order."""
    edith_dir = Path(edith_dir)
    seen: dict[str, DocEntry] = {}
    with (edith_dir / "MASTER_INDEX.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = f"{row['entity']}/{row['filename']}"
            if key in seen:
                continue
            try:
                pages = int(row.get("pages") or 0)
            except ValueError:
                pages = 0
            seen[key] = DocEntry(
                doc_id=row["doc_id"],
                entity=row["entity"],
                filename=row["filename"],
                language=row.get("language", ""),
                format=row.get("format", ""),
                pages=pages,
            )
    return list(seen.values())


def select_mixed(entries: list[DocEntry], cap: int | None = None) -> list[DocEntry]:
    """Deterministic (sorted by rel path) mixed-format subset, capped. Falls back to
    the `scanned` subset only if no mixed docs exist (documented in the report)."""
    mixed = sorted((e for e in entries if e.format == "mixed"), key=lambda e: e.rel_path())
    chosen = mixed or sorted(
        (e for e in entries if e.format == "scanned"), key=lambda e: e.rel_path()
    )
    return chosen[:cap] if cap else chosen


# ------------------------------------------------------------- routing analysis

class DocRouting(BaseModel):
    name: str
    n_pages: int
    n_digital_pages: int
    n_scanned_pages: int
    doc_coverage: float
    doc_route: str            # the single engine the current router picks
    page_routes: list[str]    # the engine each page actually warrants
    n_misrouted_pages: int    # pages whose warranted engine != doc_route
    n_content_loss_pages: int  # scanned pages sent to docling (OCR off → text lost)
    n_degraded_pages: int      # digital pages sent to an OCR engine (re-OCR'd, not lost)
    n_segments: int            # contiguous same-engine runs (1 == pure doc)


def analyze_routing(
    name: str, pages: list[PageProfile], settings: Settings
) -> DocRouting:
    """Pure: compare the document-level route decision against a per-page one.

    `doc_route` is exactly what `router.parse()` does today (probe → route). A page
    is *misrouted* when the engine it warrants differs from `doc_route`. Content is
    *lost* when a scanned page (no text layer) is handled by docling — whose OCR is
    off — and merely *degraded* when a digital page is sent to an OCR engine."""
    doc_profile = profile_from_pages(pages)
    doc_route = route(doc_profile, settings)
    page_routes = [page_route(pp, settings) for pp in pages]

    misrouted = sum(1 for r in page_routes if r != doc_route)
    content_loss = (
        sum(1 for pp in pages if not pp.has_text) if doc_route == "docling" else 0
    )
    degraded = (
        sum(1 for pp in pages if pp.has_text) if doc_route != "docling" else 0
    )
    return DocRouting(
        name=name,
        n_pages=len(pages),
        n_digital_pages=sum(1 for pp in pages if pp.has_text),
        n_scanned_pages=sum(1 for pp in pages if not pp.has_text),
        doc_coverage=round(doc_profile.text_coverage, 3),
        doc_route=doc_route,
        page_routes=page_routes,
        n_misrouted_pages=misrouted,
        n_content_loss_pages=content_loss,
        n_degraded_pages=degraded,
        n_segments=len(contiguous_segments(page_routes)),
    )


# ---------------------------------------------------------------- aggregation

class Summary(BaseModel):
    n_docs: int
    total_pages: int
    subset: str                       # "mixed" or "scanned" (fallback)
    n_docs_both_formats: int          # docs with >=1 digital AND >=1 scanned page
    n_docs_with_misroute: int
    frac_docs_with_misroute: float
    total_misrouted_pages: int
    total_content_loss_pages: int
    total_degraded_pages: int
    n_docs_content_loss: int          # docs losing >=1 page entirely
    n_docs_degraded: int
    doc_route_dist: dict[str, int] = Field(default_factory=dict)


def summarize(routings: list[DocRouting], subset: str) -> Summary:
    if not routings:
        raise ValueError("no documents analyzed")
    route_dist: dict[str, int] = {}
    for r in routings:
        route_dist[r.doc_route] = route_dist.get(r.doc_route, 0) + 1
    with_misroute = [r for r in routings if r.n_misrouted_pages > 0]
    return Summary(
        n_docs=len(routings),
        total_pages=sum(r.n_pages for r in routings),
        subset=subset,
        n_docs_both_formats=sum(
            1 for r in routings if r.n_digital_pages and r.n_scanned_pages
        ),
        n_docs_with_misroute=len(with_misroute),
        frac_docs_with_misroute=round(len(with_misroute) / len(routings), 3),
        total_misrouted_pages=sum(r.n_misrouted_pages for r in routings),
        total_content_loss_pages=sum(r.n_content_loss_pages for r in routings),
        total_degraded_pages=sum(r.n_degraded_pages for r in routings),
        n_docs_content_loss=sum(1 for r in routings if r.n_content_loss_pages),
        n_docs_degraded=sum(1 for r in routings if r.n_degraded_pages),
        doc_route_dist=route_dist,
    )


def format_report(routings: list[DocRouting], summary: Summary) -> str:
    lines = [
        "=== EDiTh mixed-document routing measurement ===",
        f"{'doc':<44} {'pg':>3} {'dig':>3} {'scn':>3} {'cov':>5}"
        f" {'doc-route':<10} {'misrt':>5} {'lost':>4}",
    ]
    for r in routings:
        lines.append(
            f"{r.name[:44]:<44} {r.n_pages:>3} {r.n_digital_pages:>3}"
            f" {r.n_scanned_pages:>3} {r.doc_coverage:>5.2f} {r.doc_route:<10}"
            f" {r.n_misrouted_pages:>5} {r.n_content_loss_pages:>4}"
        )
    s = summary
    lines += [
        "",
        f"subset={s.subset} docs={s.n_docs} pages={s.total_pages}"
        f" doc-route-dist={s.doc_route_dist}",
        f"docs with both digital+scanned pages: {s.n_docs_both_formats}",
        f"docs with >=1 MISROUTED page: {s.n_docs_with_misroute}"
        f" ({s.frac_docs_with_misroute:.1%})",
        f"total misrouted pages: {s.total_misrouted_pages}"
        f"  (content-LOST scanned→docling: {s.total_content_loss_pages}"
        f" in {s.n_docs_content_loss} docs;"
        f" degraded digital→OCR: {s.total_degraded_pages} in {s.n_docs_degraded} docs)",
    ]
    return "\n".join(lines)


# ------------------------------------------------------ parse confirmation (opt-in)

def chars_per_page(ir: DocumentIR, n_pages: int) -> list[int]:
    """Total block-text characters attributed to each 1-based page of the IR."""
    counts = [0] * n_pages
    for b in ir.blocks:
        if b.bbox is not None and 1 <= b.bbox.page <= n_pages:
            counts[b.bbox.page - 1] += len(b.text)
    return counts


class ParseCheck(BaseModel):
    name: str
    n_pages: int
    doc_route: str
    scanned_pages: list[int]           # 1-based scanned page numbers
    doclevel_chars: list[int]          # chars/page, current single-route parse
    perpage_chars: list[int]           # chars/page, per-page router
    scanned_chars_doclevel: int        # chars recovered on scanned pages, today
    scanned_chars_perpage: int         # chars recovered on scanned pages, per-page


def confirm_doc(
    pdf: Path,
    pages: list[PageProfile],
    settings: Settings,
    doclevel_parse: Callable[[Path], DocumentIR],
    perpage_parse: Callable[[Path], DocumentIR],
) -> ParseCheck:
    """Parse `pdf` both ways and count characters recovered per page. Confirms that
    the pages the structural analysis flags as content-lost really do come back
    near-empty under today's route, and are recovered by the per-page router."""
    n = len(pages)
    scanned = [pp.page for pp in pages if not pp.has_text]
    dl = chars_per_page(doclevel_parse(pdf), n)
    pp = chars_per_page(perpage_parse(pdf), n)
    doc_route = route(profile_from_pages(pages), settings)
    return ParseCheck(
        name=pdf.stem,
        n_pages=n,
        doc_route=doc_route,
        scanned_pages=scanned,
        doclevel_chars=dl,
        perpage_chars=pp,
        scanned_chars_doclevel=sum(dl[p - 1] for p in scanned),
        scanned_chars_perpage=sum(pp[p - 1] for p in scanned),
    )


# ---------------------------------------------------------------- dataset IO

def ensure_dataset(edith_dir: Path, needed: list[DocEntry] | None = None) -> Path:
    """Download the metadata (small) + extract the needed PDFs from by_entity.tar.gz.
    Only the members we actually analyze are extracted, so a mixed-only run costs a
    few MB on disk rather than 1.7 GB unpacked."""
    edith_dir = Path(edith_dir)
    edith_dir.mkdir(parents=True, exist_ok=True)
    if not (edith_dir / "MASTER_INDEX.csv").exists():
        from huggingface_hub import hf_hub_download

        for f in ("MASTER_INDEX.csv", "README.md", "ANSWER_KEY.json"):
            hf_hub_download(HF_REPO, f, repo_type="dataset", local_dir=str(edith_dir))
    if needed:
        _ensure_pdfs(edith_dir, needed)
    return edith_dir


def _ensure_pdfs(edith_dir: Path, needed: list[DocEntry]) -> None:
    """Extract just the `needed` PDFs from by_entity.tar.gz if they're not on disk."""
    missing = [e for e in needed if not (edith_dir / e.rel_path()).exists()]
    if not missing:
        return
    import tarfile

    tar_path = edith_dir / "by_entity.tar.gz"
    if not tar_path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            HF_REPO, "by_entity.tar.gz", repo_type="dataset", local_dir=str(edith_dir)
        )
    wanted = {e.rel_path() for e in missing}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if name in wanted:
                member.name = name
                tar.extract(member, path=edith_dir)


# ----------------------------------------------------------------- impure runner

def run_edith(
    edith_dir: Path,
    settings: Settings,
    cache_dir: Path,
    cap: int | None = None,
    parse_cap: int = 0,
    probe_fn: Callable[[Path], list[PageProfile]] | None = None,
) -> tuple[list[DocRouting], Summary, list[ParseCheck]]:
    """Structural routing analysis over the mixed subset (+ optional parse
    confirmation on the first `parse_cap` docs). `probe_fn` is the test seam."""
    probe_fn = probe_fn or probe_pages
    cache_dir = Path(cache_dir)

    entries = load_index(edith_dir)
    chosen = select_mixed(entries, cap)
    if not chosen:
        raise SystemExit(f"no mixed/scanned docs in {edith_dir}/MASTER_INDEX.csv")
    subset = "mixed" if any(e.format == "mixed" for e in chosen) else "scanned"
    _ensure_pdfs(edith_dir, chosen)

    routings: list[DocRouting] = []
    profiles_by_doc: dict[str, list[PageProfile]] = {}
    for e in chosen:
        pdf = Path(edith_dir) / e.rel_path()
        if not pdf.exists():
            continue
        pages = probe_fn(pdf)
        profiles_by_doc[e.rel_path()] = pages
        # filenames collide across entities (16 distinct files named doc_005.pdf), so
        # disambiguate the display name with the entity
        routings.append(analyze_routing(f"{e.entity}/{Path(e.filename).stem}", pages, settings))

    summary = summarize(routings, subset)

    checks: list[ParseCheck] = []
    if parse_cap > 0:
        checks = _run_parse_checks(edith_dir, chosen, profiles_by_doc, settings, cache_dir, parse_cap)
    return routings, summary, checks


def _run_parse_checks(
    edith_dir: Path,
    chosen: list[DocEntry],
    profiles_by_doc: dict[str, list[PageProfile]],
    settings: Settings,
    cache_dir: Path,
    parse_cap: int,
) -> list[ParseCheck]:
    from contract_rag.eval.ir_cache import ir_cache
    from contract_rag.parse.router import parse as router_parse

    doclevel = ir_cache(
        cache_dir / "ir" / "doclevel", lambda p: router_parse(p, settings)
    )
    perpage = ir_cache(
        cache_dir / "ir" / "perpage", lambda p: router_parse(p, settings, per_page=True)
    )
    checks: list[ParseCheck] = []
    for e in chosen[:parse_cap]:
        pdf = Path(edith_dir) / e.rel_path()
        pages = profiles_by_doc.get(e.rel_path())
        if not pdf.exists() or not pages:
            continue
        checks.append(confirm_doc(pdf, pages, settings, doclevel, perpage))
    return checks


def format_parse_checks(checks: list[ParseCheck]) -> str:
    lines = [
        "",
        "=== parse confirmation (chars recovered on scanned pages) ===",
        f"{'doc':<40} {'route':<10} {'scanned pg':>11} {'doc-level':>10} {'per-page':>9}",
    ]
    tot_dl = tot_pp = 0
    for c in checks:
        tot_dl += c.scanned_chars_doclevel
        tot_pp += c.scanned_chars_perpage
        lines.append(
            f"{c.name[:40]:<40} {c.doc_route:<10} {str(c.scanned_pages):>11}"
            f" {c.scanned_chars_doclevel:>10} {c.scanned_chars_perpage:>9}"
        )
    lines.append(
        f"TOTAL chars on scanned pages: doc-level={tot_dl} per-page={tot_pp}"
        + (f" (+{tot_pp - tot_dl} recovered)" if tot_pp > tot_dl else "")
    )
    return "\n".join(lines)


def main() -> None:
    import json
    import os

    from contract_rag.config import get_settings

    edith_dir = Path(
        os.environ.get("EDITH_DIR", str(Path.home() / ".cache" / "contract-rag" / "edith"))
    )
    cache = Path(
        os.environ.get("EDITH_CACHE", str(Path.home() / ".cache" / "contract-rag" / "edith-run"))
    )
    ensure_dataset(edith_dir)
    cap = int(os.environ["EDITH_SET_SIZE"]) if os.environ.get("EDITH_SET_SIZE") else None
    parse_cap = int(os.environ.get("EDITH_PARSE_SIZE", "0"))

    routings, summary, checks = run_edith(
        edith_dir, get_settings(), cache, cap=cap, parse_cap=parse_cap
    )
    print(format_report(routings, summary))
    if checks:
        print(format_parse_checks(checks))
    out = os.environ.get("EDITH_OUT")
    if out:
        payload = {
            "summary": summary.model_dump(),
            "routings": [r.model_dump() for r in routings],
            "parse_checks": [c.model_dump() for c in checks],
        }
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
