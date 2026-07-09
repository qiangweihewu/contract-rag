"""Tiny pure-Python static-site builder: markdown -> GEO-optimized HTML with the
benchmark's numbers substituted at build time (single source of truth). `markdown`
is imported lazily (extra `site`)."""
from __future__ import annotations

import json
import shutil
import tomllib
from datetime import date
from importlib import resources
from pathlib import Path
from string import Template
from urllib.parse import quote

from contract_rag.benchmark.core import BenchmarkResult
from contract_rag.site.geo import json_ld, landing_json_ld, llms_txt
from contract_rag.site.models import (
    LandingContent,
    LandingEvidenceRow,
    LandingNegative,
    LandingPillar,
    PageMeta,
    parse_front_matter,
)

# Search + AI crawlers get an explicit invitation (each its own User-agent block,
# on top of the general Allow-all) — the GEO surface robots.txt has to cover.
_AI_CRAWLERS = ["GPTBot", "ClaudeBot", "Claude-Web", "PerplexityBot", "Google-Extended"]

# Structured per-language landing copy, not flat measured-value tables — excluded
# from the generic content/*.toml static-token auto-merge (see build_site).
_LANDING_CONTENT_NAMES = {"landing.en.toml", "landing.zh.toml"}


def benchmark_tokens(result: BenchmarkResult) -> dict[str, str]:
    """`{{ token }}` values injected into article bodies from the benchmark."""
    return {
        "quality_dirty": f"{result.quality_dirty_mean:.2f}",
        "quality_clean": f"{result.quality_clean_mean:.2f}",
        "quality_lift": f"{result.quality_lift:+.2f}",
        "f1_dirty": f"{result.f1_dirty:.2f}",
        "f1_clean": f"{result.f1_clean:.2f}",
        "f1_lift": f"{result.f1_lift:+.2f}",
        "n_docs": str(result.n_docs),
    }


def load_static_tokens(path: Path | str) -> dict[str, str]:
    """Committed *point-in-time* measured values (flat TOML) injected exactly like
    benchmark tokens. For results that CANNOT be recomputed at build time (the
    dataset is download-gated and never committed, or the run needs a GPU): the
    values are committed as strings so the published formatting is exact, and the
    article states the measurement date + reproduction command instead of
    pretending they are live."""
    data = tomllib.loads(Path(path).read_text())
    return {key: str(value) for key, value in data.items()}


def _substitute_tokens(body: str, tokens: dict[str, str]) -> str:
    for k, v in tokens.items():
        body = body.replace("{{ " + k + " }}", v).replace("{{" + k + "}}", v)
    return body


def _substitute_recursive(obj, tokens: dict[str, str]):
    """Apply `_substitute_tokens` through nested TOML data (dict/list of
    dict/str) — the landing content has tokens inside list-of-table entries
    (pillars/evidence/negatives/faq), not just top-level strings."""
    if isinstance(obj, str):
        return _substitute_tokens(obj, tokens)
    if isinstance(obj, list):
        return [_substitute_recursive(v, tokens) for v in obj]
    if isinstance(obj, dict):
        return {k: _substitute_recursive(v, tokens) for k, v in obj.items()}
    return obj


def _template() -> Template:
    html = resources.files("contract_rag.site.templates").joinpath("base.html").read_text()
    return Template(html)


def _landing_template() -> Template:
    html = resources.files("contract_rag.site.templates").joinpath("landing.html").read_text()
    return Template(html)


def robots_txt(*, base_url: str) -> str:
    """robots.txt: allow-all plus an explicit invitation for each known AI crawler
    (search + generative-engine bots), and a pointer at sitemap.xml."""
    base = base_url.rstrip("/")
    lines = ["User-agent: *", "Allow: /", ""]
    for ua in _AI_CRAWLERS:
        lines += [f"User-agent: {ua}", "Allow: /", ""]
    lines.append(f"Sitemap: {base}/sitemap.xml")
    return "\n".join(lines) + "\n"


def sitemap_xml(pages: list[PageMeta], *, now: str) -> str:
    """sitemap.xml: one <url> per built page, with a `<lastmod>` from the `now`
    seam so the build stays deterministic in tests."""
    urls = "\n".join(
        f"  <url><loc>{p.canonical}</loc><lastmod>{now}</lastmod></url>"
        for p in pages
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>\n"
    )


def _hreflang(pages: list[PageMeta], this: PageMeta) -> str:
    base = this.canonical.rsplit("/", 1)[0]
    return "\n".join(
        f'<link rel="alternate" hreflang="{p.lang}" href="{base}/{p.slug}.html">'
        for p in pages
    )


def load_landing_content(path: Path | str, tokens: dict[str, str]) -> LandingContent:
    """Load one `content/landing.{lang}.toml` file with every `{{ token }}`
    (headline, pillars, evidence, negatives, FAQ) resolved before validation."""
    raw = tomllib.loads(Path(path).read_text())
    raw = _substitute_recursive(raw, tokens)
    return LandingContent(**raw)


def _render_pillars(pillars: list[LandingPillar]) -> str:
    return "\n".join(
        f'<div class="pillar"><h3>{p.title}</h3><p>{p.body}</p></div>' for p in pillars
    )


def _render_evidence(rows: list[LandingEvidenceRow], base: str) -> str:
    """Evidence-row links are made root-absolute for the same reason as the
    research nav (the zh landing sits at /zh/); an already-absolute link
    (http(s):// or leading /) is left untouched so external citations still work."""
    trs = []
    for r in rows:
        href = r.link if r.link.startswith(("http://", "https://", "/")) else f"{base}/{r.link}"
        cell = f'<a href="{href}">{r.label}</a>' if r.link else r.label
        trs.append(f"<tr><td>{cell}</td><td>{r.value}</td></tr>")
    return "\n".join(trs)


def _render_negatives(negatives: list[LandingNegative]) -> str:
    return "\n".join(f"<li>{n.text}</li>" for n in negatives)


def _render_faq(faq) -> str:
    return "\n".join(
        f"<details><summary>{f.q}</summary><p>{f.a}</p></details>" for f in faq
    )


def _render_research(pages: list[PageMeta], lang: str, base: str) -> str:
    """Research nav lists articles in the current page's language only, so the
    en landing page doesn't surface zh-titled links (and vice versa). Links are
    root-absolute (`{base}/slug.html`): the zh landing lives at `/zh/`, so a
    relative `slug.html` would resolve to `/zh/slug.html` and 404 — the articles
    are emitted at the site root."""
    return "\n".join(
        f'<li><a href="{base}/{p.slug}.html">{p.title}</a></li>'
        for p in pages if p.lang == lang
    )


def render_landing(tokens: dict[str, str], lang: str, pages: list[PageMeta], *,
                   content_dir: Path | str, base_url: str) -> str:
    """Render the bilingual product landing page (`index.html` / `zh/index.html`).
    Loads `content/landing.{lang}.toml`, substitutes every `{{ token }}`, and
    fills `templates/landing.html`. `pages` are the existing article pages,
    listed under the Research nav/footer section."""
    if lang not in ("en", "zh"):
        raise ValueError(f"unsupported landing lang: {lang!r}")
    content_dir = Path(content_dir)
    content = load_landing_content(content_dir / f"landing.{lang}.toml", tokens)

    base = base_url.rstrip("/")
    en_href, zh_href = f"{base}/", f"{base}/zh/"
    canonical = en_href if lang == "en" else zh_href
    lang_switch_href = zh_href if lang == "en" else en_href
    hreflang = (
        f'<link rel="alternate" hreflang="en" href="{en_href}">\n'
        f'<link rel="alternate" hreflang="zh" href="{zh_href}">'
    )
    jsonld = landing_json_ld(name="contract-rag", url=canonical, description=content.description,
                             github_url=content.github_url, faq=content.faq)
    cta_mailto = f"mailto:{content.cta_email}?subject={quote(content.cta_text)}"

    tmpl = _landing_template()
    return tmpl.substitute(
        lang=lang, title=content.title, description=content.description,
        canonical=canonical, hreflang=hreflang,
        jsonld=json.dumps(jsonld, ensure_ascii=False),
        headline=content.headline, subhead=content.subhead,
        cta_text=content.cta_text, cta_suffix=content.cta_suffix,
        cta_email=content.cta_email, cta_mailto=cta_mailto,
        proof_field_f1=content.proof_field_f1, proof_quality=content.proof_quality,
        proof_caption=content.proof_caption,
        pillars_heading=content.pillars_heading,
        evidence_heading=content.evidence_heading,
        negatives_heading=content.negatives_heading,
        faq_heading=content.faq_heading,
        pillars_html=_render_pillars(content.pillars),
        evidence_html=_render_evidence(content.evidence, base),
        negatives_html=_render_negatives(content.negatives),
        faq_html=_render_faq(content.faq),
        research_html=_render_research(pages, lang, base),
        research_label=content.research_label,
        footer_project_label=content.footer_project_label,
        footer_language_label=content.footer_language_label,
        github_url=content.github_url,
        llms_href=f"{base}/llms.txt",
        lang_switch_href=lang_switch_href,
        lang_switch_label=content.lang_switch_label,
        tagline=content.tagline,
    )


def build_site(content_dir, out_dir, *, base_url: str,
               benchmark: BenchmarkResult | None = None, charts_dir=None,
               static_tokens: dict[str, str] | None = None,
               now: str | None = None) -> list[Path]:
    import markdown  # lazy: extra `site`

    if now is None:
        now = date.today().isoformat()

    content_dir, out_dir = Path(content_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if static_tokens is None:
        static_tokens = {}
        for toml_path in sorted(content_dir.glob("*.toml")):
            if toml_path.name in _LANDING_CONTENT_NAMES:
                continue  # structured per-language copy, not a flat token table
            static_tokens.update(load_static_tokens(toml_path))
    tokens = dict(static_tokens)
    if benchmark is not None:
        tokens.update(benchmark_tokens(benchmark))
    tmpl = _template()

    parsed = [parse_front_matter(p.read_text()) for p in sorted(content_dir.glob("*.md"))]
    metas = [m for m, _ in parsed]
    written: list[Path] = []

    for meta in metas:
        meta.canonical = f"{base_url.rstrip('/')}/{meta.slug}.html"

    for meta, body in parsed:
        # front matter (description / FAQ answers) may carry tokens too, so the
        # committed data file stays the single source of truth for en+zh pages
        meta.description = _substitute_tokens(meta.description, tokens)
        for item in meta.faq:
            item.a = _substitute_tokens(item.a, tokens)
        body = _substitute_tokens(body, tokens)
        html_body = markdown.markdown(body, extensions=["extra", "toc", "sane_lists"])
        page_html = tmpl.substitute(
            lang=meta.lang, title=meta.title, description=meta.description,
            canonical=meta.canonical, hreflang=_hreflang(metas, meta),
            jsonld=json.dumps(json_ld(meta), ensure_ascii=False), body=html_body,
        )
        dest = out_dir / f"{meta.slug}.html"
        dest.write_text(page_html)
        written.append(dest)

    # copy charts (if produced) so the HTML can reference charts/*.png
    if charts_dir is not None and Path(charts_dir).exists():
        dst = out_dir / "charts"
        shutil.copytree(charts_dir, dst, dirs_exist_ok=True)

    llms = out_dir / "llms.txt"
    llms.write_text(llms_txt(metas, base_url=base_url))
    written.append(llms)

    robots = out_dir / "robots.txt"
    robots.write_text(robots_txt(base_url=base_url))
    written.append(robots)

    # The product landing page (/ + /zh/) replaces the auto-generated article
    # index when `content/landing.{en,zh}.toml` are present — additive: content
    # dirs without them (all existing tests, any future minimal fixture) keep
    # getting the old auto-index, byte-identical.
    en_landing = content_dir / "landing.en.toml"
    zh_landing = content_dir / "landing.zh.toml"
    sitemap_pages = list(metas)
    if en_landing.exists() and zh_landing.exists():
        base = base_url.rstrip("/")
        en_html = render_landing(tokens, "en", metas, content_dir=content_dir, base_url=base_url)
        zh_html = render_landing(tokens, "zh", metas, content_dir=content_dir, base_url=base_url)

        index = out_dir / "index.html"
        index.write_text(en_html)
        written.append(index)

        zh_dir = out_dir / "zh"
        zh_dir.mkdir(parents=True, exist_ok=True)
        zh_index = zh_dir / "index.html"
        zh_index.write_text(zh_html)
        written.append(zh_index)

        en_content = load_landing_content(en_landing, tokens)
        zh_content = load_landing_content(zh_landing, tokens)
        sitemap_pages += [
            PageMeta(title=en_content.title, description=en_content.description,
                     lang="en", slug="", canonical=f"{base}/"),
            PageMeta(title=zh_content.title, description=zh_content.description,
                     lang="zh", slug="zh", canonical=f"{base}/zh/"),
        ]
    else:
        index = out_dir / "index.html"
        links = "\n".join(f'<li><a href="{m.slug}.html">{m.title} ({m.lang})</a></li>' for m in metas)
        index.write_text(f"<!DOCTYPE html><html><head><meta charset=utf-8>"
                         f"<title>contract-rag</title></head><body><ul>{links}</ul></body></html>")
        written.append(index)

    sitemap = out_dir / "sitemap.xml"
    sitemap.write_text(sitemap_xml(sitemap_pages, now=now))
    written.append(sitemap)

    return written
