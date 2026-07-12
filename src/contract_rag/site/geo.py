"""Pure GEO builders: schema.org JSON-LD + llms.txt. No I/O, fully unit-testable."""
from __future__ import annotations

import html as _html

from contract_rag.site.models import FAQItem, PageMeta


def json_ld(meta: PageMeta, *, author: str = "contract-rag") -> dict:
    """A schema.org @graph a generative engine can cite: TechArticle + FAQPage + HowTo.
    `publisher` is always emitted (Organization, per-vertical `author` name, the
    site root URL); `datePublished`/`dateModified` are added only when the page's
    front matter carries a `date` (both set to that same value — the site has no
    separate edit-tracking, so "modified" collapses to "published" honestly)."""
    article: dict = {
        "@type": "TechArticle",
        "headline": meta.title,
        "description": meta.description,
        "inLanguage": meta.lang,
        "url": meta.canonical,
        "author": {"@type": "Organization", "name": author},
        "publisher": {"@type": "Organization", "name": author, "url": "https://contractrag.com"},
        "keywords": ", ".join(meta.target_queries),
    }
    if meta.date:
        article["datePublished"] = meta.date
        article["dateModified"] = meta.date
    graph: list[dict] = [article]
    if meta.faq:
        graph.append({
            "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": f.q,
                 "acceptedAnswer": {"@type": "Answer", "text": f.a}}
                for f in meta.faq
            ],
        })
    if meta.howto:
        graph.append({
            "@type": "HowTo",
            "name": f"Reproduce: {meta.title}",
            "step": [{"@type": "HowToStep", "text": s.step} for s in meta.howto],
        })
    return {"@context": "https://schema.org", "@graph": graph}


def faq_json_ld(entries: list[FAQItem]) -> dict:
    """A standalone schema.org FAQPage node — additive counterpart to the FAQPage
    entry `json_ld` folds into an article's @graph, usable on its own (e.g. the
    landing page, which has no TechArticle)."""
    return {
        "@type": "FAQPage",
        "mainEntity": [
            {"@type": "Question", "name": f.q,
             "acceptedAnswer": {"@type": "Answer", "text": f.a}}
            for f in entries
        ],
    }


def organization_json_ld(*, name: str, url: str, github_url: str = "") -> dict:
    node: dict = {"@type": "Organization", "name": name, "url": url}
    if github_url:
        node["sameAs"] = [github_url]
    return node


def website_json_ld(*, name: str, url: str) -> dict:
    return {"@type": "WebSite", "name": name, "url": url}


def landing_json_ld(*, name: str, url: str, description: str, github_url: str = "",
                    faq: list[FAQItem] | None = None) -> dict:
    """Organization + WebSite (+ FAQPage, if entries given) @graph for the
    product landing page — additive, parallel to `json_ld` (articles)."""
    graph: list[dict] = [
        organization_json_ld(name=name, url=url, github_url=github_url),
        {**website_json_ld(name=name, url=url), "description": description},
    ]
    if faq:
        graph.append(faq_json_ld(faq))
    return {"@context": "https://schema.org", "@graph": graph}


def social_meta(*, title: str, description: str, url: str, lang: str,
                og_type: str = "article", site_name: str = "contract-rag",
                image: str | None = None) -> str:
    """Open Graph + Twitter Card `<meta>` tags for one page's `<head>` — pure
    string building, no I/O. `og:locale` derives from `lang` (`en`/`zh` ->
    `en_US`/`zh_CN`); `og:type` is `article` for content pages, `website` for
    the landing page. `image` (an absolute URL, 1200x630) upgrades the Twitter
    card to `summary_large_image` and adds the `og:image`/`twitter:image`
    pair; without it the tags are byte-identical to before the banner existed.
    Attribute values are HTML-escaped defensively (titles and descriptions are
    prose without literal quotes today, but this is cheap insurance against a
    future one that has them)."""
    locale = "zh_CN" if lang == "zh" else "en_US"
    esc_title = _html.escape(title, quote=True)
    esc_description = _html.escape(description, quote=True)
    esc_url = _html.escape(url, quote=True)
    esc_site_name = _html.escape(site_name, quote=True)
    lines = [
        f'<meta property="og:title" content="{esc_title}">',
        f'<meta property="og:description" content="{esc_description}">',
        f'<meta property="og:url" content="{esc_url}">',
        f'<meta property="og:site_name" content="{esc_site_name}">',
        f'<meta property="og:type" content="{og_type}">',
        f'<meta property="og:locale" content="{locale}">',
    ]
    if image is not None:
        esc_image = _html.escape(image, quote=True)
        lines += [
            f'<meta property="og:image" content="{esc_image}">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:image" content="{esc_image}">',
        ]
    else:
        lines.append('<meta name="twitter:card" content="summary">')
    return "\n".join(lines)


def llms_txt(pages: list[PageMeta], *, base_url: str, project: str = "contract-rag",
            include_landing: bool = False,
            github_url: str = "https://github.com/qiangweihewu/contract-rag") -> str:
    """Root llms.txt: a compact, link-first map for AI crawlers (llmstxt.org).
    `include_landing` (set by `build_site` only when `landing.{en,zh}.toml` are
    actually built) prepends the two product landing pages ahead of the article
    list; the GitHub repository link is always appended as its own `## Links`
    section, since it exists regardless of landing content."""
    base = base_url.rstrip("/")
    lines = [f"# {project}", "",
             "> Reproducible before/after benchmark: cleaning dirty contract PDFs for RAG.",
             "", "## Pages", ""]
    if include_landing:
        lines.append(f"- [{project} — home (en)]({base}/): Product landing page — "
                      "pillars, measured evidence, FAQ.")
        lines.append(f"- [{project} — 主页 (zh)]({base}/zh/): 产品首页——特性、实测证据、常见问题。")
    for p in pages:
        lines.append(f"- [{p.title}]({base}/{p.slug}.html): {p.description}")
    lines += ["", "## Links", "", f"- [GitHub repository]({github_url})"]
    return "\n".join(lines) + "\n"
