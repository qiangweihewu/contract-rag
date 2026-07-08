"""Pure GEO builders: schema.org JSON-LD + llms.txt. No I/O, fully unit-testable."""
from __future__ import annotations

from contract_rag.site.models import FAQItem, PageMeta


def json_ld(meta: PageMeta, *, author: str = "contract-rag") -> dict:
    """A schema.org @graph a generative engine can cite: TechArticle + FAQPage + HowTo."""
    graph: list[dict] = [
        {
            "@type": "TechArticle",
            "headline": meta.title,
            "description": meta.description,
            "inLanguage": meta.lang,
            "url": meta.canonical,
            "author": {"@type": "Organization", "name": author},
            "keywords": ", ".join(meta.target_queries),
        }
    ]
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


def llms_txt(pages: list[PageMeta], *, base_url: str, project: str = "contract-rag") -> str:
    """Root llms.txt: a compact, link-first map for AI crawlers (llmstxt.org)."""
    lines = [f"# {project}", "",
             "> Reproducible before/after benchmark: cleaning dirty contract PDFs for RAG.",
             "", "## Pages", ""]
    for p in pages:
        lines.append(f"- [{p.title}]({base_url}/{p.slug}.html): {p.description}")
    return "\n".join(lines) + "\n"
