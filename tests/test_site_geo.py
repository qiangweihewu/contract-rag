from __future__ import annotations

import json

from contract_rag.site.geo import json_ld, llms_txt, social_meta
from contract_rag.site.models import PageMeta, parse_front_matter

_FM = '''+++
title = "Why your contract RAG returns garbage"
description = "A reproducible before/after benchmark."
lang = "en"
slug = "benchmark"
canonical = "https://example.github.io/contract-rag/benchmark.html"
target_queries = ["why is my RAG returning garbage", "how to clean PDFs for RAG"]
[[faq]]
q = "Why does dirty text break RAG?"
a = "Mojibake and repeated headers poison retrieval and extraction."
[[howto]]
step = "Run python -m contract_rag.benchmark"
+++
# Body starts here

Real content.'''


def test_parse_front_matter():
    meta, body = parse_front_matter(_FM)
    assert isinstance(meta, PageMeta)
    assert meta.title.startswith("Why your contract RAG")
    assert meta.lang == "en"
    assert meta.faq[0].q.startswith("Why does dirty")
    assert body.lstrip().startswith("# Body starts here")


def test_json_ld_has_required_types():
    meta, _ = parse_front_matter(_FM)
    graph = json_ld(meta)
    # must be valid json and contain the citable schema types
    types = {node["@type"] for node in graph["@graph"]}
    assert {"TechArticle", "FAQPage", "HowTo"} <= types
    assert json.loads(json.dumps(graph))  # round-trips


def test_llms_txt_lists_pages():
    meta, _ = parse_front_matter(_FM)
    txt = llms_txt([meta], base_url="https://example.github.io/contract-rag")
    assert "# contract-rag" in txt
    assert "benchmark.html" in txt
    assert meta.description in txt


def test_llms_txt_default_has_no_landing_but_always_has_repo_link():
    meta, _ = parse_front_matter(_FM)
    txt = llms_txt([meta], base_url="https://contractrag.com")
    assert "https://contractrag.com/): " not in txt  # no bare-root landing link
    assert "https://contractrag.com/zh/): " not in txt
    assert "[GitHub repository](https://github.com/qiangweihewu/contract-rag)" in txt


def test_llms_txt_include_landing_adds_both_language_landing_pages():
    meta, _ = parse_front_matter(_FM)
    txt = llms_txt([meta], base_url="https://contractrag.com", include_landing=True)
    assert "(https://contractrag.com/):" in txt
    assert "(https://contractrag.com/zh/):" in txt
    # landing links precede the article list
    assert txt.index("https://contractrag.com/):") < txt.index("benchmark.html")
    assert "[GitHub repository](https://github.com/qiangweihewu/contract-rag)" in txt


def test_json_ld_adds_publisher_always_and_dates_when_set():
    meta, _ = parse_front_matter(_FM)
    graph = json_ld(meta)
    article = next(n for n in graph["@graph"] if n["@type"] == "TechArticle")
    assert article["publisher"] == {"@type": "Organization", "name": "contract-rag",
                                     "url": "https://contractrag.com"}
    assert "datePublished" not in article  # no `date` in front matter -> omitted

    meta_dated = meta.model_copy(update={"date": "2026-07-09"})
    graph_dated = json_ld(meta_dated)
    article_dated = next(n for n in graph_dated["@graph"] if n["@type"] == "TechArticle")
    assert article_dated["datePublished"] == "2026-07-09"
    assert article_dated["dateModified"] == "2026-07-09"
    assert json.loads(json.dumps(graph_dated))  # still round-trips


def test_social_meta_has_og_and_twitter_tags_escaped_and_localized():
    tags = social_meta(title='A "quoted" title', description="A description.",
                       url="https://contractrag.com/x.html", lang="en", og_type="article")
    assert 'property="og:title" content="A &quot;quoted&quot; title"' in tags
    assert 'property="og:description" content="A description."' in tags
    assert 'property="og:url" content="https://contractrag.com/x.html"' in tags
    assert 'property="og:site_name" content="contract-rag"' in tags
    assert 'property="og:type" content="article"' in tags
    assert 'property="og:locale" content="en_US"' in tags
    assert '<meta name="twitter:card" content="summary">' in tags

    zh_tags = social_meta(title="T", description="D", url="https://contractrag.com/zh/x.html",
                          lang="zh", og_type="website")
    assert 'property="og:locale" content="zh_CN"' in zh_tags
    assert 'property="og:type" content="website"' in zh_tags
