from __future__ import annotations

from pathlib import Path

import pytest

from contract_rag.benchmark.core import run_nda_benchmark
from contract_rag.site.builder import (
    analytics_snippet,
    benchmark_tokens,
    build_site,
    favicon_svg,
    load_static_tokens,
    robots_txt,
    sitemap_xml,
)
from contract_rag.site.models import PageMeta

_PAGE = '''+++
title = "T"
description = "D"
lang = "en"
slug = "benchmark"
canonical = "https://x.github.io/contract-rag/benchmark.html"
target_queries = ["why is my RAG returning garbage"]
[[faq]]
q = "Q?"
a = "A."
[[howto]]
step = "python -m contract_rag.benchmark"
+++
# Heading

Quality lift was {{ quality_lift }} and field-F1 went {{ f1_dirty }} to {{ f1_clean }}.
'''


def test_analytics_snippet():
    assert analytics_snippet(None) == ""
    assert analytics_snippet("") == ""
    snip = analytics_snippet("contractrag")
    assert 'data-goatcounter="https://contractrag.goatcounter.com/count"' in snip
    assert "gc.zgo.at/count.js" in snip
    # a full endpoint (self-hosted) is passed through verbatim
    assert 'data-goatcounter="https://stats.example.com/count"' in \
        analytics_snippet("https://stats.example.com/count")


def test_build_site_injects_analytics_when_code_set(tmp_path):
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text(_PAGE)
    out = tmp_path / "site_out"
    # unset: no beacon anywhere (byte-for-byte the analytics-free build)
    build_site(content, out, base_url="https://x.github.io/contract-rag",
               benchmark=run_nda_benchmark(seed=0))
    assert "goatcounter" not in (out / "benchmark.html").read_text()
    # set: the beacon lands in the article <head>
    build_site(content, out, base_url="https://x.github.io/contract-rag",
               benchmark=run_nda_benchmark(seed=0), analytics_code="contractrag")
    assert "contractrag.goatcounter.com/count" in (out / "benchmark.html").read_text()


def test_benchmark_tokens():
    tok = benchmark_tokens(run_nda_benchmark(seed=0))
    assert set(tok) >= {"quality_dirty", "quality_clean", "quality_lift",
                        "f1_dirty", "f1_clean", "f1_lift", "n_docs"}
    assert tok["quality_lift"].startswith(("+", "-", "0"))


def test_build_site_emits_geo_html(tmp_path):
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text(_PAGE)
    out = tmp_path / "site_out"
    written = build_site(content, out, base_url="https://x.github.io/contract-rag",
                         benchmark=run_nda_benchmark(seed=0))
    page = out / "benchmark.html"
    assert page in written
    html = page.read_text()
    assert '<script type="application/ld+json">' in html
    assert '<link rel="canonical"' in html
    assert "TechArticle" in html and "FAQPage" in html
    assert "{{" not in html  # every token substituted, none left dangling
    assert (out / "llms.txt").exists()
    assert (out / "index.html").exists()


_STATIC_PAGE = '''+++
title = "K"
description = "F1 reached {{ kleister_f1_improved }}."
lang = "en"
slug = "kleister"
[[faq]]
q = "Score?"
a = "It was {{ kleister_f1_improved }} at source-acc {{ kleister_source_acc }}."
[[howto]]
step = "Reproduce the {{ kleister_f1_improved }} run"
+++
# Heading

Initial {{ kleister_f1_initial }} improved to {{ kleister_f1_improved }}.
'''


def test_load_static_tokens(tmp_path):
    toml = tmp_path / "vals.toml"
    toml.write_text('kleister_f1_improved = "0.697"\nkleister_n_docs = "40"\n')
    assert load_static_tokens(toml) == {"kleister_f1_improved": "0.697",
                                        "kleister_n_docs": "40"}


def test_build_site_injects_static_tokens_in_body_and_front_matter(tmp_path):
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "kleister.en.md").write_text(_STATIC_PAGE)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://x.github.io/contract-rag",
               static_tokens={"kleister_f1_initial": "0.523",
                              "kleister_f1_improved": "0.697",
                              "kleister_source_acc": "1.0"})
    html = (out / "kleister.html").read_text()
    assert "0.523" in html and "0.697" in html
    assert "{{" not in html
    # FAQ answer + description flow into the JSON-LD / meta, substituted too
    assert "source-acc 1.0" in html
    assert "F1 reached 0.697." in html
    # HowTo steps flow into the JSON-LD too, substituted as well
    assert "Reproduce the 0.697 run" in html


def test_real_content_dir_builds_with_all_tokens_resolved(tmp_path):
    """The committed articles (benchmark live-injected + kleister static-injected)
    must build with every `{{ token }}` resolved."""
    pytest.importorskip("markdown")
    repo = Path(__file__).resolve().parent.parent
    content = repo / "content"
    tokens: dict[str, str] = {}
    for path in sorted(content.glob("*.toml")):
        tokens.update(load_static_tokens(path))
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://x.github.io/contract-rag",
               benchmark=run_nda_benchmark(seed=0), static_tokens=tokens)
    for page in sorted(out.glob("*.html")):         # every article, both langs
        assert "{{" not in page.read_text(), page.name
    for page in ("kleister-nda.html", "kleister-nda.zh.html"):
        html = (out / page).read_text()
        assert "0.697" in html and "0.523" in html   # headline Kleister numbers
        assert "0/40" in html and "12/40" in html    # structured-decoding result
        assert "2026-07-06" in html                  # point-in-time statement
    for page in ("benchmark.html", "benchmark.zh.html"):
        assert "{{" not in (out / page).read_text(), page


def test_robots_txt_allows_all_and_invites_ai_crawlers():
    text = robots_txt(base_url="https://contractrag.com")
    assert "User-agent: *" in text
    assert "Allow: /" in text
    for ua in ("GPTBot", "ClaudeBot", "Claude-Web", "PerplexityBot", "Google-Extended"):
        assert f"User-agent: {ua}" in text
    assert "Sitemap: https://contractrag.com/sitemap.xml" in text


def test_robots_txt_strips_trailing_slash_from_base_url():
    text = robots_txt(base_url="https://contractrag.com/")
    assert "Sitemap: https://contractrag.com/sitemap.xml" in text
    assert "//sitemap.xml" not in text


def test_sitemap_xml_lists_pages_with_fixed_lastmod():
    pages = [
        PageMeta(title="A", description="d", lang="en", slug="a",
                 canonical="https://contractrag.com/a.html"),
        PageMeta(title="B", description="d", lang="zh", slug="b",
                 canonical="https://contractrag.com/b.html"),
    ]
    xml = sitemap_xml(pages, now="2026-07-09")
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<urlset" in xml
    assert "<loc>https://contractrag.com/a.html</loc>" in xml
    assert "<loc>https://contractrag.com/b.html</loc>" in xml
    assert xml.count("<lastmod>2026-07-09</lastmod>") == 2


def test_build_site_emits_robots_and_sitemap_deterministically(tmp_path):
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text(_PAGE)
    out = tmp_path / "site_out"
    written = build_site(content, out, base_url="https://contractrag.com",
                         benchmark=run_nda_benchmark(seed=0), now="2026-07-09")
    robots_path = out / "robots.txt"
    sitemap_path = out / "sitemap.xml"
    assert robots_path in written
    assert sitemap_path in written
    robots = robots_path.read_text()
    assert "User-agent: GPTBot" in robots
    assert "Sitemap: https://contractrag.com/sitemap.xml" in robots
    sitemap = sitemap_path.read_text()
    assert "<loc>https://contractrag.com/benchmark.html</loc>" in sitemap
    assert "<lastmod>2026-07-09</lastmod>" in sitemap
    # deterministic: re-building with the same fixed `now` reproduces byte-identical output
    out2 = tmp_path / "site_out2"
    build_site(content, out2, base_url="https://contractrag.com",
              benchmark=run_nda_benchmark(seed=0), now="2026-07-09")
    assert (out2 / "sitemap.xml").read_text() == sitemap


def test_build_site_now_defaults_when_omitted(tmp_path):
    """Without an explicit `now`, build_site still emits a sitemap (CLI convenience) —
    just not byte-deterministic across real calendar days."""
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    content.mkdir()
    (content / "benchmark.en.md").write_text(_PAGE)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://x.github.io/contract-rag",
              benchmark=run_nda_benchmark(seed=0))
    sitemap = (out / "sitemap.xml").read_text()
    assert "<lastmod>" in sitemap and "</lastmod>" in sitemap


def test_sitemap_xml_uses_page_date_when_set_else_now():
    pages = [
        PageMeta(title="A", description="d", lang="en", slug="a",
                 canonical="https://contractrag.com/a.html", date="2026-01-01"),
        PageMeta(title="B", description="d", lang="en", slug="b",
                 canonical="https://contractrag.com/b.html"),
    ]
    xml = sitemap_xml(pages, now="2026-07-09")
    assert "<loc>https://contractrag.com/a.html</loc><lastmod>2026-01-01</lastmod>" in xml
    assert "<loc>https://contractrag.com/b.html</loc><lastmod>2026-07-09</lastmod>" in xml


# --- items 1/2/5/6: hreflang pairing, OG/Twitter tags, favicon, research nav --

_EN_A = '''+++
title = "Article A"
description = "Desc A"
lang = "en"
slug = "a"
date = "2026-01-01"
+++
# A

Body A.
'''

_ZH_A = '''+++
title = "文章 A"
description = "描述 A"
lang = "zh"
slug = "a.zh"
date = "2026-01-01"
+++
# A zh

正文 A。
'''

_EN_B_NO_ZH = '''+++
title = "Article B"
description = "Desc B"
lang = "en"
slug = "b"
+++
# B

Body B.
'''

_ZH_C_NO_EN = '''+++
title = "文章 C"
description = "描述 C"
lang = "zh"
slug = "c.zh"
+++
# C zh

正文 C。
'''


def _write_multi_lang_content(content_dir: Path) -> None:
    content_dir.mkdir(parents=True, exist_ok=True)
    (content_dir / "a.en.md").write_text(_EN_A)
    (content_dir / "a.zh.md").write_text(_ZH_A)
    (content_dir / "b.en.md").write_text(_EN_B_NO_ZH)
    (content_dir / "c.zh.md").write_text(_ZH_C_NO_EN)


def test_hreflang_pairs_only_and_x_default(tmp_path):
    """Item 1 (bug fix): an article's hreflang block lists ONLY its own
    en/zh pair, not every article on the site, plus an x-default pointing at
    the en version (or itself, when there is no en counterpart)."""
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    _write_multi_lang_content(content)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://contractrag.com", now="2026-07-09")

    html_a_en = (out / "a.html").read_text()
    assert html_a_en.count('rel="alternate" hreflang="en"') == 1
    assert html_a_en.count('rel="alternate" hreflang="zh"') == 1
    assert html_a_en.count('hreflang="x-default"') == 1
    assert '<link rel="alternate" hreflang="en" href="https://contractrag.com/a.html">' in html_a_en
    assert '<link rel="alternate" hreflang="zh" href="https://contractrag.com/a.zh.html">' in html_a_en
    assert '<link rel="alternate" hreflang="x-default" href="https://contractrag.com/a.html">' in html_a_en
    # not every article on the site — b/c must not leak in
    assert "b.html" not in html_a_en.split("<script")[0]
    assert "c.zh.html" not in html_a_en.split("<script")[0]

    html_a_zh = (out / "a.zh.html").read_text()
    assert html_a_zh.count('rel="alternate" hreflang="en"') == 1
    assert html_a_zh.count('rel="alternate" hreflang="zh"') == 1
    assert html_a_zh.count('hreflang="x-default"') == 1
    # x-default always points at the en version, even from the zh page
    assert '<link rel="alternate" hreflang="x-default" href="https://contractrag.com/a.html">' in html_a_zh

    # b has no zh counterpart: just itself (en) + x-default (2 links total)
    html_b = (out / "b.html").read_text()
    assert html_b.count('rel="alternate" hreflang="en"') == 1
    assert 'hreflang="zh"' not in html_b
    assert '<link rel="alternate" hreflang="x-default" href="https://contractrag.com/b.html">' in html_b

    # c has no en counterpart: just itself (zh) + x-default pointing at itself
    html_c = (out / "c.zh.html").read_text()
    assert 'hreflang="en"' not in html_c
    assert html_c.count('rel="alternate" hreflang="zh"') == 1
    assert '<link rel="alternate" hreflang="x-default" href="https://contractrag.com/c.zh.html">' in html_c


def test_build_site_articles_have_og_and_twitter_tags(tmp_path):
    """Item 2: every article page gets Open Graph + Twitter Card meta, with
    og:type=article and og:locale derived from the page's own language."""
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    _write_multi_lang_content(content)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://contractrag.com", now="2026-07-09")

    html_en = (out / "a.html").read_text()
    assert 'property="og:title" content="Article A"' in html_en
    assert 'property="og:description" content="Desc A"' in html_en
    assert 'property="og:url" content="https://contractrag.com/a.html"' in html_en
    assert 'property="og:site_name" content="contract-rag"' in html_en
    assert 'property="og:type" content="article"' in html_en
    assert 'property="og:locale" content="en_US"' in html_en
    assert '<meta name="twitter:card" content="summary">' in html_en

    html_zh = (out / "a.zh.html").read_text()
    assert 'property="og:locale" content="zh_CN"' in html_zh
    assert 'property="og:type" content="article"' in html_zh


def test_build_site_datepublished_flows_into_json_ld(tmp_path):
    """Item 3: a `date` front-matter field produces datePublished/dateModified
    in the article's TechArticle JSON-LD; b has no `date`, so neither key appears."""
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    _write_multi_lang_content(content)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://contractrag.com", now="2026-07-09")

    html_a = (out / "a.html").read_text()
    assert '"datePublished": "2026-01-01"' in html_a
    assert '"dateModified": "2026-01-01"' in html_a
    assert '"publisher": {"@type": "Organization"' in html_a

    html_b = (out / "b.html").read_text()
    assert "datePublished" not in html_b
    assert '"publisher": {"@type": "Organization"' in html_b  # publisher is unconditional


def test_build_site_writes_favicon(tmp_path):
    """Item 5: build_site emits a self-contained favicon.svg, linked from every
    page's <head>."""
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    _write_multi_lang_content(content)
    out = tmp_path / "site_out"
    written = build_site(content, out, base_url="https://contractrag.com", now="2026-07-09")

    favicon_path = out / "favicon.svg"
    assert favicon_path in written
    svg = favicon_path.read_text()
    assert svg == favicon_svg()
    assert svg.startswith("<svg")
    assert "#1a7f5e" in svg  # matches the landing page's --accent

    for name in ("a.html", "a.zh.html", "b.html"):
        assert '<link rel="icon" type="image/svg+xml" href="/favicon.svg">' in (out / name).read_text()


def test_build_site_related_research_nav_is_language_scoped(tmp_path):
    """Item 6: the 'More research' / '更多研究' footer nav on an article page
    lists only OTHER articles in that page's OWN language, root-absolute."""
    pytest.importorskip("markdown")
    content = tmp_path / "content"
    _write_multi_lang_content(content)
    out = tmp_path / "site_out"
    build_site(content, out, base_url="https://contractrag.com", now="2026-07-09")

    html_a_en = (out / "a.html").read_text()
    assert "More research" in html_a_en
    assert '<a href="https://contractrag.com/b.html">Article B</a>' in html_a_en
    assert "文章" not in html_a_en  # no zh-titled article surfaced on the en page

    html_b_en = (out / "b.html").read_text()
    assert '<a href="https://contractrag.com/a.html">Article A</a>' in html_b_en

    html_a_zh = (out / "a.zh.html").read_text()
    assert "更多研究" in html_a_zh
    assert '<a href="https://contractrag.com/c.zh.html">文章 C</a>' in html_a_zh
    nav_zh = html_a_zh[html_a_zh.index('<nav class="research-nav">'):]
    assert "Article" not in nav_zh  # no en-titled article surfaced on the zh page's nav

    # a single-article language family (no other same-language articles) gets
    # no <nav> at all, not an empty one
    solo_content = tmp_path / "content_solo"
    solo_content.mkdir()
    (solo_content / "a.en.md").write_text(_EN_A)
    solo_out = tmp_path / "site_out_solo"
    build_site(solo_content, solo_out, base_url="https://contractrag.com", now="2026-07-09")
    assert '<nav class="research-nav">' not in (solo_out / "a.html").read_text()
